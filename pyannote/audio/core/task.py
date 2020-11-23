# MIT License
#
# Copyright (c) 2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, List, Optional, Text, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader, IterableDataset
from torch_audiomentations.augmentations.background_noise import ApplyBackgroundNoise
from torchaudio.transforms import Resample

from pyannote.audio.transforms.pipeline import TransformsPipe
from pyannote.audio.transforms.transforms import Reverb
from pyannote.database import Protocol

if TYPE_CHECKING:
    from pyannote.audio.core.model import Model


def _override_bad_defaults(kwargs):
    "Fix bad default for spectrograms"

    if "n_fft" not in kwargs or kwargs["n_fft"] is None:
        kwargs["n_fft"] = 1024
    if "win_length" not in kwargs or kwargs["win_length"] is None:
        kwargs["win_length"] = kwargs["n_fft"]
    if "hop_length" not in kwargs or kwargs["hop_length"] is None:
        kwargs["hop_length"] = int(kwargs["win_length"] / 2)
    return kwargs


# Type of machine learning problem
class Problem(Enum):
    BINARY_CLASSIFICATION = 0
    MONO_LABEL_CLASSIFICATION = 1
    MULTI_LABEL_CLASSIFICATION = 2
    REPRESENTATION = 3
    REGRESSION = 4
    # any other we could think of?


# A task takes an audio chunk as input and returns
# either a temporal sequence of predictions
# or just one prediction for the whole audio chunk
class Scale(Enum):
    FRAME = 1  # model outputs a sequence of frames
    CHUNK = 2  # model outputs just one vector for the whole chunk


@dataclass
class TaskSpecification:
    problem: Problem
    scale: Scale

    # chunk duration in seconds.
    # use None for variable-length chunks
    duration: Optional[float] = None

    # (for classification tasks only) list of classes
    classes: Optional[List[Text]] = None

    def __len__(self):
        # makes it possible to do something like:
        # multi_task = len(task_specifications) > 1
        # because multi-task specifications are stored as {task_name: specifications} dict
        return 1

    def items(self):
        yield None, self


class TrainDataset(IterableDataset):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task

    def __iter__(self):
        return self.task.train__iter__()

    def __len__(self):
        return self.task.train__len__()


class ValDataset(IterableDataset):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task

    def __iter__(self):
        return self.task.val__iter__()

    def __len__(self):
        return self.task.val__len__()


class Task(pl.LightningDataModule):
    """Base task class

    A task is the combination of a "problem" and a "dataset".
    For example, here are a few tasks:
    - voice activity detection on the AMI corpus
    - speaker embedding on the VoxCeleb corpus
    - end-to-end speaker diarization on the VoxConverse corpus

    A task is expected to be solved by a "model" that takes an
    audio chunk as input and returns the solution. Hence, the
    task is in charge of generating (input, expected_output)
    samples used for training the model.

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
    duration : float, optional
        Chunks duration. Defaults to variable duration (None).
    batch_size : int, optional
        Number of training samples per batch.
    num_workers : int, optional
        Number of workers used for generating training samples.
    pin_memory : bool, optional
        If True, data loaders will copy tensors into CUDA pinned
        memory before returning them. See pytorch documentation
        for more details. Defaults to False.

    Attributes
    ----------
    specifications : TaskSpecification or dict of TaskSpecification
        Task specifications (available after `Task.setup` has been called.)
        For multi-task learning, this should be a dictionary where keys are
        task names and values are corresponding TaskSpecification instances.
    """

    def __init__(
        self,
        protocol: Protocol,
        duration: float = None,
        batch_size: int = None,
        num_workers: int = 1,
        pin_memory: bool = False,
        gpu_transforms: bool = True,
        transforms: Union[List[torch.nn.Module], TransformsPipe] = [],
    ):
        super().__init__()

        self.gpu = gpu_transforms and torch.cuda.is_available()

        # dataset
        self.protocol = protocol

        # batching
        self.duration = duration
        self.batch_size = batch_size

        # multi-processing
        if (
            num_workers > 0
            and sys.platform == "darwin"
            and sys.version_info[0] >= 3
            and sys.version_info[1] >= 8
        ):
            warnings.warn(
                "num_workers > 0 is not supported with macOS and Python 3.8+: "
                "setting num_workers = 0."
            )
            num_workers = 0

        self.num_workers = num_workers

        self.pin_memory = pin_memory
        self._waveform_transforms = None

    @property
    def waveform_transforms(self) -> TransformsPipe:
        """Basic transforms for waveforms


        Parameters
        ---------
        task : Task
            The task that need transformations to be applied to
        bg_noise: Path, str
            Where to find background noise data

        Returns
        -------

        TransformsPipe
        """
        if self._waveform_transforms is None:
            sr = self.audio.sample_rate
            augs = []
            reverb = Reverb(sample_rate=sr)

            # Test and validation augmentations will be applied without the
            # random transformations
            augs += [Resample(sr), reverb]

            # Setup Background Noise
            if self.bg_noise:
                augs.append(ApplyBackgroundNoise(self.bg_noise))

            self._waveform_transforms = TransformsPipe(*augs)
        return self._waveform_transforms

    @waveform_transforms.setter
    def waveform_transforms(self, v: Union[List, TransformsPipe]):
        if isinstance(v, list):
            v = TransformsPipe(*v)
        self._waveform_transforms = v

    def _setup_transforms(self, model: Model, stage=None):
        tfms = (
            self.waveform_transforms
            if stage != "test"
            else self.waveform_transforms.validate_pipe()
        )
        if self.gpu:
            model.waveform_transforms = tfms

        # Support loading the waveforms from workers
        # in the dataloader if we can't do on the GPU
        if not self.gpu:
            old_func = self.train__iter__

            def _new_iter_with_tfms():
                it = old_func()
                while True:
                    batch = next(it)
                    batch["X"] = tfms(batch["X"])
                    yield batch

            self.train__iter__ = _new_iter_with_tfms

    def prepare_data(self):
        """Use this to download and prepare data

        This is where we might end up downloading datasets
        and transform them so that they are ready to be used
        with pyannote.database. but for now, the API assume
        that we directly provide a pyannote.database.Protocol.

        Notes
        -----
        Called only once.
        """
        pass

    def setup(self, stage=None):
        """Called at the beginning of fit and test just before Model.setup()

        Parameters
        ----------
        stage : "fit" or "test"
            Whether model is being trained ("fit") or used for inference ("test").

        Notes
        -----
        This hook is called on every process when using DDP.

        If `specifications` attribute has not been set in `__init__`,
        `setup` is your last chance to set it.

        """
        pass

    @cached_property
    def is_multi_task(self) -> bool:
        """"Check whether multiple tasks are addressed at once"""
        return len(self.specifications) > 1

    def train__iter__(self):
        # will become train_dataset.__iter__ method
        msg = f"Missing '{self.__class__.__name__}.train__iter__' method."
        raise NotImplementedError(msg)

    def train__len__(self):
        # will become train_dataset.__len__ method
        msg = f"Missing '{self.__class__.__name__}.train__len__' method."
        raise NotImplementedError(msg)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            TrainDataset(self),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    @cached_property
    def example_input_duration(self) -> float:
        return 2.0 if self.duration is None else self.duration

    @cached_property
    def example_input_array(self):
        # this method is called in Model.introspect where it is used
        # to automagically infer the temporal resolution of the
        # model output, and hence allow the dataloader to shape
        # its targets correctly.

        # since we plan to have the feature extraction step done
        # on GPU as part of the model, the example input array is
        # basically always a chunk of audio

        if self.audio.mono:
            num_channels = 1
        else:
            msg = "Only 'mono' audio is supported."
            raise NotImplementedError(msg)

        return torch.randn(
            (
                self.batch_size,
                num_channels,
                int(self.audio.sample_rate * self.example_input_duration),
            )
        )

    def default_loss(
        self, specifications: TaskSpecification, y, y_pred
    ) -> torch.Tensor:
        """Guess and compute default loss according to task specification"""

        if specifications.problem == Problem.BINARY_CLASSIFICATION:
            loss = F.binary_cross_entropy(y_pred.squeeze(dim=-1), y.float())

        elif specifications.problem == Problem.MONO_LABEL_CLASSIFICATION:
            loss = F.nll_loss(y_pred.view(-1, len(specifications.classes)), y.view(-1))

        elif specifications.problem == Problem.MULTI_LABEL_CLASSIFICATION:
            loss = F.binary_cross_entropy(y_pred, y.float())

        else:
            msg = "TODO: implement for other types of problems"
            raise NotImplementedError(msg)

        return loss

    # default training_step provided for convenience
    # can obviously be overriden for each task
    def training_step(self, model: Model, batch, batch_idx: int):
        """Default training_step according to task specification

            * binary cross-entropy loss for binary or multi-label classification
            * negative log-likelihood loss for regular classification

        In case of multi-tasking, it will default to summing loss of each task.

        Parameters
        ----------
        model : Model
            Model currently being trained.
        batch : (usually) dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.

        Returns
        -------
        loss : {str: torch.tensor}
            {"loss": loss} with additional "loss_{task_name}" keys for multi-task models.
        """

        X, y = batch["X"], batch["y"]
        y_pred = model(X)

        if self.is_multi_task:
            loss = dict()
            for task_name, specifications in self.specifications.items():
                loss[task_name] = self.default_loss(
                    specifications, y[task_name], y_pred[task_name]
                )
                model.log(f"{task_name}_train_loss", loss[task_name])

            loss["loss"] = sum(loss.values())
            model.log("train_loss", loss["loss"])
            return loss

        loss = self.default_loss(self.specifications, y, y_pred)
        model.log("train_loss", loss)
        return {"loss": loss}

    def val__iter__(self):
        # will become val_dataset.__iter__ method
        msg = f"Missing '{self.__class__.__name__}.val__iter__' method."
        raise NotImplementedError(msg)

    def val__len__(self):
        # will become val_dataset.__len__ method
        msg = f"Missing '{self.__class__.__name__}.val__len__' method."
        raise NotImplementedError(msg)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            ValDataset(self),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    # default validation_step provided for convenience
    # can obviously be overriden for each task
    def validation_step(self, model: Model, batch, batch_idx: int):
        """Guess default validation_step according to task specification

            * binary cross-entropy loss for binary or multi-label classification
            * negative log-likelihood loss for regular classification

        In case of multi-tasking, it will default to summing loss of each task.

        Parameters
        ----------
        model : Model
            Model currently being validated.
        batch : (usually) dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.

        Returns
        -------
        loss : {str: torch.tensor}
            {"loss": loss} with additional "loss_{task_name}" keys for multi-task models.
        """

        X, y = batch["X"], batch["y"]
        y_pred = model(X)

        if self.is_multi_task:
            loss = dict()
            for task_name, specifications in self.specifications.items():
                loss[task_name] = self.default_loss(
                    specifications, y[task_name], y_pred[task_name]
                )
                model.log(f"{task_name}_val_loss", loss[task_name])

            loss["loss"] = sum(loss.values())
            model.log("val_loss", loss["loss"])
            return loss

        loss = self.default_loss(self.specifications, y, y_pred)
        model.log("val_loss", loss)
        return {"loss": loss}

    # default configure_optimizers provided for convenience
    # can obviously be overriden for each task
    def configure_optimizers(self, model: Model):
        # for tasks such as SpeakerEmbedding,
        # other parameters should be added here
        return torch.optim.Adam(model.parameters(), lr=1e-3)
