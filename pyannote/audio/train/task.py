#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019-2020 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

"""
Tasks
#####

"""


from typing import List, Dict, Text, Type, Union, Tuple
from pathlib import Path

from enum import Enum
import multiprocessing

from pyannote.database import Protocol
from pyannote.database import ProtocolFile
from pyannote.database import Preprocessors
from pyannote.database import Subset
from pyannote.database.custom import gather_loaders

from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
from argparse import Namespace

import math
import pickle
import numpy as np
from pyannote.core.utils.helper import get_class_by_name
from pyannote.core import Segment

from tqdm import tqdm
import warnings


class Resolution(Enum):
    FRAME = "frame"
    CHUNK = "chunk"


class Problem(Enum):
    """Type of machine learning problem

    Used to automatically suggest reasonable default final activation layer
    and loss function.
    """

    MULTI_CLASS_CLASSIFICATION = "classification"
    MULTI_LABEL_CLASSIFICATION = "multi-label classification"
    REGRESSION = "regression"
    REPRESENTATION = "representation"


class BaseTask(pl.LightningModule):
    @staticmethod
    def load_config(
        config_yml: Path, hparams_yml: Path = None
    ) -> Tuple[Type["BaseTask"], Namespace, Preprocessors]:
        """Load and parse configuration file

        Parameters
        ----------
        config_yml : Path
            Path to configuration file
        hparams_yml : Path
            Path to Pytorch-lightning hyper-parameter file

        Returns
        -------
        task_class : type
            Task class (e.g. SpeechActivityDetection)
        hparams : Namespace
            Hyper-parameters.
        preprocessors : Preprocessors
            Preprocessors.
        """

        with open(config_yml, "r") as fp:
            configuration = yaml.load(fp, Loader=yaml.SafeLoader)

        task_section = configuration.pop("task")
        task_class = get_class_by_name(task_section["name"])

        if hparams_yml is not None:
            with open(hparams_yml, "r") as fp:
                configuration = yaml.load(fp, Loader=yaml.SafeLoader)

        for key, value in task_section["params"].items():
            configuration[key] = value

        # preprocessors
        preprocessors_section = configuration.pop("preprocessors", dict())
        preprocessors, loaders = dict(), dict()
        for key, preprocessor in preprocessors_section.items():
            #    key:
            #       name: package.module.ClassName
            #       params:
            #          param1: value1
            #          param2: value2
            if isinstance(preprocessor, dict):
                preprocessor_class = get_class_by_name(preprocessor["name"])
                preprocessors[key] = preprocessor_class(
                    **preprocessor.get("params", dict())
                )
            #    key: /path/to/file.suffix
            else:
                loaders[key] = preprocessor
        preprocessors.update(gather_loaders(loaders))

        hparams = Namespace(**configuration)
        return task_class, hparams, preprocessors

    def __init__(
        self,
        hparams: Union[Namespace, Dict],
        train_dir: Path = None,
        protocol: Protocol = None,
        subset: Subset = "train",
        num_workers: int = None,
    ):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        self.hparams = hparams
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
        self.num_workers = num_workers

        # AUGMENTATION
        if train_dir and "data_augmentation" in self.hparams:
            AugmentationClass = get_class_by_name(
                self.hparams.data_augmentation["name"]
            )
            augmentation = AugmentationClass(**self.hparams.data_augmentation["params"])

        else:
            augmentation = None

        # FEATURE EXTRACTION
        FeatureExtractionClass = get_class_by_name(
            self.hparams.feature_extraction["name"]
        )
        self.feature_extraction = FeatureExtractionClass(
            **self.hparams.feature_extraction["params"], augmentation=augmentation
        )

        if train_dir is not None:
            self.train_dir = train_dir

        if protocol is not None:
            self.protocol = protocol
            self.subset = subset

        # MODEL
        ArchitectureClass = get_class_by_name(self.hparams.architecture["name"])
        architecture_params = self.hparams.architecture.get("params", dict())
        self.model = ArchitectureClass(self, **architecture_params)

        # EXAMPLE INPUT ARRAY (used by Pytorch Lightning to display in and out
        # sizes of each layer, for a batch of size 5)
        if "duration" in self.hparams:
            duration = self.hparams.duration
            context = self.feature_extraction.get_context_duration()
            num_samples = math.ceil(
                (2 * context + duration) * self.feature_extraction.sample_rate
            )
            waveform = np.random.randn(num_samples, 1)
            self.example_input_array = torch.unsqueeze(
                torch.FloatTensor(
                    self.feature_extraction.crop(
                        {"waveform": waveform, "duration": 2 * context + duration},
                        Segment(context, context + duration),
                        mode="center",
                        fixed=duration,
                    ),
                ),
                0,
            ).repeat(5, 1, 1)

    def prepare_metadata(self, files: List[ProtocolFile]) -> Dict:
        return {"files": files}

    def prepare_data(self):

        path = self.train_dir / "train_metadata.pkl"
        if path.exists():
            msg = f"Using pre-existing '{path}' file."
            warnings.warn(msg)
            with open(path, "rb") as f:
                train_metadata = pickle.load(f)

        else:
            files = list(
                tqdm(
                    getattr(self.protocol, self.subset)(),
                    desc="Loading training protocol",
                    unit="file",
                )
            )
            train_metadata = self.prepare_metadata(files)
            with open(path, "wb") as f:
                pickle.dump(train_metadata, f)

        self.train_metadata = train_metadata

    def setup(self, stage: str) -> None:

        if stage != "fit":
            return

        path = self.train_dir / "train_metadata.pkl"
        with open(path, "rb") as f:
            self.train_metadata = pickle.load(f)
        self.hparams.classes = self.train_metadata["classes"]
        self.model.setup()

    def train_dataset(self) -> IterableDataset:
        raise NotImplementedError("")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset(),
            batch_size=self.hparams.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    # =========================================================================
    # LOSS FUNCTION
    # =========================================================================

    def guess_activation(self):

        if self.problem == Problem.MULTI_CLASS_CLASSIFICATION:
            return nn.LogSoftmax(dim=-1)

        elif self.problem == Problem.MULTI_LABEL_CLASSIFICATION:
            return nn.Sigmoid()

        elif self.problem == Problem.REGRESSION:
            return nn.Identity()

        elif self.problem == Problem.REPRESENTATION:
            return nn.Identity()

        else:
            msg = f"Unknown default activation for '{self.problem}' problems."
            raise NotImplementedError(msg)

    def get_activation(self):
        return self.guess_activation()

    def guess_loss(self):

        if (
            self.problem == Problem.MULTI_CLASS_CLASSIFICATION
            and self.resolution_output == Resolution.FRAME
        ):

            def loss(
                y_pred: torch.Tensor, y: torch.Tensor, weight=None
            ) -> torch.Tensor:
                return F.nll_loss(
                    y_pred.view((-1, len(self.hparams.classes))),
                    y.view((-1,)),
                    weight=weight,
                    reduction="mean",
                )

            return loss

        if (
            self.problem == Problem.MULTI_CLASS_CLASSIFICATION
            and self.resolution_output == Resolution.CHUNK
        ):

            def loss(
                y_pred: torch.Tensor, y: torch.Tensor, weight=None
            ) -> torch.Tensor:
                return F.nll_loss(y_pred, y, weight=weight, reduction="mean",)

            return loss

        msg = (
            f"Cannot guess loss function for {self.__class__.__name__}. "
            f"Please implement {self.__class__.__name__}.get_loss method."
        )
        raise NotImplementedError(msg)

    def get_loss(self):
        return self.guess_loss()

    @property
    def loss(self):
        if not hasattr(self, "loss_"):
            self.loss_ = self.get_loss()
        return self.loss_

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================

    def configure_optimizers(self):
        OptimizerClass = get_class_by_name(self.hparams.optimizer["name"])
        optimizer_params = self.hparams.optimizer.get("params", dict())
        optimizer = OptimizerClass(
            self.parameters(), lr=self.hparams.learning_rate, **optimizer_params,
        )

        return optimizer

    def forward(self, chunks: torch.Tensor) -> torch.Tensor:
        return self.model(chunks)

    def training_step(self, batch, batch_idx):
        X = batch["X"]
        y = batch["y"]
        y_pred = self(X)
        loss = self.loss(y_pred, y)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}
