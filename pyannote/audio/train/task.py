#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

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

This module provides a `Task` class meant to specify machine learning tasks
(e.g. classification or regression).

This may be used to infer parts of the network architecture and the associated
loss function automatically.

Example
-------
>>> voice_activity_detection = Task(type=TaskType.MULTI_CLASS_CLASSIFICATION,
...                                 output=TaskOutput.SEQUENCE)
"""


from typing import Optional, List, Dict, Text, Type, Union
from typing import TYPE_CHECKING

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from enum import Enum

from pyannote.database import Protocol
from pyannote.database import ProtocolFile
from pyannote.database import Subset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace

import math
import numpy as np
from pyannote.core.utils.helper import get_class_by_name
from pyannote.core import Segment


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
    def __init__(
        self,
        hparams: Namespace,
        protocol: Protocol = None,
        subset: Subset = "train",
        files: List[ProtocolFile] = None,
    ):
        super().__init__()

        self.hparams = hparams

        # FEATURE EXTRACTION
        FeatureExtractionClass = get_class_by_name(
            self.hparams.feature_extraction["name"]
        )
        self.feature_extraction = FeatureExtractionClass(
            **self.hparams.feature_extraction["params"]
        )

        # TRAINING DATA
        if files is not None:
            self.files = files
        elif protocol is not None:
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
                torch.tensor(
                    self.feature_extraction.crop(
                        {"waveform": waveform, "duration": 2 * context + duration},
                        Segment(context, context + duration),
                        mode="center",
                        fixed=duration,
                    ),
                ),
                0,
            ).repeat(5, 1, 1)

    @property
    def files(self):
        if not hasattr(self, "files_"):
            # load protocol files once and for all
            if self.protocol is None:
                msg = f"No training protocol available. Please provide one."
                raise ValueError(msg)
            self.files_ = list(getattr(self.protocol, self.subset)())

        return self.files_

    @files.setter
    def files(self, files: List[ProtocolFile]):
        self.files_ = files

    def prepare_data(self):
        raise NotImplementedError("")

    def train_dataset(self) -> IterableDataset:
        raise NotImplementedError("")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset(), batch_size=self.hparams.batch_size,)

    # =========================================================================
    # CLASSES
    # =========================================================================

    @property
    def classes(self) -> List[Text]:
        """List of classes
        
        Used to automatically infer the output dimension of the model
        """
        if "classes" not in self.hparams:
            self.hparams.classes = self.get_classes()
        return self.hparams.classes

    def get_classes(self) -> List[Text]:
        """Compute list of classes
        
        Called when classes depend on the training data (e.g. for domain 
        classification experiments where we do not know in advance what
        domains are)
        """
        msg = f"Class {self.__class__.__name__} must define a 'get_classes' method."
        raise NotImplementedError(msg)

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

    @property
    def activation(self):
        if not hasattr(self, "activation_"):
            self.activation_ = self.get_activation()
        return self.activation_

    def guess_loss(self):

        if (
            self.problem == Problem.MULTI_CLASS_CLASSIFICATION
            and self.resolution_output == Resolution.FRAME
        ):

            def loss(
                y_pred: torch.Tensor, y: torch.Tensor, weight=None
            ) -> torch.Tensor:
                return F.nll_loss(
                    y_pred.view((-1, len(self.classes))),
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
        return self.activation(self.model(chunks))

    def training_step(self, batch, batch_idx):
        X = batch["X"]
        y = batch["y"]
        y_pred = self(X)
        loss = self.loss(y_pred, y)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}
