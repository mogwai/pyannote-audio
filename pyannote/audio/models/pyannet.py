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


from typing import Dict, Union
from pyannote.core import SlidingWindow
from pyannote.core.utils.types import Alignment
from pyannote.audio.train.model import Model
from pyannote.audio.train.task import Resolution
from pyannote.audio.train.task import Problem

import torch
import torch.nn as nn
from .sincnet import SincNet
from .recurrent import Recurrent
from .pooling import Pooling
from .linear import Linear
from .scaling import Scaling


class PyanNet(Model):
    """waveform -> SincNet -> Recurrent [-> Pooling] -> Linear -> output

    Parameters
    ----------
    task : BaseTask
        Task addressed by the model.
    sincnet : `dict`, optional
        Configuration of SincNet layers.
        Use {'skip': True} to use handcrafted features instead of waveforms.
    recurrent : `dict`, optional
        Configuration of recurrent layers.
    pooling : dict, optional
        Configuration of pooling layer.
    linear : `dict`, optional
        Configuration of linear layers.
    scaling : `dict`, optional
        Configuration of scaling layers.
    """

    def init(
        self,
        sincnet: Dict = None,
        recurrent: Dict = None,
        pooling: Dict = None,
        linear: Dict = None,
        scaling: Dict = None,
    ):
        """Initialize layers

        Parameters
        ----------
        sincnet : `dict`, optional
            Configuration of SincNet layers.
            Use {'skip': True} to use handcrafted features instead of waveforms.
        recurrent : `dict`, optional
            Configuration of recurrent layers.
        pooling : dict, optional
            Configuration of pooling layer.
        linear : `dict`, optional
            Configuration of linear layers.
        scaling : `dict`, optional
            Configuration of scaling layers.
        """

        n_features = self.task.feature_extraction.dimension

        if sincnet is None:
            sincnet = dict()
        self.sincnet = sincnet

        if not sincnet.get("skip", False):
            if n_features != 1:
                msg = (
                    f"SincNet only supports mono waveforms. "
                    f"Here, waveform has {n_features} channels."
                )
                raise ValueError(msg)
            self.sincnet_ = SincNet(**sincnet)
            n_features = self.sincnet_.dimension

        if recurrent is None:
            recurrent = dict()
        self.recurrent = recurrent
        self.recurrent_ = Recurrent(n_features, **recurrent)
        n_features = self.recurrent_.dimension

        if pooling is None:
            pooling = dict()
        if self.task.resolution_output == Resolution.CHUNK and not pooling:
            msg = (
                f"Model should implement temporal poopling to address {self.task.__class__.__name__}. "
                f"Use 'pooling' parameter to add a pooling layer."
            )
            raise ValueError(msg)
        self.pooling = pooling
        self.pooling_ = Pooling(n_features, **pooling)
        n_features = self.pooling_.dimension

        if linear is None:
            linear = dict()
        self.linear = linear
        self.linear_ = Linear(n_features, **linear)
        n_features = self.linear_.dimension

        if self.task.problem == Problem.REPRESENTATION:
            if scaling is None:
                scaling = dict()
            self.scaling = scaling
            self.scaling_ = Scaling(n_features, **scaling)

        elif "classes" in self.task.hparams:
            self.classification_ = nn.Linear(
                n_features, len(self.task.hparams.classes), bias=True
            )

        self.final_activation_ = self.task.get_activation()

    def setup(self):
        if self.task.problem == Problem.REPRESENTATION:
            self.init_dimension()
            return 

        self.classification_ = nn.Linear(
            self.linear_.dimension, len(self.task.hparams.classes), bias=True
        )

    def forward(self, chunks: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        chunks : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of chunks. In case SincNet is skipped, a tensor with shape
            (batch_size, n_samples, n_features) is expected.

        Returns
        -------
        output : `torch.Tensor`
            Final network output.
        """

        if self.sincnet.get("skip", False):
            output = chunks
        else:
            output = self.sincnet_(chunks)

        output = self.recurrent_(output)
        output = self.pooling_(output)
        output = self.linear_(output)

        if self.task.problem == Problem.REPRESENTATION:
            output = self.scaling_(output)

        else:
            output = self.classification_(output)

        return self.final_activation_(output)

    def get_resolution(self) -> Union[Resolution, SlidingWindow]:
        if self.task.resolution_output == Resolution.CHUNK:
            return Resolution.CHUNK

        if self.sincnet.get("skip", False):
            return self.task.feature_extraction.sliding_window

        return self.sincnet_.get_resolution()

    def get_alignment(self) -> Alignment:
        if self.sincnet.get("skip", False):
            return "center"

        return self.sincnet_.get_alignment()

    def get_dimension(self) -> int:
        if self.task.problem == Problem.REPRESENTATION:
            return self.scaling_.dimension

        return len(self.task.hparams.classes)
