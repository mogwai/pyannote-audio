#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

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

import os.path

import keras.backend as K
from keras.models import Sequential
from keras.models import Model

from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import merge
from keras.layers.pooling import GlobalAveragePooling1D

from pyannote.audio.callback import LoggingCallback
from keras.models import model_from_yaml


class SequenceEmbedding(object):
    """Base class for sequence embedding

    Parameters
    ----------
    log_dir: str, optional
        When provided, log status after each epoch into this directory. This
        will create several files, including loss plots and weights files.
    """
    def __init__(self, log_dir=None):
        super(SequenceEmbedding, self).__init__()
        self.log_dir = log_dir

    @classmethod
    def from_disk(cls, architecture, weights):
        """Load pre-trained sequence embedding from disk

        Parameters
        ----------
        architecture : str
            Path to architecture file (e.g. created by `to_disk` method)
        weights : str
            Path to pre-trained weight file (e.g. created by `to_disk` method)

        Returns
        -------
        sequence_embedding : SequenceEmbedding
            Pre-trained sequence embedding model.
        """
        self = SequenceEmbedding()

        with open(architecture, 'r') as fp:
            yaml_string = fp.read()
        self.embedding_ = model_from_yaml(yaml_string,
                                          custom_objects=custom_objects)
        self.embedding_.load_weights(weights)
        return self

    def to_disk(self, architecture=None, weights=None, overwrite=False, input_shape=None, model=None):
        """Save trained sequence embedding to disk

        Parameters
        ----------
        architecture : str, optional
            When provided, path where to save architecture.
        weights : str, optional
            When provided, path where to save weights
        overwrite : boolean, optional
            Overwrite (architecture or weights) file in case they exist.
        """

        if not hasattr(self, 'model_'):
            raise AttributeError('Model must be trained first.')

        if architecture and os.path.isfile(architecture) and not overwrite:
            raise ValueError("File '{architecture}' already exists.".format(architecture=architecture))

        if weights and os.path.isfile(weights) and not overwrite:
            raise ValueError("File '{weights}' already exists.".format(weights=weights))

        embedding = self.get_embedding(self.model_)

        if architecture:
            yaml_string = embedding.to_yaml()
            with open(architecture, 'w') as fp:
                fp.write(yaml_string)

        if weights:
            embedding.save_weights(weights, overwrite=overwrite)

    def loss(self, y_true, y_pred):
        raise NotImplementedError('')

    def fit(self, input_shape, generator,
            samples_per_epoch, nb_epoch, callbacks=[]):
        """Train model

        Parameters
        ----------
        input_shape :
        generator :
        samples_per_epoch :
        np_epoch :
        callbacks :
        """

        if not callbacks and self.log_dir:
            default_callback = LoggingCallback(self, log_dir=self.log_dir)
            callbacks = [default_callback]

        self.model_ = self.design_model(input_shape)
        self.model_.compile(optimizer=self.optimizer,
                            loss=self.loss)

        self.model_.fit_generator(
            generator, samples_per_epoch, nb_epoch,
            verbose=1, callbacks=callbacks)

    def transform(self, sequence, batch_size=32, verbose=0):
        if not hasattr(self, 'embedding_'):
            self.embedding_ = self.get_embedding(self.model_)

        return self.embedding_.predict(
            sequence, batch_size=batch_size, verbose=verbose)


class TristouNet(object):
    """TristouNet sequence embedding

    Reference
    ---------
    Hervé Bredin, "TristouNet: Triplet Loss for Speaker Turn Embedding"
    Submitted to ICASSP 2017.
    https://arxiv.org/abs/1609.04301

    Parameters
    ----------
    lstm: list, optional
        List of output dimension of stacked LSTMs.
        Defaults to [16, ] (i.e. one LSTM with output dimension 16)
    bidirectional: boolean, optional
        When True, use bi-directional LSTMs
    pooling: {'last', 'average'}
        By default ('last'), only the last output of the last LSTM layer is
        returned. Use 'average' pooling if you want the last LSTM layer to
        return the whole sequence and take the average.
    dense: list, optional
        Number of units of additionnal stacked dense layers.
        Defaults to [16, ] (i.e. add one dense layer with 16 units)
    output_dim: int, optional
        Embedding dimension. Defaults to 16
    space: {'sphere', 'quadrant'}, optional
        When 'sphere' (resp. 'quadrant'), use 'tanh' (resp. 'sigmoid') as
        final activation. Defaults to 'sphere'.
    """

    def __init__(self, lstm=[16,], bidirectional=True, pooling='average',
                 dense=[16,], output_dim=16, space='sphere'):

        self.lstm = lstm
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.dense = dense
        self.output_dim = output_dim
        self.space = space

    def __call__(self, input_shape):
        """

        Parameters
        ----------
        input_shape : (n_frames, n_features) tuple
            Shape of input sequence.

        Returns
        -------
        model : Keras model

        """

        inputs = Input(shape=input_shape,
                       name="embedding_input")
        x = inputs

        # stack LSTM layers
        n_lstm = len(self.lstm)
        for i, output_dim in enumerate(self.lstm):

            if self.pooling == 'last':
                # only last LSTM should not return a sequence
                return_sequences = i+1 < n_lstm
            elif self.pooling == 'average':
                return_sequences = True
            else:
                raise NotImplementedError(
                    'unknown "{pooling}" pooling'.format(pooling=self.pooling))

            if i:
                # all but first LSTM
                forward = LSTM(output_dim=output_dim,
                               return_sequences=return_sequences,
                               activation='tanh',
                               dropout_W=0.0,
                               dropout_U=0.0)(forward)
                if self.bidirectional:
                    backward = LSTM(output_dim=output_dim,
                                    return_sequences=return_sequences,
                                    activation='tanh',
                                    dropout_W=0.0,
                                    dropout_U=0.0)(backward)
            else:
                # first forward LSTM needs to be given the input shape
                forward = LSTM(input_shape=input_shape,
                               output_dim=output_dim,
                               return_sequences=return_sequences,
                               activation='tanh',
                               dropout_W=0.0,
                               dropout_U=0.0)(x)
                if self.bidirectional:
                    # first backward LSTM needs to be given the input shape
                    # AND to be told to process the sequence backward
                    backward = LSTM(go_backwards=True,
                                    input_shape=input_shape,
                                    output_dim=output_dim,
                                    return_sequences=return_sequences,
                                    activation='tanh',
                                    dropout_W=0.0,
                                    dropout_U=0.0)(x)

        if self.pooling == 'average':
            forward = GlobalAveragePooling1D()(forward)
            if self.bidirectional:
                backward = GlobalAveragePooling1D()(backward)

        # concatenate forward and backward
        if self.bidirectional:
            x = merge([forward, backward], mode='concat', concat_axis=1)
        else:
            x = forward

        # stack dense layers
        for i, output_dim in enumerate(self.dense):
            x = Dense(output_dim, activation='tanh')(x)

        # stack final dense layer
        if self.space == 'sphere':
            activation = 'tanh'
        elif self.space == 'quadrant':
            activation = 'sigmoid'
        x = Dense(self.output_dim, activation=activation)(x)

        # stack L2 normalization layer
        embeddings = Lambda(lambda x: K.l2_normalize(x, axis=-1),
                            name="embedding_output")(x)

        return Model(input=inputs, output=embeddings)


class TripletLossSequenceEmbedding(SequenceEmbedding):
    """Triplet loss for sequence embedding

    Reference
    ---------
    Hervé Bredin, "TristouNet: Triplet Loss for Speaker Turn Embedding"
    Submitted to ICASSP 2017. https://arxiv.org/abs/1609.04301

    Parameters
    ----------
    design_embedding : callable, or func
        This function should take input_shape as input and return a Keras model
        (see TristouNet.__call__ for an example)
    optimizer: str, optional
        Keras optimizer. Defaults to 'rmsprop'.
    log_dir: str, optional
        When provided, log status after each epoch into this directory. This
        will create several files, including loss plots and weights files.
    """
    def __init__(self, design_embedding, margin=0.2, optimizer='rmsprop', log_dir=None):

        super(TripletLossSequenceEmbedding, self).__init__(log_dir=log_dir)

        self.design_embedding = design_embedding
        self.margin = margin
        self.optimizer = optimizer

    def _triplet_loss(self, inputs):
        p = K.sum(K.square(inputs[0] - inputs[1]), axis=-1, keepdims=True)
        n = K.sum(K.square(inputs[0] - inputs[2]), axis=-1, keepdims=True)
        return K.maximum(0, p + self.margin - n)

    @staticmethod
    def _output_shape(input_shapes):
        return (input_shapes[0][0], 1)

    @staticmethod
    def _identity_loss(y_true, y_pred):
        return K.mean(y_pred - 0 * y_true)

    def loss(self, y_true, y_pred):
        return self._identity_loss(y_true, y_pred)

    def get_embedding(self, model):
        """Extract embedding from Keras model (a posteriori)"""
        return model.layers_by_depth[1][0]

    def design_model(self, input_shape):
        """
        Parameters
        ----------
        input_shape: (n_samples, n_features) tuple
            Shape of input sequences.
        """

        anchor = Input(shape=input_shape, name="anchor")
        positive = Input(shape=input_shape, name="positive")
        negative = Input(shape=input_shape, name="negative")

        embedding = self.design_embedding(input_shape)
        embedded_anchor = embedding(anchor)
        embedded_positive = embedding(positive)
        embedded_negative = embedding(negative)

        distance = merge(
            [embedded_anchor, embedded_positive, embedded_negative],
            mode=self._triplet_loss, output_shape=self._output_shape)

        model = Model(input=[anchor, positive, negative], output=distance)

        return model
