from enum import Enum

import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Normalizer

from export import get_features


class PCATransformer(PCA):
    def __init__(self, threshold=1., **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.number_of_components = None

    def fit(self, X, **kwargs):
        super().fit(X, **kwargs)
        explained_variance = np.cumsum(self.explained_variance_ratio_)
        self.number_of_components = (explained_variance <=
                                     self.threshold).sum()
        return self

    def transform(self, X):
        X = super().transform(X)
        return X[:, :self.number_of_components]


class FeatureSubsetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, indices=None):
        self.indices = indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.indices]


class AutoencoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, encoded_dim=None, layer_dims=None):
        self.encoded_dim = encoded_dim
        self.layer_dims = layer_dims
        self.encoder_layers = []
        self.decoder_layers = []
        self.input_dim = None
        self.input = None
        self.encoded = None
        self.decoded = None
        self.encoder = None

    def fit(self, X, y=None, epochs=100, batch_size=100, optimizer='adadelta',
            loss='binary_crossentropy'):
        self.input_dim = X.shape[1]
        self.input = Input(shape=(self.input_dim, ))
        if not self.layer_dims:
            self.encoded = Dense(self.encoded_dim,
                                 activation='relu')(self.input)
            self.decoded = Dense(self.input_dim,
                                 activation='sigmoid')(self.encoded)

        elif type(self.layer_dims) == int or (type(self.layer_dims) == list and
                                              len(self.layer_dims) == 1):
            if type(self.layer_dims) == list:
                self.layer_dims = self.layer_dims[0]
            self.encoder_layers.append(Dense(self.layer_dims,
                                             activation='relu')(self.input))
            self.encoded = Dense(self.encoded_dim,
                                 activation='relu')(self.encoder_layers[0])
            self.decoder_layers.append(Dense(self.layer_dims,
                                             activation='relu')(self.encoded))
            self.decoded = Dense(self.input_dim,
                                 activation='sigmoid')(self.decoder_layers[0])

        else:
            for i, layer_dim in enumerate(self.layer_dims):
                if not i:
                    self.encoder_layers.append(Dense(layer_dim,
                                                     activation='relu')
                                                    (self.input))
                    continue

                self.encoder_layers.append(Dense(layer_dim,
                                                 activation='relu')
                                                (self.encoder_layers[i - 1]))
                if i == len(self.layer_dims) - 1:
                    self.encoded = Dense(self.encoded_dim,
                                         activation='relu')(
                                             self.encoder_layers[i])

            for i, layer_dim in enumerate(self.layer_dims[::-1]):
                if not i:
                    self.decoder_layers.append(Dense(layer_dim,
                                                     activation='relu')
                                                    (self.encoded))
                    continue
                self.decoder_layers.append(Dense(layer_dim,
                                                 activation='relu')
                                                (self.decoder_layers[i - 1]))
                if i == len(self.layer_dims) - 1:
                    self.decoded = Dense(self.input_dim,
                                         activation='sigmoid')(
                                             self.decoder_layers[i])

        autoencoder = Model(input=self.input, output=self.decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.fit(X,
                        X,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        verbose=0)
        self.encoder = Model(input=self.input, output=self.encoded)
        return self

    def transform(self, X):
        return self.encoder.predict(X)


class Transformers(Enum):
    pca = 1
    subset = 2
    autoencoder = 3
    normalize = 4
    kpca = 5

Transformers.pca.transformer = PCATransformer
Transformers.subset.transformer = FeatureSubsetTransformer
Transformers.autoencoder.transformer = AutoencoderTransformer
Transformers.normalize.transformer = Normalizer
Transformers.kpca.transformer = KernelPCA
