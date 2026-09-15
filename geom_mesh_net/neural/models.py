"""Coordinate MLPs that map a position (and optional features) to a guest probability.

These are the networks of experiments E8 and E9 in ``experiments/neural_field/``.
Each ends in a sigmoid, so its output is a probability.
"""

import torch.nn as nn


class ContinuousNeuralField(nn.Module):
    def __init__(self):
        super().__init__()
        # define model: we are using a sequential multi layer perceptron with 3 features, a 128 neurons hidden layer, and 1 output with a ReLU activation function
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() # make sure answer is between 0 and 1
        )

        # forward pass input through model
    def forward(self, x):
        return self.model(x)


class ContinuousNeuralField2(nn.Module):
    def __init__(self):
        super().__init__()
    # need a more complex model
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    # forward pass input through model
    def forward(self, x):
        return self.model(x)

class ContinuousNeuralFieldspatstat_01(nn.Module):
    def __init__(self, barcode_bins = 5):
        super().__init__()
        input_features = 3 + barcode_bins
    # need a more complex model
        self.model = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    # forward pass input through model
    def forward(self, x):
        return self.model(x)


class ContinuousNeuralFieldGlobalFeatures(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        if feature_count <= 0:
            raise ValueError("feature_count must be greater than zero")
        input_features = 3 + feature_count
        self.model = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class ContinuousNeuralFieldFeatures(nn.Module):
    def __init__(
        self,
        feature_count=0,
        hidden_width=128,
        hidden_layers=3,
    ):
        super().__init__()
        if feature_count < 0:
            raise ValueError("feature_count must be nonnegative")
        if hidden_width <= 0:
            raise ValueError("hidden_width must be greater than zero")
        if hidden_layers <= 0:
            raise ValueError("hidden_layers must be greater than zero")

        layers = []
        input_features = 3 + feature_count
        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(input_features, hidden_width),
                    nn.ReLU(),
                ]
            )
            input_features = hidden_width
        layers.extend([nn.Linear(hidden_width, 1), nn.Sigmoid()])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
