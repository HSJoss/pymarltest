REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .cnn_agent import CNNAgent
REGISTRY["cnn"] = CNNAgent