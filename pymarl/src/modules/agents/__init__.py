REGISTRY = {}

from .rnn_agent import RNNAgent
from .iqn_rnn_agent import IQNRNNAgent
from .rnn_agent_dist import RNNDistAgent
from .central_rnn_agent import CentralRNNAgent
from .qmix_agent import QMIXRNNAgent, FFAgent


REGISTRY["rnn"] = RNNAgent
REGISTRY["iqn_rnn"] = IQNRNNAgent
REGISTRY["rnn_dist"] = RNNDistAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["qmixrnn"] = QMIXRNNAgent
REGISTRY["ff"] = FFAgent
