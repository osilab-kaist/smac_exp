from .q_learner import QLearner
from .iqn_learner import IQNLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .drima_learner import QLearner as DrimaLearner
from .max_q_learner_ddpg import DDPGQLearner
from .max_q_learner_sac import SACQLearner
from .maddpg_learner import MADDPGLearner
from .maddpg_learner_discrete import MADDPGDiscreteLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["iqn_learner"] = IQNLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["drima_learner"] = DrimaLearner
REGISTRY["ddpg"] = DDPGQLearner
REGISTRY["sac"] = SACQLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["maddpg_learner_discrete"] = MADDPGDiscreteLearner