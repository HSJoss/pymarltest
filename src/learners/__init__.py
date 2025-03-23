from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .dqn_learner import DQNLearner
from .origin_dqn_learner import OriginDQNLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["dqn_learner"] = DQNLearner
REGISTRY["origin_dqn_learner"] = OriginDQNLearner
