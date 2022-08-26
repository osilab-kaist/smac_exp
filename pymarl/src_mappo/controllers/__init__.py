REGISTRY = {}

from .basic_controller import BasicMAC
from .ddpg_controller import ddpg_BasicMAC
from .dist_controller import DistMAC
from .central_basic_controller import CentralBasicMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["dist_mac"] = DistMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["ddpg_mac"] = ddpg_BasicMAC