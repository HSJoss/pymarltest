REGISTRY = {}

from .basic_controller import BasicMAC

REGISTRY["basic_mac"] = BasicMAC

from .atari_controller import AtariMAC

REGISTRY["atari_mac"] = AtariMAC

from .origin_atari_controller import OriginAtariMAC

REGISTRY["origin_atari_mac"] = OriginAtariMAC
