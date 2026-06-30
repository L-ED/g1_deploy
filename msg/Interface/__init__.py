from .cyclone import CDDSInterface
from .zmq import ZMQInterface

INTERFACE_MAP = {
    "zmq": ZMQInterface,
    "cyclone": CDDSInterface
}
    