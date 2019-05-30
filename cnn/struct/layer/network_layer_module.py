from import_lib import lib
from tensor.main_module import Tensor
from cnn.struct.layer_module import Layer
from ctypes import Structure, c_int, POINTER, c_bool

def createNetworkLayer(size):
    return lib.cnn_create_network_layer(size)

def isNetworkLayer(layer):
    return lib.cnn_isNetworkLayer(layer)

lib.cnn_isNetworkLayer.argtypes = [Layer]
lib.cnn_isNetworkLayer.restype = c_bool

lib.cnn_create_network_layer.argtypes = [c_int]
lib.cnn_create_network_layer.restype = Layer
