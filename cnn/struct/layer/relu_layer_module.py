from import_lib import lib
from tensor.main_module import Tensor
from cnn.struct.layer_module import Layer
from ctypes import Structure, c_int, POINTER

def createReluLayer():
    return lib.cnn_create_relu_layer()

#lib.cnn_create_relu_layer.argtypes = (Layer, Layer)
lib.cnn_create_relu_layer.restype = Layer

