from import_lib import lib
from tensor.main_module import Tensor
from cnn.struct.layer_module import Layer
from ctypes import Structure, c_int, POINTER

def createSigmoidLayer(inLayer, outLayer):
    return lib.cnn_create_sigmoid_layer(inLayer, outLayer)

lib.cnn_create_sigmoid_layer.argtypes = (Layer, Layer)
lib.cnn_create_sigmoid_layer.restype = Layer
