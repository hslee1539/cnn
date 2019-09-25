from cnn.import_lib import lib
from cnn.tensor.struct.tensor_module import Tensor
from cnn.struct.layer_module import Layer
from ctypes import Structure, c_int, POINTER

def createSigmoidLayer():
    return lib.cnn_create_sigmoid_layer()

#lib.cnn_create_sigmoid_layer.argtypes = (Layer, Layer)
lib.cnn_create_sigmoid_layer.restype = Layer

