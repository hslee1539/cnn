from import_lib import lib
from tensor.main_module import Tensor
from cnn.struct.layer_module import Layer
from ctypes import Structure, c_int, POINTER

def createMeansquareLayer(inLayer, datasetLayer):
    return lib.cnn_create_meansquare_layer(inLayer, datasetLayer)

lib.cnn_create_meansquare_layer.argtypes = (Layer, Layer)
lib.cnn_create_meansquare_layer.restype = Layer
