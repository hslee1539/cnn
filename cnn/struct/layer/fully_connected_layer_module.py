from import_lib import lib
from tensor.main_module import Tensor
from cnn.struct.layer_module import Layer
from ctypes import Structure, c_int, POINTER

def createFullyConnectedLayer(inLayer, outLayer, w, b):
    """cnn.struct.Layer기반의 완전연결 레이어를 만듭니다. 활성함수는 포함되어 있지 않습니다."""
    return lib.cnn_create_fully_connected_layer(inLayer, outLayer, w, b)

lib.cnn_create_fully_connected_layer.argtypes = (Layer, Layer, Tensor, Tensor)
lib.cnn_create_fully_connected_layer.restype = Layer
