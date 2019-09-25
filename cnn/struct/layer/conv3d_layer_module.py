from cnn.import_lib import lib
from cnn.tensor.struct.tensor_module import Tensor
from cnn.struct.layer_module import Layer
from ctypes import Structure, c_int, POINTER
from cnn.struct.layer.parse_tensor_module import getTensor

def createConv3dLayer(filter, bias, stride, pad, padding):
    """cnn.struct.Layer기반의 conv3d 레이어를 만듭니다."""
    filter = getTensor(filter)
    bias = getTensor(bias)
    return lib.cnn_create_conv3d_layer(filter, bias, stride, pad, padding)

lib.cnn_create_conv3d_layer.argtypes = (Tensor, Tensor, c_int, c_int, c_int)
lib.cnn_create_conv3d_layer.restype = Layer

