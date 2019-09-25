from cnn.import_lib import lib
from cnn.tensor.import_module import Tensor
from cnn.struct.layer_module import Layer
from ctypes import Structure, c_int, POINTER
from cnn.struct.layer.parse_tensor_module import getTensor

def createBatchnormLayer():
    """cnn.struct.Layer기반의 batchnorm 레이어를 만듭니다."""
    return lib.cnn_create_batchnorm_layer()

lib.cnn_create_batchnorm_layer.argtypes = []
lib.cnn_create_batchnorm_layer.restype = Layer

