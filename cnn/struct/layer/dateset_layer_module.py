from import_lib import lib
from tensor.main_module import Tensor
from cnn.struct.layer_module import Layer
from ctypes import Structure, c_int, POINTER

def createDatasetLayer(x, table):
    """cnn.struct.Layer기반의 데이타셋 레이어를 만듭니다."""
    return lib.cnn_create_dataset_layer(x, table)

lib.cnn_create_dataset_layer.argtypes = (Tensor, Tensor)
lib.cnn_create_dataset_layer.restype = Layer