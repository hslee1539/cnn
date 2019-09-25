from import_lib import lib
from tensor.main_module import Tensor
from cnn.struct.layer_module import Layer
from ctypes import Structure, c_int, POINTER
from cnn.struct.layer.parse_tensor_module import getTensor

def createDatasetLayer(x, table):
    """cnn.struct.Layer기반의 데이타셋 레이어를 만듭니다. x와 table를 참조하여 생성합니다."""
    x = getTensor(x)
    table = getTensor(table)
    return lib.cnn_create_dataset_layer(x, table)

lib.cnn_create_dataset_layer.argtypes = (Tensor, Tensor)
lib.cnn_create_dataset_layer.restype = Layer