from import_lib import lib
from tensor.main_module import Tensor
#from cnn.struct.layer_module import Layer
from cnn.struct.optimizer_module import Optimizer
from ctypes import Structure, c_float, POINTER

def createAda(learning_rate):
    return lib.cnn_create_Ada(learning_rate)

lib.cnn_create_Ada.argtypes = [c_float]
lib.cnn_create_Ada.restype = Optimizer
