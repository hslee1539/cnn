from import_lib import lib
from cnn.struct.layer_module import Layer
from ctypes import Structure, c_int, POINTER, c_ulonglong

lib.cnn_getLayerAdress.argtypes = [Layer]
lib.cnn_getLayerAdress.restype = c_ulonglong
def _getLayerAdress(self):
    return lib.cnn_getLayerAdress(self)
Layer = Layer
Layer.getLayerAdress = _getLayerAdress