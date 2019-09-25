from cnn.import_lib import lib
from ctypes import Structure, POINTER, c_int, c_float, c_char
from cnn.tensor.struct.tensor_module import Tensor

# 여기서 Tensor는 POINTER(_Tensor) 임

lib.tensor_tostring.argtypes = (Tensor, POINTER(c_char), c_int)
lib.tensor_tostring.restype = POINTER(c_char)

def _tostring(self, out = b" " * 1024):
    c_out = (c_char * len(out))(*out)
    lib.tensor_tostring(self, c_out, len(out))
    return out

def _str(self):
    c_out = (c_char * 1024)(*b' ' * 24)
    lib.tensor_tostring(self, c_out, len(c_out))
    return c_out

Tensor.tostring = _tostring
Tensor.__str__ = _str
