from cnn.import_lib import lib
from ctypes import Structure, POINTER, c_longlong, c_float, c_int
import numpy as np

class _Tensor(Structure):
    _fields_ = [("size", c_int),
    ("dim", c_int),
    ("scalas", POINTER(c_float)),
    ("shapes", POINTER(c_longlong)),
    ("information", c_int)
    ]

lib.tensor_create_nonstruct.argtypes = (POINTER(c_float), c_int, POINTER(c_longlong), c_int)
lib.tensor_create_nonstruct.restype = POINTER(_Tensor)
lib.tensor_create_nonstruct_deep.argtypes = (POINTER(c_float), c_int, POINTER(c_longlong), c_int)
lib.tensor_create_nonstruct_deep.restype = POINTER(_Tensor)
lib.tensor_create_nonstruct_deep_shape.argtypes = (POINTER(c_float), c_int, POINTER(c_longlong), c_int)
lib.tensor_create_nonstruct_deep_shape.restype = POINTER(_Tensor)
lib.tensor_referTo.argtypes = (POINTER(_Tensor), POINTER(_Tensor))
lib.tensor_release_deep.argtypes = [POINTER(_Tensor)]

def _getSize(self):
    return self.contents.size

def _getDim(self):
    return self.contents.dim

def _getScalas(self):
    return self.contents.scalas

def _getShapes(self):
    return self.contents.shapes

def _setSize(self, value):
    self.contents.size = value

def _setDim(self, value):
    self.contents.dim = value

def _setScalas(self, value):
    self.contents.scalas = value

def _setShapes(self, value):
    self.contents.shapes = value

def _numpy2Tensor(nparray):
    return lib.tensor_create_nonstruct_deep_shape(nparray.ctypes.data_as(POINTER(c_float)), nparray.size, nparray.ctypes.shape, nparray.ndim)

def _create(scalas32f, shape, deep = True):
    c_scalas = (c_float * len(scalas32f))(scalas32f)
    c_shapes = (c_int * len(shape))
    if (deep):
        return lib.tensor_create_nonstruct_deep( c_scalas, len(scalas32f), c_shapes, len(shape))
    else:
        return lib.tensor_create_nonstruct( c_scalas, len(scalas32f), c_shapes, len(shape))

def _release_deep(self):
    lib.tensor_release_deep(self)
    del self # 검증 안됨

def _referTo(self, tensor):
    lib.tensor_referTo(self, tensor)
    return self

    

Tensor = POINTER(_Tensor)
Tensor.__doc__ = "텐서 구조체 포인터에 프로퍼티와 메소드를 추가한 클래스입니다."
Tensor.numpy2Tensor = staticmethod(_numpy2Tensor)
Tensor.release_deep = _release_deep
Tensor.referTo = _referTo
Tensor.create = staticmethod(_create)
Tensor.size = property(_getSize, _setSize)
Tensor.dim = property(_getDim, _setDim)
Tensor.scalas = property(_getScalas, _setScalas)
Tensor.shapes = property(_getShapes, _setShapes)

