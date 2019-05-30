from import_lib import lib
from tensor.main_module import Tensor
from ctypes import Structure, c_int, POINTER, c_float

class _ExtraData(Structure):
    _fields_ = [('tensorSize', c_int),
    ('floatSize', c_int),
    ('intSize', c_int),
    ('tensors', POINTER(Tensor)),
    ('floats', POINTER(c_float)),
    ('ints', POINTER(c_int))]

lib.cnn_create_extraData.restype = POINTER(_ExtraData)
lib.cnn_create_extraData.argtypes = (c_int, c_int, c_int)
lib.cnn_release_extradata.argtypes = [POINTER(_ExtraData)]
lib.cnn_release_extradata_deep.argtypes = [POINTER(_ExtraData)]
 
def _release(self, deep = True):
    if(deep):
        lib.cnn_release_extradata_deep(self)
    else:
        lib.cnn_release_extradata(self)
    del self # 검증 안됨

def _create(tensorsSize, floatsSize, intsSize):
    """ExtraData의 포인터를 반환합니다."""
    return lib.cnn_create_extraData(tensorsSize, floatsSize, intsSize)

def _getTensorSize(self):
    return self.contents.tensorSize

def _getFloatSize(self):
    return self.contents.floatSize

def _getIntSize(self):
    return self.contents.intSize

def _getTensors(self):
    return self.contents.tensors

def _getFloats(self):
    return self.contents.floats

def _getInts(self):
    return self.contents.ints

def _setTensorSize(self, value):
    self.contents.tensorSize = value

def _setFloatSize(self, value):
    self.contents.floatSize = value

def _setIntSize(self, value):
    self.contents.intSize = value

def _setTensors(self, value):
    self.contents.tensors = value

def _setFloats(self, value):
    self.contents.floats = value

def _setInts(self, value):
    self.contents.ints = value

ExtraData = POINTER(_ExtraData)
ExtraData.__doc__ = " cnn_ExtraData 구조체의 포인터에 프로퍼티와 메소드를 추가한 클래스입니다."

ExtraData.tensorSize = property(_getTensorSize, _setTensorSize)
ExtraData.floatSize = property(_getFloatSize, _setFloatSize)
ExtraData.intSize = property(_getIntSize, _setIntSize)
ExtraData.tensors = property(_getTensors, _setTensors)
ExtraData.floats = property(_getFloats, _setFloats)
ExtraData.ints = property(_getInts, _setInts)
ExtraData.create = staticmethod(_create)
ExtraData.release = _release
