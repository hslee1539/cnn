from import_lib import lib
from tensor.main_module import *
from ctypes import Structure, c_int, POINTER, c_float

# Tensor는 POINTER(_Tensor) 임

class _UpdateSet(Structure):
    _fields_ = [
        ('delta', Tensor),
        ('value', Tensor),
        ('momnt', Tensor)
    ] 

lib.cnn_create_updateset.argtypes = (Tensor, Tensor)
lib.cnn_create_updateset.restype = POINTER(_UpdateSet)
lib.cnn_release_updateset_deep.argtypes = [POINTER(_UpdateSet)]
#lib.cnn_release_updateset.argtypes = [POINTER(_UpdateSet)]


def _create(delta, value):
    return lib.cnn_create_updateset(delta, value)

def _release(self, deep = True):
    if(deep):
        lib.cnn_release_updateset_deep(self)
    else:
        lib.cnn_release_updateset(self)
    del self # 검증 안됨

def _getDelta(self):
    return self.contents.delta

def _getValue(self):
    return self.contents.value

def _getMomnt(self):
    return self.contents.momnt

def _setDelta(self, value):
    self.contents.delta = value

def _setValue(self, value):
    self.contents.value = value

def _setMomnt(self, value):
    self.contents.momnt = value


UpdateSet = POINTER(_UpdateSet)
UpdateSet.__doc__ = "cnn_UpdateSet 구조체의 포인터에 프로퍼티와 메소드를 추가한 클래스입니다."
UpdateSet.delta = property(_getDelta, _setDelta)
UpdateSet.value = property(_getValue, _setValue)
UpdateSet.momnt = property(_getMomnt, _setMomnt)
UpdateSet.create = staticmethod(_create)
UpdateSet.release = _release

