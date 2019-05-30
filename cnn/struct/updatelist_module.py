from import_lib import lib
from cnn.struct.updateset_module import UpdateSet
from ctypes import Structure, c_int, POINTER, c_float

class _UpdateList(Structure):
    _fields_ = [
        ('sets', POINTER(UpdateSet)),
        ('setSize', c_int)
    ]

def _create(size):
    return lib.cnn_create_updatelist(size)
    
def _release_deep(pUpdateList):
    lib.cnn_release_updatelist_deep(pUpdateList)

def _getSets(self):
    return self.contents.sets

def _getSetSize(self):
    return self.contents.setSize

def _setSets(self, value):
    self.contents.sets = value

def _setSetSize(self, value):
    self.contents.setSize = value

    
lib.cnn_create_updatelist.argtypes = [c_int]
lib.cnn_create_updatelist.restype = POINTER(_UpdateList)
lib.cnn_release_updatelist_deep.argtypes = [POINTER(_UpdateList)]

UpdateList = POINTER(_UpdateList)
UpdateList.__doc__ = 'cnn_UpdateList의 구조체 포인터인 클래스로, 프로퍼티와 메소드를 제공합니다.'
UpdateList.sets = property(_getSets, _setSets)
UpdateList.setSize = property(_getSetSize, _setSetSize)
UpdateList.create = staticmethod(_create)
UpdateList._release = _release_deep

