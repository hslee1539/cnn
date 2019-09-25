from import_lib import lib
from cnn.struct.layer_module import Layer
from ctypes import Structure, POINTER

class _Iterator(Structure):
    _fields_ = [
        ('start', Layer),
        ('stop', Layer),
        ('cur', Layer)
    ]



lib.cnn_create_iterator1.argtypes = [Layer]
lib.cnn_create_iterator1.restype = POINTER(_Iterator)
lib.cnn_create_iterator2.argtypes = [Layer, Layer]
lib.cnn_create_iterator2.restype = POINTER(_Iterator)
"""_create(network) , _create(start, stop)"""
def _create(*args):
    if(len(args) == 1):
        return lib.cnn_create_iterator1(args[0])
    elif(len(args) == 2):
        return lib.cnn_create_iterator2(args[0], args[1])
    else:
        raise "잘못된 생성"

lib.cnn_release_iterator.argtypes = [POINTER(_Iterator)]
def _release(self):
    lib.cnn_release_iterator(self)

lib.cnn_iterator_next.argtypes = [POINTER(_Iterator)]
lib.cnn_iterator_next.restype = Layer
def _next(self):
    return lib.cnn_iterator_next(self)

lib.cnn_iterator_back.argtypes = [POINTER(_Iterator)]
lib.cnn_iterator_back.restype = Layer
def _back(self):
    return lib.cnn_iterator_back(self)


Iterator = POINTER(_Iterator)
Iterator.create = staticmethod(_create)
Iterator.release = _release
Iterator.next = _next
Iterator.back = _back
