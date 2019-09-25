from cnn.import_lib import lib
from cnn.tensor.struct.tensor_module import Tensor
from cnn.struct.updatelist_module import UpdateList, UpdateSet
from cnn.struct.extradata_module import ExtraData
from cnn.struct.optimizer_module import Optimizer
from ctypes import Structure, c_int, POINTER, c_float, c_char_p, c_void_p, CFUNCTYPE, cast, c_ulonglong

callback_init = CFUNCTYPE(None, c_void_p)
callback_computing = CFUNCTYPE(None, c_void_p, c_int, c_int)
callback_update = CFUNCTYPE(None, c_void_p, Optimizer, c_int, c_int)
callback_initUpdate = CFUNCTYPE(None, c_void_p, Optimizer)

class _Layer(Structure):
    _fields_ = [
        ('name', c_char_p),
        ('out', Tensor),
        ('dx', Tensor),
        ('updateList', UpdateList),
        ('extra', ExtraData),
        ('inLayer', c_void_p),
        ('outLayer', c_void_p),
        ('childLayer', POINTER(c_void_p)),
        ('childLayer_size', c_int),
        ('forward', callback_computing),
        ('backward', callback_computing),
        ('initForward', callback_init),
        ('initBackward', callback_init),
        ('release', callback_init),
        ('update', callback_update),
        ('initUpdate', callback_initUpdate)
        ]

def _getName(self):
    return self.contents.name

def _setName(self, value):
    self.contents.name = value

def _getOut(self):
    return self.contents.out

def _setOut(self, value):
    self.contents.out = value

def _getDx(self):
    return self.contents.dx

def _setDx(self, value):
    self.contents.dx = value

def _getUpdateList(self):
    return self.contents.updateList

def _setUpdateList(self, value):
    self.contents.updateList = value

def _getExtra(self):
    return self.contents.extra

def _setExtra(self, value):
    self.contents.extra = value

def _getInLayer(self):
    return cast(self.contents.inLayer, Layer)

def _setInLayer(self, value):
    self.contents.inLayer = value

def _getOutLayer(self):
    return cast(self.contents.outLayer, Layer)

def _setOutLayer(self, value):
    self.contents.outLayer = value

def _getChildLayer(self):
    return cast(self.contents.childLayer, POINTER(Layer))

def _setChildLayer(self, value):
    self.contents.childLayer = value

def _getChildLayerSize(self):
    return self.contents.childLayer_size

def _setChildLayerSize(self, value):
    self.contents.childLayer_size = value

def _getForward(self):
    return self.contents.forward

def _setForward(self, value):
    self.contents.forward = value

def _getBackward(self):
    return self.contents.backward

def _setBackawrd(self, value):
    self.contents.backwrad = value

def _getInitForward(self):
    return self.contents.initForward

def _setInitForward(self, value):
    self.contents.initForward = value

def _getInitBackward(self):
    return self.contents.initBackward

def _setInitBackward(self, value):
    self.contents.initBackward = value

def _getRelease(self):
    return self.contents.release

def _setRelease(self, value):
    self.contents.release = value

def _getUpdate(self):
    return self.contents.update

def _setUpdate(self, value):
    self.contents.update = value

def _create(name, inLayer_size, outLayer_size, forward, backward, initForward, initBackward, release, update, initUpdate):
    return lib.cnn_create_layer(name, inLayer_size, outLayer_size, forward, backward, initForward, initBackward, release, update, initUpdate)

def _release_deep(self):
    return lib.cnn_release_layer_deep(self)

def _forward(self, index, max_index):
    return lib.cnn_layer_forward(self, index, max_index)

def _backward(self, index, max_index):
    return lib.cnn_layer_backward(self, index, max_index)

def _initForward(self):
    return lib.cnn_layer_initForward(self)

def _initBackward(self):
    return lib.cnn_layer_initBackward(self)

def _update(self, optimizer, index, max_index):
    return lib.cnn_layer_update(self, optimizer, index, max_index)

def _initUpdate(self, optimizer):
    return lib.cnn_layer_initUpdate(self, optimizer)

def _getLeftTerminal(self):
    return lib.cnn_layer_getLeftTerminal(self)

def _getRightTerminal(self):
    return lib.cnn_layer_getRightTerminal(self)

def _link(self, right):
    return lib.cnn_layer_link(self, right)

def _setLearningData(self, dataLayer):
    return lib.cnn_layer_setLearningData(self, dataLayer)

    
    

lib.cnn_create_layer.argtypes = (c_char_p, c_int, c_int, callback_computing, callback_computing, callback_init, callback_init, callback_init, callback_update, callback_initUpdate)
lib.cnn_create_layer.restype = POINTER(_Layer)
lib.cnn_release_layer_deep.argtypes = [POINTER(_Layer)]
lib.cnn_release_layer_deep.restype = c_int
lib.cnn_layer_forward.argtypes = [POINTER(_Layer), c_int, c_int]
lib.cnn_layer_forward.restype = POINTER(_Layer)
lib.cnn_layer_backward.argtypes = (POINTER(_Layer), c_int, c_int)
lib.cnn_layer_backward.restype = POINTER(_Layer)
lib.cnn_layer_initForward.argtypes = [POINTER(_Layer)]
lib.cnn_layer_initForward.restype = POINTER(_Layer)
lib.cnn_layer_initBackward.argtypes = [POINTER(_Layer)]
lib.cnn_layer_initBackward.restype = POINTER(_Layer)
lib.cnn_layer_update.argtypes = (POINTER(_Layer), Optimizer, c_int, c_int)
lib.cnn_layer_update.restype = POINTER(_Layer)
lib.cnn_layer_initUpdate.argtypes = (POINTER(_Layer), Optimizer)
lib.cnn_layer_initUpdate.restype = POINTER(_Layer)
lib.cnn_layer_getLeftTerminal.argtypes = [POINTER(_Layer)]
lib.cnn_layer_getLeftTerminal.restype = POINTER(_Layer)
lib.cnn_layer_getRightTerminal.argtypes = [POINTER(_Layer)]
lib.cnn_layer_getRightTerminal.restype = POINTER(_Layer)
lib.cnn_layer_link.argtypes = [POINTER(_Layer), POINTER(_Layer)]
lib.cnn_layer_link.restype = c_int
lib.cnn_layer_setLearningData.argtypes = [POINTER(_Layer), POINTER(_Layer)]
lib.cnn_layer_setLearningData.restype = POINTER(_Layer)


Layer = POINTER(_Layer)
Layer.__doc__ = " cnn_Layer 구조체의 포인터에 프로퍼티와 메소드를 추가한 클래스입니다. Layer.create 정젹 변수로 생성하여 사용하세요."
Layer.create = staticmethod(_create)
Layer.release = _release_deep
Layer.forward = _forward
Layer.backward = _backward
Layer.initForward = _initForward
Layer.initBackward = _initBackward
Layer.update = _update
Layer.initUpdate = _initUpdate
Layer.getLeftTerminal = _getLeftTerminal
Layer.getRightTerminal = _getRightTerminal
Layer.link = _link
Layer.setLearningData = _setLearningData


Layer.name = property(_getName, _setName)
Layer.out = property(_getOut, _setOut)
Layer.dx = property(_getDx, _setDx)
Layer.updateList = property(_getUpdateList, _setUpdateList)
Layer.extra = property(_getExtra, _setExtra)
Layer.inLayer = property(_getInLayer, _setInLayer)
Layer.outLayer = property(_getOutLayer, _setOutLayer)
Layer.childLayer = property(_getChildLayer, _setChildLayer)
Layer.childLayer_size = property(_getChildLayerSize, _setChildLayerSize)
Layer._forward = property(_getForward, _setForward)
Layer._backward = property(_getBackward, _setBackawrd)
Layer._initForward = property(_getInitForward, _setInitForward)
Layer._initBackward = property(_getInitBackward, _setInitBackward)
Layer._release = property(_getRelease, _setRelease)
Layer._update = property(_getUpdate, _setUpdate)

