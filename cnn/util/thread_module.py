from cnn.import_lib import lib
from cnn.struct.layer_module import Layer
from cnn.struct.optimizer_module import Optimizer
from ctypes import Structure, c_int, POINTER, c_float, c_ulong, CFUNCTYPE

cnn_thread_func = CFUNCTYPE(None, POINTER(c_ulong))

class _ThreadArgs(Structure):
    _fields_ = [
        ('func', cnn_thread_func),
        ('layer', Layer),
        ('optimizer', Optimizer),
        ('index', c_int),
        ('max_index', c_int),
        ('running', c_int)
    ] 

class _Thread(Structure):
    _fields_ = [
        ('pthread', POINTER(c_ulong)),
        ('args', POINTER(_ThreadArgs)),
        ('count', c_int)
    ]

lib.cnn_create_thread.argtypes = [c_int]
lib.cnn_create_thread.restype = POINTER(_Thread)
def _create(count):
    return lib.cnn_create_thread(count)

lib.cnn_release_thread.argtypes = [POINTER(_Thread)]
def _release(self):
    lib.cnn_release_thread(self)

lib.cnn_thread_forward.argtypes = [POINTER(_Thread), Layer]
def _forward(self, layer):
    lib.cnn_thread_forward(self, layer)

lib.cnn_thread_backward.argtypes = [POINTER(_Thread), Layer]
def _backward(self, layer):
    lib.cnn_thread_backward(self, layer)

lib.cnn_thread_update.argtypes = [POINTER(_Thread), Layer, Optimizer]
def _update(self, layer, optimizer):
    lib.cnn_thread_update(self, layer, optimizer)

lib.cnn_thread_networkNext.argtypes = [POINTER(_Thread), Layer]
lib.cnn_thread_networkNext.restype = c_int
def _networkNext(self, layer):
    return lib.cnn_thread_networkNext(self, layer)

lib.cnn_thread_start.argtypes = [POINTER(_Thread)]
def _start(self):
    lib.cnn_thread_start(self)

lib.cnn_thread_end.argtypes = [POINTER(_Thread)]
def _end(self):
    lib.cnn_thread_end(self)

def _getFunc(self):
    return self.contents.func

def _setFunc(self, value):
    self.contents.func = value

def _getLayer(self):
    return self.contents.layer

def _setLayer(self, value):
    self.contents.layer = value

def _getOptimizer(self):
    return self.contents.optimizer

def _setOptimizer(self, value):
    self.contents.optimizer = value

def _getIndex(self):
    return self.contents.index

def _setIndex(self, value):
    self.contents.index = value

def _getMax_index(self):
    return self.contents.max_index

def _setMax_index(self, value):
    self.contents.max_index = value

def _getRunning(self):
    return self.contents.running

def _setRunning(self, value):
    self.contents.running = value

def _getPThread(self):
    return self.contents.pthread

def _setPthread(self, value):
    self.contents.pthread = value

def _getArgs(self):
    return self.contents.args

def _setArgs(self, value):
    self.contents.args = value

def _getCount(self):
    return self.contents.count

def _setCount(self, value):
    self.contents.count = value

def _getSw(self):
    return self.contents.sw

def _setSw(self, value):
    self.contents.sw = value

ThreadArgs = POINTER(_ThreadArgs)
ThreadArgs.func = property(_getFunc, _setFunc)
ThreadArgs.layer = property(_getLayer, _setLayer)
ThreadArgs.optimzer = property(_getOptimizer, _setOptimizer)
ThreadArgs.index = property(_getIndex, _setIndex)
ThreadArgs.max_index = property(_getMax_index, _setMax_index)
ThreadArgs.running = property(_getRunning, _setRunning)

Thread = POINTER(_Thread)
Thread.pthreads = property(_getPThread, _setPthread)
Thread.args = property(_getArgs, _setArgs)
Thread.count = property(_getCount, _setCount)
Thread.sw = property(_getSw, _setSw)
Thread.create = staticmethod(_create)
Thread.release = _release
Thread.forward = _forward
Thread.backward = _backward
Thread.update = _update
Thread.networkNext = _networkNext
Thread.start = _start
Thread.end = _end