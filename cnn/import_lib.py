from ctypes import *
import os
# 전 패키지를 가지는 dll 혹은 so를 로드하게 코딩
#lib = cdll.LoadLibrary('%s/cnn' % os.path.dirname(__file__))
lib = cdll.LoadLibrary(os.path.dirname(__file__) + '/cnn_module.dll')
