import cnn_module as cnn
import numpy as np
import conv3d_module
import time

# 성능 테스트 실험 입니다.

stride = 2
pad = 1
X = np.float32(np.random.randn(10,10,25,25))
DOUT = np.float32(np.random.randn(10,10,11,11))
F = np.float32(np.random.randn(10,10,5,5))
B = np.float32(np.random.randn(10))

dataLayer = cnn.createDatasetLayer(X, DOUT)

convLayer = cnn.createConv3dLayer(F, B, stride, pad, 0)
convLayer.setLearningData(dataLayer)
out2 = np.float32(np.random.randn(2,10,11,11))
dx2 = X.copy()
convLayer.out = cnn.Tensor.numpy2Tensor(out2)
convLayer.dx = cnn.Tensor.numpy2Tensor(dx2)

out1 = out2.reshape(-1)
x = X.reshape(-1)
f = F.reshape(-1)
b = B.reshape(-1)
t1 = time.time_ns()
conv3d_module.partialForward(x, X.shape, f, F.shape, b, stride, pad, 0, out1, out2.shape, 0, 1)
t2 = time.time_ns()
convLayer.forward(0,1)
t3 = time.time_ns()
print('py/cnn',(t2 - t1) / (t3 - t2 + 1))
print(np.sum(out1.reshape(out2.shape) - out2) == 0)

dx1 = dx2.reshape(-1)
dout = DOUT.reshape(-1)

t4 = time.time_ns()
conv3d_module.partialBackward(dout, DOUT.shape, f, F.shape, stride, pad, dx1, dx2.shape, 0, 1)
t5 = time.time_ns()
convLayer.backward(0,1)
t6 = time.time_ns()
print('py/cnn', (t5 - t4) / (t6 - t5 +1))
print(np.sum(dx1.reshape(dx2.shape) - dx2) == 0)

