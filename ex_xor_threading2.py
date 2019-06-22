import cnn_module as cnn
import numpy as np
import threading

# 데이타
X = np.array([[0,0], [0,1], [1,0], [1,1]], np.float32)
Y = np.array([0,1,1,0], np.float32)
dataLayer = cnn.createDatasetLayer(X, Y)
# 가중치들

W1 = np.float32(np.random.randn(2,4,5))
B1 = np.float32(np.random.randn(20))
W2 = np.float32(np.random.randn(20,1))
B2 = np.float32(np.random.randn(1))

learning_rate = 0.02
optimizer = cnn.createSGD(learning_rate)

#"""
# 2층 구조의 네트워크 생성
with cnn.NetworkBuilder() as builder:
    builder.createNetwork(5)
    builder.addFCLayer(W1, B1)
    builder.addBatchnormLayer()
    builder.addReluLayer()
    #builder.addSigmoidLayer()
    builder.addFCLayer(W2, B2)
    #builder.addReluLayer()
    builder.addSigmoidLayer()
    mainNetwork = builder.getNetwork()

# 위 네트워크 뒤에 손실 함수 레이어가 붙은 학습용 네트워크 생성
with cnn.NetworkBuilder() as builder:
    builder.createNetwork(2)
    builder.addNetworkLayer(mainNetwork)
    builder.addMeansquareLayer()
    trainNetwork = builder.getNetwork()

trainNetwork.setLearningData(dataLayer)
trainNetwork.initForward()
trainNetwork.initBackward()

thread = cnn.Thread.create(2)
thread.start()
#학습
for i in range(5000):
    if(i % 1000 == 0):
        print(i, trainNetwork.out.scalas[0])
    while(True):
        #print(trainNetwork.extra.ints[0], mainNetwork.extra.ints[0])
        thread.forward(trainNetwork)
        if thread.networkNext(trainNetwork) > 0: continue
        break
    while(True):
        #print(trainNetwork.extra.ints[0], mainNetwork.extra.ints[0])
        thread.backward(trainNetwork)
        if thread.networkNext(trainNetwork) > 0: continue
        break
    while(True):
        #print(trainNetwork.extra.ints[0], mainNetwork.extra.ints[0])
        thread.update(trainNetwork, optimizer)
        if thread.networkNext(trainNetwork) > 0: continue
        break
    
#thread.end()
#thread.release()
print('정답')
for i in range(4):
    print(mainNetwork.out.scalas[i])