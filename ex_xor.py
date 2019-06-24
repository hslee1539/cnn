import cnn_module as cnn
import numpy as np

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
"""
# builder의 도움이 없는 경우
fc1 = cnn.createFullyConnectedLayer(W1, B1)
sig1 = cnn.createSigmoidLayer()
fc2 = cnn.createFullyConnectedLayer(W2, B2)
sig2 = cnn.createSigmoidLayer()

mainNetwork = cnn.createNetworkLayer(4)
mainNetwork.childLayer[0] = fc1
mainNetwork.childLayer[1] = sig1
mainNetwork.childLayer[2] = fc2
mainNetwork.childLayer[3] = sig2

fc1.link(sig1) #mainNetwork.childLayer[0].link(mainNetwork.childLayer[1])
sig1.link(fc2) #mainNetwork.childLayer[1].link(mainNetwork.childLayer[2])
fc2.link(sig2) #mainNetwork.childLayer[2].link(mainNetwork.childLayer[3])
"""


# 위 네트워크 뒤에 손실 함수 레이어가 붙은 학습용 네트워크 생성
with cnn.NetworkBuilder() as builder:
    builder.createNetwork(2)
    builder.addNetworkLayer(mainNetwork)
    builder.addMeansquareLayer()
    trainNetwork = builder.getNetwork()

trainNetwork.setLearningData(dataLayer)
trainNetwork.initForward()
trainNetwork.initBackward()
trainNetwork.initUpdate(optimizer)

#학습
for i in range(1):
    if(i % 1000 == 0):
        print(i, trainNetwork.out.scalas[0])
    while(cnn.networkNext(trainNetwork.forward(0,1)) > 0):
        pass
    while(cnn.networkNext(trainNetwork.backward(0,1)) > 0):
        pass
    while(cnn.networkNext(trainNetwork.update(optimizer, 0, 1)) > 0):
        pass

print('정답')
for i in range(4):
    print(mainNetwork.out.scalas[i])
