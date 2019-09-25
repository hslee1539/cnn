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
for i in range(10000):
    if(i % 1000 == 0):
        print(i, trainNetwork.out.scalas[0])

    # 이전보다 개선된 버전
    # 다중쓰레드에서 네트워크 단위로 forward나 backward 등을 비동기 실행 하면,
    # 처리해야 하는 단말 노드를 가기 전에 networkNext가 처리되어 다음 노드를 처리하는
    # 불상사가 생길 위험이 있음.
    # 이를 해결하려면 forward나 기타 등등을 쓰래드가 완료될때 까지 대기하는 식의 동기
    # 작업이 필요함.
    # 이러면 비동기로 실행하고 다음 연산때 대기하고 완료 즉시 바로 연산을 하지 못함.
    for layer in cnn.PyIterator(trainNetwork):
        layer.forward(0,1)
    
    for layer in cnn.PyBackwardIterator(trainNetwork):
        layer.backward(0,1)
        
    for layer in cnn.PyIterator(trainNetwork):
        layer.update(optimizer, 0, 1)

print('정답')
for i in range(4):
    print(mainNetwork.out.scalas[i])
