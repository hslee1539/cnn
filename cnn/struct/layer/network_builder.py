from import_lib import lib
from tensor.main_module import Tensor
from cnn.struct.layer_module import Layer
from cnn.struct.layer.conv3d_layer_module import createConv3dLayer
#from cnn.struct.layer.dateset_layer_module import createDatasetLayer
from cnn.struct.layer.fully_connected_layer_module import createFullyConnectedLayer
from cnn.struct.layer.meansquare_layer_module import createMeansquareLayer
from cnn.struct.layer.network_layer_module import createNetworkLayer
from cnn.struct.layer.relu_layer_module import createReluLayer
from cnn.struct.layer.sigmoid_layer_module import createSigmoidLayer

class NetworkBuilder:
    def __init__(self):
        self.network = None
        self.pos = 0

    def createNetwork(self, size):
        if (self.network == None):
            self.network = createNetworkLayer(size)
        else:
            # 네트워크를 중복 생성하려고 함
            raise Exception
        return self
    
    def setNetwork(self, network, pos):
        """해당 네트워크의 해당 위치부터 만들기 시작합니다. 생성자로 만든 경우에만 해당됩니다."""
        if (self.network == None):
            self.network = network
            self.pos = pos
        else:
            # 네트워크를 중복 생성하려고 함.
            raise Exception
        return self
    
    def __iadd__(self, right):
        if self.network == None:
            # 연결된 네트워크가 없음
            raise Exception
        if type(right) != Layer:
            # 추가하려는 것이 레이어가 아님
            raise Exception
        self.network.inLayer[self.pos] = right
        self.pos += 1
        return self
    
    def addLayer(self, layer):
        if(self.pos == 0):
            self.network.inLayer[0] = layer
        elif(self.pos < self.network.inLayerSize):
            self.network.inLayer[self.pos] = layer
            self.network.inLayer[self.pos - 1].outLayer[0] = self.network.inLayer[self.pos]
        else:
            # 네트워크가 꽉참
            raise Exception
        self.pos += 1
        return self
    
    def addConv3dLayer(self, filter, bias, stride, pad, padding):
        return self.addLayer(createConv3dLayer(0, 0, filter, bias, stride, pad, padding))
    
    def addFCLayer(self, w, b):
        return self.addLayer(createFullyConnectedLayer(0, 0, w, b))
    
    def addMeansquareLayer(self):
        return self.addLayer(createMeansquareLayer(0, 0))

    def addNetworkLayer(self, networkLayer):
        return self.addLayer()
    def linkDataset(self, dataset):
        
        self.network.inLayer[0]
    def getNetwork(self):
        return self.network 