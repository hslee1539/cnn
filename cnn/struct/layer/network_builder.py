from import_lib import lib
from tensor.main_module import Tensor
from cnn.struct.layer_module import Layer
from cnn.struct.layer.batchnorm_layer_module import createBatchnormLayer
from cnn.struct.layer.conv3d_layer_module import createConv3dLayer
from cnn.struct.layer.deconv3d_layer_module import createDeconv3dLayer
#from cnn.struct.layer.dateset_layer_module import createDatasetLayer
from cnn.struct.layer.fully_connected_layer_module import createFullyConnectedLayer
from cnn.struct.layer.meansquare_layer_module import createMeansquareLayer
from cnn.struct.layer.network_layer_module import createNetworkLayer, isNetworkLayer
from cnn.struct.layer.relu_layer_module import createReluLayer
from cnn.struct.layer.sigmoid_layer_module import createSigmoidLayer
import numpy as np

class NetworkBuilder:
    """네트워크를 설계해주는 클래스입니다."""
    def __init__(self):
        """빈 네트워크를 생성합니다"""
        self.network = None
        self.pos = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        pass

    @staticmethod
    def getTensor(value):
        if type(value) is Tensor:
            return value
        elif type(value) is np.ndarray:
            return Tensor.numpy2Tensor(value)
        else:
            raise Exception

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
    
    def addLayer(self, layer):
        if(self.pos == 0):
            self.network.childLayer[0] = layer
        elif(self.pos < self.network.childLayer_size):
            self.network.childLayer[self.pos] = layer
            self.network.childLayer[self.pos - 1].link(layer)
        else:
            # 네트워크가 꽉참
            raise Exception
        self.pos += 1
        return self

    def addBatchnormLayer(self):
        return self.addLayer(createBatchnormLayer())
    
    def addConv3dLayer(self, filter, bias, stride, pad, padding):
        return self.addLayer(createConv3dLayer(filter, bias, stride, pad, padding))
    
    def addDeconv3dLayer(self, filter, bias, stride, pad, padding):
        return self.addLayer(createDeconv3dLayer(filter, bias, stride, pad, padding))
    
    def addFCLayer(self, w, b):
        return self.addLayer(createFullyConnectedLayer(w, b))
    
    def addMeansquareLayer(self):
        return self.addLayer(createMeansquareLayer())

    def addNetworkLayer(self, networkLayer):
        return self.addLayer(networkLayer)

    def addReluLayer(self):
        return self.addLayer(createReluLayer())
    
    def addSigmoidLayer(self):
        return self.addLayer(createSigmoidLayer())
    
    def getNetwork(self):
        return self.network 