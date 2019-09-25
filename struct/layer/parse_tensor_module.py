from tensor.main_module import Tensor
import numpy as np

def getTensor(value):
    if type(value) is np.ndarray:
        return Tensor.numpy2Tensor(value)
    elif type(value) is Tensor:
        return value
    else:
        raise Exception