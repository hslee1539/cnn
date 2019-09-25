from cnn.struct.iterator_module import Iterator

class PyIterator:
    """파이썬 기반의 iter 객체를 만듭니다.
    cnn_Iterator를 활용합니다.
    PyIterator(network) : 
    PyIterator(start, stop) : """
    def __init__(self, *args):
        if(len(args) == 1):
            self.iterator = Iterator.create(args[0])
        elif(len(args) == 2):
            self.iterator = Iterator.create(args[0], args[1])

    def __iter__(self):
        return self
    
    def __next__(self):
        retval = self.iterator.next()
        if(retval.getLayerAdress() == 0):
            raise StopIteration
        return retval
    
    def __del__(self):
        self.iterator.release()

class PyBackwardIterator:
    """파이썬 기반의 iter 객체를 만듭니다.
    cnn_Iterator를 활용합니다.
    PyIterator(network) : 
    PyIterator(start, stop) : """
    def __init__(self, *args):
        if(len(args) == 1):
            self.iterator = Iterator.create(args[0])
        elif(len(args) == 2):
            self.iterator = Iterator.create(args[0], args[1])
        
    
    def __iter__(self):
        return self
    
    def __next__(self):
        retval = self.iterator.back()
        if(retval.getLayerAdress() == 0):
            raise StopIteration
        return retval
    
    def __del__(self):
        self.iterator.release()