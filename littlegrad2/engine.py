import numpy as np

class Tensor:
    def __init__(self, data = [], children = (), grad = 0, op = ''):
        self.data = np.array(object = data)
        self.children = children
        self.grad = grad
        self.op = op
    def __repr__(self):
        return f"Tensor object with data {self.data}"
    
    def __add__(self, other):
        if type(other) != Tensor:
            other = Tensor(other)
        out = Tensor(self.data + other.data)
        out.children = (self, other)
        return out
    
    def __sub__(self, other):
        if type(other) != Tensor:
            other = Tensor(other)
        out = Tensor(self.data - other.data)
        out.children = (self, other)
        return out
    
    def __mul__(self, other):
        if type(other) != Tensor:
            other = Tensor(other)
        out = Tensor(self.data * other.data)
        out.children = (self, other)
        return out
    
    def __truediv__(self, other):
        if type(other) != Tensor:
            other = Tensor(other)
        out = Tensor(self.data / other.data)
        out.children = (self, other)
        return out