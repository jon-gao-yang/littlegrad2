import numpy as np

class Tensor:
    def __init__(self, data = [], children = (), grad = 0, op = ''):
        self.data = np.array(object = data, dtype = float)
        self.children = children
        self.grad = grad
        self.backward = lambda: None
        self._op = op

    def __repr__(self):
        return f"Tensor object with data {self.data}"
    
    def __add__(self, other):
        other = other if type(other) == Tensor else Tensor(other)
        out = Tensor(data = (self.data + other.data), children = (self, other), op = '+')

        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out.backward = backward
        return out
    
    def __sub__(self, other):
        other = other if type(other) == Tensor else Tensor(other)
        out = Tensor(data = (self.data - other.data), children = (self, other), op = '-')
        
        def backward():
            self.grad += out.grad
            other.grad += -out.grad
        out.backward = backward
        return out
    
    def __mul__(self, other):
        other = other if type(other) == Tensor else Tensor(other)
        out = Tensor(data = (self.data * other.data), children = (self, other), op = '*')
        
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.backward = backward
        return out
    
    def __truediv__(self, other):
        other = other if type(other) == Tensor else Tensor(other)
        out = Tensor(data = (self.data / other.data), children = (self, other), op = '/')
    
        def backward():
            self.grad += (1 / other.data) * out.grad
            other.grad += (-self.data/(other.data**2)) * out.grad
        out.backward = backward
        return out
    
    def __pow__(self, other):
        other = other if type(other) == Tensor else Tensor(other)
        out = Tensor(data = (self.data ** other.data), children = (self, other), op = '**')
        
        def backward():
            self.grad += (other.data * (self.data ** (other.data - 1))) * out.grad
            other.grad += (self.data ** other.data) * np.log(abs(self.data)+1e-10) * out.grad
        out.backward = backward
        return out
    
    def __radd__(self, other):
        other = other if type(other) == Tensor else Tensor(other)
        return other + self
    
    def __rsub__(self, other):
        other = other if type(other) == Tensor else Tensor(other)
        return other - self
    
    def __rmul__(self, other):
        other = other if type(other) == Tensor else Tensor(other)
        return other * self
    
    def __rtruediv__(self, other):
        other = other if type(other) == Tensor else Tensor(other)
        return other / self
    
    def __rpow__(self, other):
        other = other if type(other) == Tensor else Tensor(other)
        return other ** self
    
    def __neg__(self):
        return self * -1
    
    def relu(self):
        out = Tensor(data = max(0, self.data), children = (self,), op = 'ReLU')

        def backward():
            #self.grad += out.grad if self.data else 0 #negative numbers still evaluate to True for some dumb reason
            self.grad += (self.data > 0) * out.grad
        out.backward = backward
        return out
    
    def backprop(self):
        nodeList = []
        visited = set()
        def toposort(node):
            for child in node.children:
                if child not in visited:
                    toposort(child)
            visited.add(node)
            nodeList.append(node)
        toposort(self)

        self.grad = 1
        for node in reversed(nodeList):
            node.backward()
            #print(node, ': ', [child.grad for child in node.children]) #for debug