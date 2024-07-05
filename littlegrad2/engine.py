import numpy as np

class Tensor:
    def __init__(self, data = [], children = (), op = ''):
        self.children = children
        self.backward = lambda: None
        self._op = op
        
        self.data = data if isinstance(data, np.ndarray) else np.array(object = data, dtype = float)
        self.data = self.data if self.data.ndim >= 2 else self.data.reshape((1,-1))
        self.grad, self.v, self.s = np.zeros_like(self.data), np.zeros_like(self.data), np.zeros_like(self.data)

    def __repr__(self):
        return f"Tensor object with data {self.data}"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        other.grad = other.grad if other.grad.shape == self.grad.shape else np.zeros_like(self.grad) #works for broadcasting literals but not params
        out = Tensor(data = (self.data + other.data), children = (self, other), op = '+')

        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out.backward = backward
        return out
    
    def __mul__(self, other):
        other = other if type(other) == Tensor else Tensor(other)
        other.grad = other.grad if other.grad.shape == self.grad.shape else np.zeros_like(self.grad) #works for broadcasting literals but not params   
        out = Tensor(data = (self.data * other.data), children = (self, other), op = '*')
        
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.backward = backward
        return out
    
    def __pow__(self, other):
        other = other if type(other) == Tensor else Tensor(other)
        other.grad = other.grad if other.grad.shape == self.grad.shape else np.zeros_like(self.grad)
        out = Tensor(data = (self.data ** other.data), children = (self, other), op = '**')
        
        def backward():
            self.grad += (other.data * (self.data ** (other.data - 1))) * out.grad
            other.grad += (self.data ** other.data) * np.log(max(abs(self.data), 1e-10)) * out.grad
        out.backward = backward
        return out
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * (other**-1)
    
    def __radd__(self, other): # runs if other + self didn't work
        return self + other
    
    def __rmul__(self, other): # runs if other * self didn't work
        return self * other
    
    def __rsub__(self, other): # runs if other - self didn't work
        return (-self) + other # don't actually implement subtraction because it's not commutative and wouldn't be used here
    
    def __rtruediv__(self, other): # runs if other / self didn't work
        return (self ** -1) * other # don't actually implement division because it's not commutative and wouldn't be used here
    
    def __rpow__(self, other): # runs if other ** self didn't work
        other = other if type(other) == Tensor else Tensor(other)
        other.grad = other.grad if other.grad.shape == self.grad.shape else np.zeros_like(self.grad)
        out = Tensor(data = (other.data ** self.data), children = (other, self), op = '**')
        
        def backward():
            other.grad += (self.data * (other.data ** (self.data - 1))) * out.grad
            self.grad += (other.data ** self.data) * np.log(max(abs(self.data), 1e-10)) * out.grad
        out.backward = backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def relu(self):
        out = Tensor(data = (self.data > 0) * self.data, children = (self,), op = 'ReLU')

        def backward():
            #self.grad += out.grad if self.data else 0 #negative numbers still evaluate to True for some dumb reason
            self.grad += (self.data > 0) * out.grad
        out.backward = backward
        return out
    
    def __matmul__(self, other): #almost identical to __mul__ (note .T's)
        other = other if type(other) == Tensor else Tensor(other)
        #other.grad = other.grad if other.grad.shape == self.grad.shape else np.zeros_like(self.grad)
        out = Tensor(data = (self.data @ other.data), children = (self, other), op = '@')
        
        def backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out.backward = backward
        return out
    
    def transpose(self):    
        out = Tensor(data = self.data.T, children = (self,), op = 'T')

        def backward():
            self.grad += out.grad.T
        out.backward = backward
        return out
    
    def flatten(self):
        out = Tensor(data = self.data.reshape((1,-1)), children = (self,), op = 'F')

        def backward():
            self.grad += out.grad.reshape(self.grad.shape)
        out.backward = backward
        return out
    
    def exp(self):
        out = Tensor(data = np.exp(self.data), children = (self,), op = 'exp')

        def backward():
            self.grad += np.exp(self.data) * out.grad
        out.backward = backward
        return out
    
    def log(self):
        out = Tensor(data = np.log(self.data), children = (self,), op = 'log')

        def backward():
            self.grad += (self.data ** -1) * out.grad
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

        self.grad = np.ones(shape = self.grad.shape, dtype = float)
        for node in reversed(nodeList):
            node.backward()
            #print(node, ': ', [child.grad for child in node.children]) #for debug