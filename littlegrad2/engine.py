import numpy as np

class Tensor:
    def __init__(self, data = [], children = (), op = ''):
        self.children = children
        self.backward = lambda: None
        self._op = op
        
        self.data = data if isinstance(data, np.ndarray) else np.array(object = data, dtype = float)
        #self.data = data if isinstance(data, np.ndarray) else np.array(object = data)
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
        return self * (other**-1.0)
    
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
    
    def __matmul__(self, other): #almost identical to __mul__
        other = other if type(other) == Tensor else Tensor(other)
        out = Tensor(data = (self.data @ other.data), children = (self, other), op = '@')
        
        def backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out.backward = backward
        return out
    
    def dot(self, other):
        return self.flatten() @ other.flatten().transpose()
        
    # def conv2d(self, other): #TODO: enforce correct matrix dims
    #     (nx, ny, nc) = self.data.shape
    #     (fx, fy, fc, fn) = other.data.shape #TODO: ENFORCE NC == FC?
    #     out = Tensor(data = np.zeros(shape = (nx-fx+1, ny-fy+1, fn)), children = (self, other), op = 'conv')
    #     for x in range(out.data.shape[0]):
    #         for y in range(out.data.shape[1]):
    #             for f in range(out.data.shape[2]):
    #                 mask = out.slice((slice(x, x+1), slice(y, y+1), slice(f, f+1)))
    #                 mask += self.slice((slice(x, x+fx), slice(y, y+fy))).dot(other.slice((slice(fx+1), slice(fy+1), slice(fc+1), slice(f, f+1))))
    #     return out
    
    # def dconv2d(self, other): #TODO: enforce correct matrix dims
    #     (nx, ny, nc) = self.data.shape
    #     (fx, fy, fc) = other.data.shape #TODO: ENFORCE NC == FC?
    #     out = Tensor(data = np.zeros(shape = (nx-fx+1, ny-fy+1, nc)), children = (self, other), op = 'dconv')
    #     for x in range(out.data.shape[0]):
    #         for y in range(out.data.shape[1]):
    #             for c in range(out.data.shape[2]):
    #                 mask = out.slice((slice(x, x+1), slice(y, y+1), slice(c, c+1)))
    #                 mask += self.slice((slice(x, x+fx), slice(y, y+fy), slice(c, c+1))).dot(other.slice((slice(fx+1), slice(fy+1), slice(c, c+1))))
    #     return out
    
    def conv(self, other): #TODO: enforce correct matrix dims? (and other's type?)
        #return (self.dftNd() * other.flip().dftNd()).idftNd()

        otherPadded = Tensor(data = np.zeros_like(self.data, dtype = float)).sliceAdd(other.flip(), tuple([slice(dim) for dim in other.data.shape][:-1]))
        out = (self.dftNd() * otherPadded.dftNd()).idftNd()
        return out.slice((slice(-1-other.data.shape[0], self.data.shape[0]), slice(-1-other.data.shape[1], self.data.shape[1])))
    
    def maxPool2d(self, filter_size = 2, stride = 2):
        (nx, ny, nc) = self.data.shape
        out = Tensor(data = np.ndarray(shape = (nx//stride, ny//stride, nc)), children = (self,), op = 'pool')
        for x in range(out.data.shape[0]):
            for y in range(out.data.shape[1]):
                for c in range(out.data.shape[2]):
                    mask = out.slice((slice(x, x+1), slice(y, y+1), slice(c, c+1)))
                    mask += self.slice((slice(x*stride, x*stride+filter_size), slice(y*stride, y*stride+filter_size), slice(c, c+1))).max()
        return out

    def transpose(self, axes = None):    
        out = Tensor(data = np.transpose(self.data, axes = axes), children = (self,), op = 'T')

        def backward():
            self.grad += out.grad.T
        out.backward = backward
        return out
    
    def reshape(self, shape, order = 'C'):
        out = Tensor(data = self.data.reshape(shape, order = order), children = (self,), op = 'R')

        def backward():
            self.grad += out.grad.reshape(self.grad.shape)
        out.backward = backward
        return out
    
    def flatten(self):
        return self.reshape((1, -1))
    
    def slice(self, slice_object_tuple):
        out = Tensor(data = self.data[slice_object_tuple], children = (self,), op = 'S')

        def backward():
            self.grad[slice_object_tuple] += out.grad
        out.backward = backward
        return out

    def sliceAdd(self, other, slice_object_tuple): #TODO: check tensor dims? also is this correct?
        other = other if type(other) == Tensor else Tensor(other)
        out = Tensor(data = self.data, children = (self, other), op = 'S+')
        out.data[slice_object_tuple] += other.data

        def backward():
            self.grad[slice_object_tuple] += out.grad
            other.grad += out.grad
        out.backward = backward
        return out
    
    def max(self):
        index = np.argmax(self.data)
        return self.flatten().slice((slice(2), slice(index, index + 1)))
    
    def flip(self, axis = None):
        out = Tensor(data = np.flip(self.data, axis = axis), children = (self,), op = 'flip')

        def backward():
            self.grad += np.flip(out.grad, axis = axis)
        out.backward = backward
        return out
    
    # def dft1d(self): # 1-dimensional discrete fourier transform 
    #     dftMatrix = np.arange(len(self.data)).reshape((-1, 1)) @ np.arange(len(self.data)).reshape((1, -1))
    #     return self @ np.exp(-2j * np.pi * dftMatrix / len(self.data))
    
    def dftNd(self):
        out = Tensor(data = self.data)
        dims = np.arange(len(self.data.shape))
        for dim in dims:
            shape = out.data.shape
            dftMatrix = np.arange(shape[0]).reshape((-1, 1)) @ np.arange(shape[0]).reshape((1, -1))
            out = (out.reshape((-1, shape[0])) @ np.exp(-2j * np.pi * dftMatrix / shape[0])).reshape(shape)
            out = np.transpose(out, axes = np.roll(dims, 1)) #increment dims (0->1, 1->2, etc)
        return out
    
    # def idft1d(self):
    #     dftMatrix = np.arange(len(self.data)).reshape((-1, 1)) @ np.arange(len(self.data)).reshape((1, -1))
    #     return (self @ np.exp(2j * np.pi * dftMatrix / len(self.data))) / len(self.data)
    
    def idftNd(self):
        out = Tensor(data = self.data)
        dims = np.arange(len(self.data.shape))
        for dim in dims:
            shape = out.data.shape
            dftMatrix = np.arange(shape[0]).reshape((-1, 1)) @ np.arange(shape[0]).reshape((1, -1))
            out = ((out.reshape((-1, shape[0])) @ np.exp(2j * np.pi * dftMatrix / shape[0])) / shape[0]).reshape(shape)
            out = np.transpose(out, axes = np.roll(dims, 1)) #increment dims (0->1, 1->2, etc)
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