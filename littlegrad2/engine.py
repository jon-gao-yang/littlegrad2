import numpy as np

class Tensor:
    def __init__(self, data = [], children = (), op = ''):
        self.children = children
        self.backward = lambda: None
        self._op = op
        
        # FOR CONV NET:
        self.data = np.atleast_2d(np.array(object = data.data, dtype = np.complex128)) if isinstance(data, np.ndarray) else np.atleast_2d(np.array(object = data, dtype = np.complex128))
        # ^ NOTE: leave some arrays as float64 if there are memory issues (also complex128 might be hardware specific?)
        # FOR LINEAR NET:
        #self.data = np.atleast_2d(data) if isinstance(data, np.ndarray) else np.array(object = np.atleast_2d(data), dtype = float)
        # ^ NOTE: reshapes 1d arrays to (1, -1), TAKES WAY LESS TIME THAN COMPLEX128

        self.grad, self.v, self.s = np.zeros_like(self.data), np.zeros_like(self.data), np.zeros_like(self.data)

    def __repr__(self):
        return f"Tensor object with data {self.data}"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        #other.grad = other.grad if other.grad.shape == self.grad.shape else np.zeros_like(self.grad) #works for broadcasting literals but not params
        if other.data.shape != self.data.shape: # manual broadcasting to keep track of grads
            otherCopy = Tensor(np.ndarray((self.data.shape)))
            np.copyto(otherCopy.data, other.data)
            other = otherCopy
        out = Tensor(data = (self.data + other.data), children = (self, other), op = '+')

        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out.backward = backward
        return out
    
    def __mul__(self, other):
        other = other if type(other) == Tensor else Tensor(other)
        #other.grad = other.grad if other.grad.shape == self.grad.shape else np.zeros_like(self.grad) #works for broadcasting literals but not params   
        if other.data.shape != self.data.shape: # manual broadcasting to keep track of grads
            otherCopy = Tensor(np.ndarray((self.data.shape)))
            np.copyto(otherCopy.data, other.data)
            other = otherCopy
        
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
            #other.grad += (self.data ** other.data) * np.log(max(abs(self.data), 1e-10)) * out.grad # doesn't work for batched training
            other.grad += (self.data ** other.data) * np.log(self.data + ((self.data == 0) * 1e-10)) * out.grad
        out.backward = backward
        return out
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
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
            #self.grad += out.grad if self.data else 0 # negative numbers still evaluate to True for some dumb reason
            self.grad += (self.data > 0) * out.grad
        out.backward = backward
        return out
    
    def __matmul__(self, other): # almost identical to __mul__
        other = other if type(other) == Tensor else Tensor(other)
        out = Tensor(data = (self.data @ other.data), children = (self, other), op = '@')
        
        def backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out.backward = backward
        return out
    
    def conv(self, other, stride = 1): # TODO: enforce correct matrix dims?
        other = other if type(other) == Tensor else Tensor(other)
        (nc, nx, ny) = self.data.shape # NOTE: CHANNELS FIRST
        # ^ TODO: ADD M TO ALLOW VECTORIZATION (AND TRY TO USE SPLIT/CONCAT TO GET RID OF FOR LOOP)
        (fn, fc, fx, fy) = other.data.shape # NOTE: FILTERS & CHANNELS FIRST
        out = Tensor(data = np.zeros(shape = (fn, ((nx-fx)//stride)+1, ((ny-fy)//stride)+1)), op = 'conv') # NOTE: doesn't need children/backwards because it's basically a literal
        
        for f in range(fn):
            otherPadded = Tensor(data = np.zeros_like(self.data)).sliceAdd(other.slice((f)).flip(), (slice(fc), slice(fx), slice(fy)))
            out = out.sliceAdd((self.dftNd() * otherPadded.dftNd()).idftNd().slice((slice(0, 1, stride), slice(fx-1, nx, stride), slice(fy-1, ny, stride))), (slice(f, f+1), slice(out.data.shape[-2]), slice(out.data.shape[-1])))

        return out
    
    def avgPool(self, filter_size = 2, stride = 2):
        (nc, nx, ny) = self.data.shape
        out = Tensor(data = np.zeros(shape = (nc, nx//stride, ny//stride)), op = 'avgPool') # NOTE: doesn't need children/backwards because it's basically a literal
        filter = np.ones(shape = (1, 1, filter_size, filter_size))/(filter_size**2)
        for c in range(nc):
            out = out.sliceAdd(self.slice((slice(c, c+1))).conv(filter, stride = stride), (slice(c, c+1)))
        return out

    def maxPool2d(self, filter_size = 2, stride = 2): # NOTE: FORWARD PASS NO WORK
        (nc, nx, ny) = self.data.shape
        out = Tensor(data = np.ndarray(shape = (nc, nx//stride, ny//stride)), op = 'maxPool') # NOTE: doesn't need children/backwards because it's basically a literal
        for c in range(out.data.shape[0]):
            for x in range(out.data.shape[1]):
                for y in range(out.data.shape[2]):
                    out.sliceAdd(self.slice((slice(c, c+1), slice(x*stride, x*stride+filter_size), slice(y*stride, y*stride+filter_size))).max(), (slice(c, c+1), slice(x, x+1), slice(y, y+1)))
        return out
    
    def pad(self, slice_index_tuple):
        out = Tensor(data = self.data[slice_index_tuple], children = (self,), op = 'S')

        def backward():
            self.grad[slice_index_tuple] += out.grad
        out.backward = backward
        return out
    
    def split(self, slice_index_tuple):
        out = Tensor(data = self.data[slice_index_tuple], children = (self,), op = 'S')

        def backward():
            self.grad[slice_index_tuple] += out.grad
        out.backward = backward
        return out
    
    def concatenate(self, slice_index_tuple):
        out = Tensor(data = self.data[slice_index_tuple], children = (self,), op = 'S')

        def backward():
            self.grad[slice_index_tuple] += out.grad
        out.backward = backward
        return out

    def transpose(self): # shifts axes (0->1, 1->2, etc)
        dims = np.arange(len(self.data.shape))
        out = Tensor(data = np.transpose(self.data, axes = np.roll(dims, 1)), children = (self,), op = 'T')

        def backward():
            self.grad += np.transpose(out.grad, axes = np.roll(dims, -1))
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
    
    def slice(self, slice_index_tuple):
        out = Tensor(data = self.data[slice_index_tuple], children = (self,), op = 'S')

        def backward():
            self.grad[slice_index_tuple] += out.grad
        out.backward = backward
        return out

    def sliceAdd(self, other, slice_index_tuple): # TODO: check tensor dims? also is this correct?
        other = other if type(other) == Tensor else Tensor(other)
        out = Tensor(data = self.data, children = (self, other), op = 'S+')
        out.data[slice_index_tuple] += other.data

        def backward():
            self.grad += out.grad
            other.grad += out.grad[slice_index_tuple]
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
    
    def dftNd(self):
        out = self
        dims = np.arange(len(self.data.shape))
        for dim in dims:
            shape = out.data.shape
            dftMatrix = np.arange(shape[0]).reshape((-1, 1)) @ np.arange(shape[0]).reshape((1, -1))
            out = (out.reshape((-1, shape[0])) @ np.exp(-2j * np.pi * dftMatrix / shape[0])).reshape(shape).transpose()
        return out
    
    def idftNd(self):
        out = self
        dims = np.arange(len(self.data.shape))
        for dim in dims:
            shape = out.data.shape
            dftMatrix = np.arange(shape[0]).reshape((-1, 1)) @ np.arange(shape[0]).reshape((1, -1))
            out = ((out.reshape((-1, shape[0])) @ np.exp(2j * np.pi * dftMatrix / shape[0])) / shape[0]).reshape(shape).transpose()
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
            #print(node, ': ', [child.grad for child in node.children]) # for debug