import numpy as np

class Tensor:
    def __init__(self, data = [], children = (), op = ''):
        self.children = children
        self.backward = lambda: None
        self._op = op
        #self.type = np.complex128 # for ConvNet()
        self.type = float  # for LinearNet() (MUCH faster than complex128)
        self.data = np.atleast_2d(data.astype(self.type)) if isinstance(data, np.ndarray) else np.atleast_2d(np.array(object = data, dtype = self.type))
        # ^ NOTE: np.atleast_2d() reshapes 1d arrays to (1, -1)
        self.grad, self.v, self.s = np.zeros_like(self.data), np.zeros_like(self.data), np.zeros_like(self.data)

    def __repr__(self):
        return f"Tensor object with data {self.data}"
    
    def __add__(self, other):
        (self, other) = Tensor.makeCompatible(self, other, same_size = True)
        out = Tensor(data = (self.data + other.data), children = (self, other), op = '+')

        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out.backward = backward
        return out
    
    def __mul__(self, other):
        (self, other) = Tensor.makeCompatible(self, other, same_size = True)
        out = Tensor(data = (self.data * other.data), children = (self, other), op = '*')
        
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.backward = backward
        return out
    
    def __pow__(self, other):
        (self, other) = Tensor.makeCompatible(self, other, same_size = True)
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
        (self, other) = Tensor.makeCompatible(self, other, same_size = True)
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
        (self, other) = Tensor.makeCompatible(self, other, same_size = False)
        out = Tensor(data = (self.data @ other.data), children = (self, other), op = '@')
        
        def backward():
            self.grad += out.grad @ other.data.swapaxes(-2, -1)
            other.grad += self.data.swapaxes(-2, -1) @ out.grad
        out.backward = backward
        return out
    
    def conv(self, other, stride = 1): # TODO: enforce correct matrix dims? DELETE UNNECESSARY TENSOR METHODS
        other = other if type(other) == Tensor else Tensor(other)
        (m, nc, nx, ny) = self.data.shape # NOTE: CHANNELS FIRST
        (fn, fc, fx, fy) = other.data.shape # NOTE: FILTERS & CHANNELS FIRST

        axes, pad_width_tuple = (2, 3, 4), ((0, 0), (0, 0), (0, 0), (0, nx-fx), (0, ny-fy))
        selfData =  np.fft.fftn(self.data.reshape((m, 1, nc, nx, ny)), axes = axes)
        otherData = np.fft.fftn(np.pad(np.flip(other.data.reshape((1, fn, fc, fx, fy)), axes), pad_width_tuple), axes = axes)
        out = Tensor(data = np.fft.ifftn(selfData * otherData, axes = axes)[:, :, -1, fx-1:nx:stride, fy-1:ny:stride].real, children = (self, other), op = 'conv')

        def backward():
            for x in range(out.grad.shape[-2]):
                for y in range(out.grad.shape[-1]):
                    other.grad += np.average(self.data[..., x*stride:x*stride+fx, y*stride:y*stride+fy], axis = 0, keepdims = True) * np.average(out.grad[..., x, y], axis = 0).reshape((-1, 1, 1, 1))
        out.backward = backward
        return out
    
    def maxPool2d(self, filter_size = 2, stride = 2):
        (m, nc, nx, ny) = self.data.shape
        maxIndices = np.ndarray((4, m, nc, int(np.ceil(nx/stride)), int(np.ceil(ny/stride))), dtype = int)
        
        for x in range(maxIndices.shape[-2]):
            for y in range(maxIndices.shape[-1]):
                xSlice = slice(stride*x, stride*x+filter_size) if stride*x+filter_size <= nx else slice(stride*x, nx)
                ySlice = slice(stride*y, stride*y+filter_size) if stride*y+filter_size <= ny else slice(stride*y, ny)
                
                maxIndices[-2, ..., x, y] = np.argmax(np.max(self.data[..., xSlice, ySlice], axis = -1), axis = -1) + (x*filter_size)
                maxIndices[-1, ..., x, y] = np.argmax(np.max(self.data[..., xSlice, ySlice], axis = -2), axis = -1) + (y*filter_size)
        np.copyto(dst = maxIndices[0], src = np.arange(m).reshape((m, 1, 1, 1)))
        np.copyto(dst = maxIndices[1], src = np.arange(nc).reshape((1, nc, 1, 1)))
        return self.slice((maxIndices[0], maxIndices[1], maxIndices[2], maxIndices[3]))
    
    # def pad(self, other_shape, pad_width_tuple): # pad_width_tuple is a tuple of tuples for each axis, ex: ((before_1, after_1), ..., (before_N, after_N)) 
    #     out = Tensor(data = np.pad(self.data, pad_width_tuple), children = (self,), op = 'pad')

    #     def backward(): # TODO: get rid of for loop? ALSO ASSERT THE TWO INPUT SHAPES HAVE THE SAME LEN
    #         self.grad += out.grad[tuple([slice(pad_width_tuple[i][0], pad_width_tuple[i][0]+self.data.shape[i]) for i in np.arange(len(self.data.shape))])]
    #     out.backward = backward
    #     return out

    # def transpose(self): # shifts axes (0->1, 1->2, etc)
    #     dims = np.arange(len(self.data.shape))
    #     out = Tensor(data = np.transpose(self.data, axes = np.roll(dims, 1)), children = (self,), op = 'T')

    #     def backward():
    #         self.grad += np.transpose(out.grad, axes = np.roll(dims, -1))
    #     out.backward = backward
    #     return out
    
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

    # def sliceAdd(self, other, slice_index_tuple): # TODO: check tensor dims? also make sure this doesn't mess up backprop
    #     other = other if type(other) == Tensor else Tensor(other)
    #     out = Tensor(data = self.data, children = (self, other), op = 'S+')
    #     out.data[slice_index_tuple] += other.data

    #     def backward():
    #         self.grad += out.grad
    #         other.grad += out.grad[slice_index_tuple]
    #     out.backward = backward
    #     return out
    
    # def flip(self, axis = None):
    #     out = Tensor(data = np.flip(self.data, axis = axis), children = (self,), op = 'flip')

    #     def backward():
    #         self.grad += np.flip(out.grad, axis = axis)
    #     out.backward = backward
    #     return out
    
    # def dftNd(self, axis = ()):
    #     out = self
    #     axis = axis if (len(axis) > 0) else np.arange(len(self.data.shape))

    #     for dim in np.flip(np.arange(len(self.data.shape))):
    #         if (dim in axis) and (self.data.shape[dim] > 1):
    #             dftMatrix = np.arange(self.data.shape[dim]).reshape((-1, 1)) @ np.arange(self.data.shape[dim]).reshape((1, -1))
    #             out = (out @ np.exp(-2j * np.pi * dftMatrix / self.data.shape[dim])).transpose()
    #         else:
    #             out = out.transpose()
    #     return out
    
    # def idftNd(self, axis = ()):
    #     out = self
    #     axis = axis if (len(axis) > 0) else np.arange(len(self.data.shape))

    #     for dim in np.flip(np.arange(len(self.data.shape))):
    #         if (dim in axis) and (self.data.shape[dim] > 1):
    #             dftMatrix = np.arange(self.data.shape[dim]).reshape((-1, 1)) @ np.arange(self.data.shape[dim]).reshape((1, -1))
    #             out = ((out @ np.exp(2j * np.pi * dftMatrix / self.data.shape[dim])) / self.data.shape[dim]).transpose()
    #         else:
    #             out = out.transpose()
    #     return out

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
    
    def tile(self, reps):
        out = Tensor(data = np.tile(self.data, reps), children = (self,), op = 'tile')

        def backward():
            self.grad += np.average(out.grad, axis = tuple(np.arange(len(reps))[(reps > 1)]), keepdims = True) if np.any((reps > 1)) else out.grad # boolean array indexing
        out.backward = backward
        return out
    
    def makeCompatible(self, other, same_size): # manual broadcasting of 'self' and 'other' to keep track of grads
        other = other if isinstance(other, Tensor) else Tensor(other) # make sure 'other' is Tensor

        # make sure self and other have same length
        if len(self.data.shape) > len(other.data.shape):
            other = other.reshape(np.insert(np.array(other.data.shape), 0, np.ones(len(self.data.shape) - len(other.data.shape))))
        elif len(self.data.shape) < len(other.data.shape):
            self = self.reshape(np.insert(np.array(self.data.shape), 0, np.ones(len(other.data.shape) - len(self.data.shape))))

        # get broadcast repetition counts for each axis that needs broadcasting and make min broadcast count 1
        selfShape, otherShape = np.array(self.data.shape), np.array(other.data.shape)
        otherBroadcastDims = (selfShape > 1) * (otherShape == 1) * selfShape # get broadcast counts
        otherBroadcastDims += (otherBroadcastDims == 0)                      # make min broadcast count 1
        selfBroadcastDims = (otherShape > 1) * (selfShape == 1) * otherShape # get broadcast counts
        selfBroadcastDims += (selfBroadcastDims == 0)                        # make min broadcast count 1

        # if matmul instead of element-wise op, make inner dims compatible
        if not same_size:
            selfBroadcastDims[-2:] = [1, 1]
            otherBroadcastDims[-2:] = [1, 1]
            if self.data.shape[-1] == 1:
                selfBroadcastDims[-1] = other.data.shape[-2] # make inner dims compatible
            elif other.data.shape[-2] == 1:
                otherBroadcastDims[-2] = self.data.shape[-1] # make inner dims compatible

        # if any axes need broadcasting, broadcast them w/ np.tile before returning Tensors
        if np.any(selfBroadcastDims > 1):
           self = self.tile(selfBroadcastDims)
        if np.any(otherBroadcastDims > 1):
            other = other.tile(otherBroadcastDims)
        return (self, other)
    
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