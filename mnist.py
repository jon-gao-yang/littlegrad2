#import torch
import numpy as np
from littlegrad2.engine import Tensor
import matplotlib.pyplot as plt
#import csv
import time

###### [ 1/4 : MODEL INITIALIZATION ] ######

class TestNet:
    def __init__(self):
        
        self.params = { # 2 / (# of inputs from last layer) for He initialization
            'f1' : Tensor(np.random.randn(6, 1, 5, 5) * np.sqrt(2 / (28*28*1))),
            'f2' : Tensor(np.random.randn(16, 6, 5, 5) * np.sqrt(2 / (12*12*6))),

            'w1' : Tensor(np.random.randn(4*4*16, 64) * np.sqrt(2 / (4*4*16))),
            'b1' : Tensor(np.zeros((1, 64))),
            'w2' : Tensor(np.random.randn(64, 10) * np.sqrt(2 / (64))),
            'b2' : Tensor(np.zeros((1, 10))),
        }

    def __call__(self, x:Tensor) -> Tensor:
        l1 = x.reshape((-1, 1, 28, 28)).conv(self.params['f1']).relu().maxPool2d()
        l2 = l1.conv(self.params['f2']).relu().maxPool2d()
        l3 = ((l2.reshape((-1, self.params['w1'].data.shape[0])) @ self.params['w1']) + self.params['b1']).relu()
        return ((l3 @ self.params['w2']) + self.params['b2'])

    def parameters(self):
        return self.params.values()
    
    def param_num(self):
        return np.sum([t.data.size for t in self.params.values()])

    def zero_grad(self):
        for param in self.params.values():
            param.grad.fill(0)

class ConvNet:
    def __init__(self):
        
        self.params = { # 2 / (# of inputs from last layer) for He initialization
            'f1' : Tensor(np.random.randn(8, 1, 5, 5) * np.sqrt(2 / (28*28*1))),
            'f1b' : Tensor(np.zeros((1, 8, 1, 1))),
            'f2' : Tensor(np.random.randn(16, 8, 5, 5) * np.sqrt(2 / (12*12*8))),
            'f2b' : Tensor(np.zeros((1, 16, 1, 1))),

            'w1' : Tensor(np.random.randn(4*4*16, 64) * np.sqrt(2 / (4*4*16))),
            'b1' : Tensor(np.zeros((1, 64))),
            'w2' : Tensor(np.random.randn(64, 10) * np.sqrt(2 / (64))),
            'b2' : Tensor(np.zeros((1, 10))),
        }

    def __call__(self, x:Tensor) -> Tensor:
        l1 = (x.reshape((-1, 1, 28, 28)).conv(self.params['f1']) + self.params['f1b']).maxPool2d().relu()
        l2 = (l1.conv(self.params['f2']) + self.params['f2b']).maxPool2d().relu()
        l3 = ((l2.reshape((-1, self.params['w1'].data.shape[0])) @ self.params['w1']) + self.params['b1']).relu()
        return ((l3 @ self.params['w2']) + self.params['b2'])

    def parameters(self):
        return self.params.values()
    
    def zero_grad(self):
        for param in self.params.values():
            param.grad.fill(0)

    def param_num(self):
        return np.sum([t.data.size for t in self.params.values()])
    
class LinearNet:
    def __init__(self):
        
        self.params = {                
            'w1' : Tensor(np.random.randn(28*28, 160) * np.sqrt(2 / (28*28))), # 2 / (# of inputs from last layer)
            'b1' : Tensor(np.zeros((1, 160))),
            'w2' : Tensor(np.random.randn(160, 80) * np.sqrt(2 / 160)), # 2 / (# of inputs from last layer)
            'b2' : Tensor(np.zeros((1, 80))),
            'w3' : Tensor(np.random.randn(80, 40) * np.sqrt(2 / 80)), # 2 / (# of inputs from last layer)
            'b3' : Tensor(np.zeros((1, 40))),
            'w4' : Tensor(np.random.randn(40, 20) * np.sqrt(2 / 40)), # 2 / (# of inputs from last layer)
            'b4' : Tensor(np.zeros((1, 20))),
            'w5' : Tensor(np.random.randn(20, 10) * np.sqrt(2 / 20)), # 2 / (# of inputs from last layer)
            'b5' : Tensor(np.zeros((1, 10)))
        }

    def parameters(self):
        return self.params.values()
    
    def zero_grad(self):
        for param in self.params.values():
            param.grad.fill(0)

    def param_num(self):
        return np.sum([t.data.size for t in self.params.values()])

    def __call__(self, x:Tensor) -> Tensor:
        l1 = ((x @ self.params['w1']) + self.params['b1']).relu()
        l2 = ((l1 @ self.params['w2']) + self.params['b2']).relu()
        l3 = ((l2 @ self.params['w3']) + self.params['b3']).relu()
        l4 = ((l3 @ self.params['w4']) + self.params['b4']).relu()
        return (l4 @ self.params['w5']) + self.params['b5']
    
###### [ 2/4 : HELPER FUNCTIONS ] ######

# based on code from Andrew Ng's "Advanced Learning Algorithms" Coursera course
def plot_kaggle_data(X, y, model, predict=False):
    fig, axes = plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(X.shape[0])
        
        # Display the image
        ax.imshow(X[random_index].reshape(28, 28), cmap='gray')

        yhat = None
        # Predict using the Neural Network
        if predict:
            probs, log_softmax = softmax(model(Tensor(X[random_index])))
            yhat = np.argmax(probs.data)
        
        # Display the label above the image
        ax.set_title(f"{y[random_index]},{yhat}",fontsize=10)
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=14)
    plt.show()

def write_kaggle_submission(model):
    X = np.loadtxt('digit-recognizer/test.csv', dtype = int, delimiter = ',', skiprows = 1) # data loading
    X = X/255  # data normalization

    probs, log_softmax = softmax(model(Tensor(X))) # inference
    out = np.concatenate((np.arange(1, X.shape[0]+1).reshape((-1, 1)), np.argmax(probs.data, axis = 1).reshape((-1, 1))), axis = 1)
    np.savetxt('digit-recognizer/submission.csv', out, delimiter = ',', fmt = '%s', header = 'ImageId,Label', comments = '')

def softmax(logits):
  counts = logits.exp()
  denominator = counts @ np.ones(shape = (counts.data.shape[-1], counts.data.shape[-1])) #2D ones matrix avoids denom broadcasting which fucks up gradient shape
  return counts / denominator, logits - denominator.log() #probs, log_softmax

# based on code from Andrej Karpathy's "Micrograd" github repo
def loss(X, y, model, batch_size=None, regularization=True, alpha=1e-8):

    if batch_size is None:  #dataloader
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size] # shuffles the X indexes and returns the first 10
        Xb, yb = X[ri], y[ri]

    # x --(model)--> logits --(softmax)--> probs --(-log)--> nll loss --(avg over batch)--> cost --(backprop)--> grads
    probs, log_softmax = softmax(model(Tensor(Xb))) 
    losses = Tensor(np.zeros_like(probs.data))
    losses.data[np.arange(probs.data.shape[0]), yb] = -1
    losses = (losses * log_softmax).flatten() @ Tensor(np.ones((probs.data.size, 1))) / probs.data.shape[0]
    accuracy = np.average(np.argmax(probs.data, axis = -1) == yb)

    if regularization: # L2 regularization (total_loss = data_loss + reg_loss)
        losses += alpha * np.sum([p.reshape((1, -1))@p.reshape((-1, 1)) for p in model.parameters()])
    return losses, accuracy

###### [ 3/4 : MAIN FUNCTION ] ######

def kaggle_training(model, epochs = 10, batch_size = None, regularization = True, learning_rate = 0.0001, alpha = 1e-8):
    [y, X] = np.split(np.loadtxt('digit-recognizer/train.csv', dtype = int, delimiter = ',', skiprows = 1), [1], axis = 1)
    # ^ NOTE: loading data from file, then splitting into labels (first col) and pixel vals
    y = np.squeeze(y) # 2D -> 1D
    X = X/255  # data normalization
    beta1, beta2, epsilon, weight_decay = 0.9, 0.999, 1e-10, 0.01
    print('TRAINING BEGINS (with', model.param_num(), 'parameters)')
    startTime = time.time()

    # optimization
    for k in range(epochs):
        
        # forward
        total_loss, acc = loss(X, y, model, batch_size = batch_size, regularization = regularization, alpha = alpha)

        # backward
        model.zero_grad()
        total_loss.backprop()
        
        # update parameters w/ AdamW Algorithm
        for p in model.parameters(): 
            #p.data -= learning_rate * p.grad # TODO: ALLOW OPTIMIZER CHOICE
            p.data -= p.data * learning_rate * weight_decay
            p.v = (beta1 * p.v) + ((1-beta1) * p.grad)
            p.s = (beta2 * p.s) + ((1-beta2) * p.grad * p.grad)
            v_dp_corrected = p.v / (1 - (beta1**(k+1)))
            s_dp_corrected = p.s / (1 - (beta2**(k+1)))
            p.data -= learning_rate * v_dp_corrected / (np.sqrt(s_dp_corrected) + epsilon) #doesn't work for broadasted bias v/s/grad tensors

        print(f"step {k} loss {total_loss.data.real[0, 0]}, accuracy {acc*100}%")

    endTime = time.time()
    print('TRAINING COMPLETE (in', endTime - startTime, 'sec)')
    plot_kaggle_data(X, y, model, predict = True)
    print('BEGINNING TEST SET INFERENCE')
    write_kaggle_submission(model)
    print('TEST SET INFERENCE COMPLETE')

###### [ 4/4 : MAIN FUNCTION EXECUTION ] ###### 
# NOTE: cost will not converge if learning rate is too high
# NOTE: REMEMBER TO CHANGE SELF.TYPE IN ENGINE.PY IF SWITCHING FROM CONV NET TO LINEAR NET

# for ConvNet() (NOTE: set self.type = complex):
#kaggle_training(model = ConvNet(), epochs = 500, batch_size = 200, regularization = False, learning_rate = 0.0005, alpha = 0)

# for LinearNet() (NOTE: set self.type = float):
kaggle_training(model = LinearNet(), epochs = 300, batch_size = 500, regularization = False, learning_rate = 0.0007898, alpha = 0)
