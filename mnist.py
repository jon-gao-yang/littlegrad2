import torch
import numpy as np
from littlegrad2.engine import Tensor
import matplotlib.pyplot as plt
import csv
import time

#based on a practice assignment from Andrew Ng's "Advanced Learning Algorithms" Coursera course
def plot_kaggle_data(X, y, model, predict=False):
    #m, n = X.shape

    fig, axes = plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

    for i,ax in enumerate(axes.flat):
        # Select random indices
        #random_index = np.random.randint(m)
        random_index = np.random.randint(X.shape[0])
        
        # Select rows corresponding to the random indices and reshape the image
        X_random_reshaped = X[random_index].reshape((28,28))
        
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        yhat = None
        # Predict using the Neural Network
        if predict:
            probs, log_softmax = softmax(model(Tensor(X[random_index])))
            yhat = np.argmax(probs.data)
        
        # Display the label above the image
        ax.set_title(f"{int(y[random_index])},{yhat}",fontsize=10)
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=14)
    plt.show()

def write_kaggle_submission(model):
    X = np.empty((28000, 28*28), dtype = int)
    with open('digit-recognizer/test.csv', newline='\n') as csvfile:
        digitreader = csv.reader(csvfile, delimiter=',')
        for row in digitreader:
            if digitreader.line_num != 1: #line_num starts at 1, not 0
                X[digitreader.line_num-2] = [int(char) for char in row] #no labels so entire row is pixel data
    X = (X-np.average(X)) / np.std(X)  #data normalization

    with open('digit-recognizer/submission.csv', newline='\n', mode = 'w') as csvfile:
        digitwriter = csv.writer(csvfile, delimiter=',')
        digitwriter.writerow(['ImageId','Label'])
        for i in range(X.shape[0]):
            probs, log_softmax = softmax(model(Tensor(X[i])))
            digitwriter.writerow([i+1, np.argmax(probs.data)])  #take most likely digit as guess

def softmax(logits):
  counts = logits.exp()
  denominator = counts @ np.ones(shape = (counts.data.shape[-1], counts.data.shape[-1])) #2D ones matrix avoids denom broadcasting which fucks up gradient shape
  return counts / denominator, logits - denominator.log() #probs, log_softmax

#modified from karpathy's demo.ipynb
def loss(X, y, model, batch_size=None, regularization=True, alpha=1e-8):

    if batch_size is None:  #dataloader
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size] #shuffles the X indexes and returns the first 10
        Xb, yb = X[ri], y[ri]

    probs, log_softmax = softmax(model(Tensor(Xb))) # x --(model)--> logits --(softmax)--> probs --(-log)--> nll loss
    losses = Tensor(np.zeros_like(probs.data))
    losses.data[np.arange(probs.data.shape[0]), yb] = -1
    losses = (losses * log_softmax).flatten() @ Tensor(np.ones((probs.data.size, 1))) / probs.data.shape[0]
    accuracy = np.average(np.argmax(probs.data, axis = -1) == yb)

    if regularization: # L2 regularization (total_loss = data_loss + reg_loss)
        losses += alpha * np.sum([p.reshape((1, -1))@p.reshape((-1, 1)) for p in model.parameters()])
    return losses, accuracy

def kaggle_training(epochs = 10, batch_size = None, regularization = True, learning_rate = 0.0001, alpha = 1e-8):
    X = np.empty((42000, 28*28), dtype = int)
    y = np.empty(42000, dtype = int)
    with open('digit-recognizer/train.csv', newline='\n') as csvfile:
        digitreader = csv.reader(csvfile, delimiter=',')
        for row in digitreader:
            if digitreader.line_num != 1: #line_num starts at 1, not 0
                y[digitreader.line_num-2] = int(row[0])
                X[digitreader.line_num-2] = [int(char) for char in row[1:]]
    X = (X-np.average(X)) / np.std(X)  #data normalization

    class TestNet:
        def __init__(self):
            
            self.params = { # 2 / (# of inputs from last layer) for He initialization
                'f1' : Tensor(np.random.randn(6, 1, 5, 5) * np.sqrt(2 / (28*28*1))),
                'f2' : Tensor(np.random.randn(16, 6, 5, 5) * np.sqrt(2 / (12*12*6))),

                'w1' : Tensor(np.random.randn(24*24*6, 64) * np.sqrt(2 / (24*24*6))),
                #'w1' : Tensor(np.random.randn(4*4*16, 64) * np.sqrt(2 / (4*4*16))),
                'b1' : Tensor(np.zeros((1, 64))),
                'w2' : Tensor(np.random.randn(64, 10) * np.sqrt(2 / (64))),
                'b2' : Tensor(np.zeros((1, 10))),
            }

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
        
        def param_num(self):
            return np.sum([t.data.size for t in self.params.values()])

        def zero_grad(self):
            for param in self.params.values():
                param.grad.fill(0)

        def __call__(self, x:Tensor) -> Tensor:
            l2 = x.reshape((-1, 28, 28)).conv(self.params['f1']).relu()
            #l1 = x.reshape((1, 28, 28)).conv(self.params['f1']).relu().avgPool()
            #l2 = l1.conv(self.params['f2']).relu().avgPool()
            l3 = ((l2.flatten() @ self.params['w1']) + self.params['b1']).relu()
            return((l3 @ self.params['w2']) + self.params['b2'])
        
        def __call__(self, x:Tensor) -> Tensor:
            l1 = ((x @ self.params['w1']) + self.params['b1']).relu()
            l2 = ((l1 @ self.params['w2']) + self.params['b2']).relu()
            l3 = ((l2 @ self.params['w3']) + self.params['b3']).relu()
            l4 = ((l3 @ self.params['w4']) + self.params['b4']).relu()
            return (l4 @ self.params['w5']) + self.params['b5']

    # initialize a model 
    class ConvNet:
        def __init__(self):
            
            self.params = { # 2 / (# of inputs from last layer) for He initialization
                # 'f1' : Tensor(np.random.randn(5, 5, 1, 6) * np.sqrt(2 / (28*28*1))),
                # 'f2' : Tensor(np.random.randn(5, 5, 6, 16) * np.sqrt(2 / (12*12*6))),

                # 'f1d' : Tensor(np.random.randn(5, 5, 1) * np.sqrt(2 / (28*28*1))),
                # 'f1p' : Tensor(np.random.randn(1, 1, 1, 6) * np.sqrt(2 / (24*24*1))),
                # 'f2d' : Tensor(np.random.randn(5, 5, 6) * np.sqrt(2 / (12*12*6))),
                # 'f2p' : Tensor(np.random.randn(1, 1, 6, 16) * np.sqrt(2 / (8*8*6))),

                'f1' : Tensor(np.random.randn(6, 1, 5, 5) * np.sqrt(2 / (28*28*1))),
                'f2' : Tensor(np.random.randn(16, 6, 5, 5) * np.sqrt(2 / (12*12*6))),

                'w1' : Tensor(np.random.randn(4*4*16, 64) * np.sqrt(2 / (4*4*16))),
                'b1' : Tensor(np.zeros((1, 64))),
                'w2' : Tensor(np.random.randn(64, 16) * np.sqrt(2 / (64))),
                'b2' : Tensor(np.zeros((1, 16))),
                'w3' : Tensor(np.random.randn(16, 10) * np.sqrt(2 / (16))),
                'b3' : Tensor(np.zeros((1, 10))),
            }

        def parameters(self):
            return self.params.values()
        
        def zero_grad(self):
            for param in self.params.values():
                param.grad.fill(0)

        def __call__(self, x:Tensor) -> Tensor:
            l1 = x.reshape((1, 28, 28)).conv(self.params['f1']).relu().maxPool2d()
            l2 = l1.conv(self.params['f2']).relu().maxPool2d()
            
            #l1 = x.dconv(self.params['f1d']).conv(self.params['f1p']).relu().maxPool2d() #depthwise separable convolution
            #l2 = l1.dconv(self.params['f2d']).conv(self.params['f2p']).relu().maxPool2d() #depthwise separable convolution
            l3 = ((l2.flatten() @ self.params['w1']) + self.params['b1']).relu()
            l4 = ((l3 @ self.params['w2']) + self.params['b2']).relu()
            return (l4 @ self.params['w3']) + self.params['b3']
        
    class LinearNet:
        def __init__(self):
            
            self.params = {                
                'w1' : Tensor(np.random.randn(28*28, 320) * np.sqrt(2 / (28*28))), # 2 / (# of inputs from last layer)
                'b1' : Tensor(np.zeros((1, 320))),
                'w2' : Tensor(np.random.randn(320, 160) * np.sqrt(2 / 320)), # 2 / (# of inputs from last layer)
                'b2' : Tensor(np.zeros((1, 160))),
                'w3' : Tensor(np.random.randn(160, 80) * np.sqrt(2 / 160)), # 2 / (# of inputs from last layer)
                'b3' : Tensor(np.zeros((1, 80))),
                'w4' : Tensor(np.random.randn(80, 40) * np.sqrt(2 / 80)), # 2 / (# of inputs from last layer)
                'b4' : Tensor(np.zeros((1, 40))),
                'w5' : Tensor(np.random.randn(40, 20) * np.sqrt(2 / 40)), # 2 / (# of inputs from last layer)
                'b5' : Tensor(np.zeros((1, 20))),
                'w6' : Tensor(np.random.randn(20, 10) * np.sqrt(2 / 20)), # 2 / (# of inputs from last layer)
                'b6' : Tensor(np.zeros((1, 10)))
            }

        def parameters(self):
            return self.params.values()
        
        def zero_grad(self):
            for param in self.params.values():
                param.grad.fill(0)

        def __call__(self, x:Tensor) -> Tensor:
            l1 = ((x @ self.params['w1']) + self.params['b1']).relu()
            l2 = ((l1 @ self.params['w2']) + self.params['b2']).relu()
            l3 = ((l2 @ self.params['w3']) + self.params['b3']).relu()
            l4 = ((l3 @ self.params['w4']) + self.params['b4']).relu()
            l5 = ((l4 @ self.params['w5']) + self.params['b5']).relu()
            return (l5 @ self.params['w6']) + self.params['b6']
    
    #model = ConvNet()
    model = TestNet()
    #learning_rate, beta1, beta2, epsilon, weight_decay = 0.0001, 0.9, 0.999, 1e-10, 0.01 #NOTE: cost will not converge if learning rate is too high
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
        
        #update parameters w/ AdamW Algorithm
        for p in model.parameters(): 
            p.data -= p.data * learning_rate * weight_decay
            p.v = (beta1 * p.v) + ((1-beta1) * p.grad)
            p.s = (beta2 * p.s) + ((1-beta2) * p.grad * p.grad)
            v_dp_corrected = p.v / (1 - (beta1**(k+1)))
            s_dp_corrected = p.s / (1 - (beta2**(k+1)))
            p.data -= learning_rate * v_dp_corrected / (np.sqrt(s_dp_corrected) + epsilon) #doesn't work for broadasted bias v/s/grad tensors
            
            #TODO: FIX THIS
            # if p.data.shape == p.grad.shape:
            #     p.data -= learning_rate * v_dp_corrected / (np.sqrt(s_dp_corrected) + epsilon)
            # else:
            #     p.data -= np.average(learning_rate * v_dp_corrected / (np.sqrt(s_dp_corrected) + epsilon), axis = 0, keepdims = True)
        
        print(f"step {k} loss {total_loss.data.real[0, 0]}, accuracy {acc*100}%")

    endTime = time.time()
    print('TRAINING COMPLETE (in', endTime - startTime, 'sec)')
    plot_kaggle_data(X, y, model, predict = True)
    #print('BEGINNING TEST SET INFERENCE')
    #write_kaggle_submission(model)
    #print('TEST SET INFERENCE COMPLETE')

#############################################################################################

#kaggle_training(epochs = 100, batch_size = 50, regularization = False, learning_rate = 0.0001)
#kaggle_training(epochs = 100, batch_size = 100, regularization = True, learning_rate = 0.0006, alpha = 1e-6)


#.0058
kaggle_training(epochs = 210, batch_size = 1000, regularization = False, learning_rate = 0.00468, alpha = 0)