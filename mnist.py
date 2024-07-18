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
    X = X.reshape(28000, 28, 28, 1)    #reshape for ConvNet

    with open('digit-recognizer/submission.csv', newline='\n', mode = 'w') as csvfile:
        digitwriter = csv.writer(csvfile, delimiter=',')
        digitwriter.writerow(['ImageId','Label'])
        for i in range(X.shape[0]):
            probs, log_softmax = softmax(model(Tensor(X[i])))
            digitwriter.writerow([i+1, np.argmax(probs.data)])  #take most likely digit as guess

def softmax(logits):
  counts = logits.exp()
  denominator = counts @ np.ones(shape = (counts.data.size, counts.data.size)) #2D ones matrix avoids denom broadcasting which fucks up gradient shape
  return counts / denominator, logits - denominator.log() #probs, log_softmax

#modified from karpathy's demo.ipynb
def loss(X, y, model, batch_size=None, regularization=True):

    if batch_size is None:  #dataloader
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size] #shuffles the X indexes and returns the first 10
        Xb, yb = X[ri], y[ri]

    losses, accuracy = [], []
    for (xrow, yrow) in zip(Xb, yb):
        probs, log_softmax = softmax(model(Tensor(xrow)))        
        losses.append(-log_softmax @ Tensor([index == yrow for index in range(log_softmax.data.size)]).transpose())
        # ^ cross entropy loss (can't just take log_softmax[yrow] or else you lose track of gradients and backward() doesn't work)
        accuracy.append(yrow == np.argmax(probs.data))

    if regularization:
        # L2 regularization
        alpha = 0.0001
        reg_loss = alpha * sum([p.flatten()@p.flatten().transpose() for p in model.parameters()])
        return np.average(losses) + reg_loss, np.average(accuracy) # (total_loss = data_loss + reg_loss)
    return np.average(losses), np.average(accuracy)

def kaggle_training(epochs = 10, batch_size = None, regularization = True):
    X = np.empty((42000, 28*28), dtype = int)
    y = np.empty(42000, dtype = int)
    with open('digit-recognizer/train.csv', newline='\n') as csvfile:
        digitreader = csv.reader(csvfile, delimiter=',')
        for row in digitreader:
            if digitreader.line_num != 1: #line_num starts at 1, not 0
                y[digitreader.line_num-2] = int(row[0])
                X[digitreader.line_num-2] = [int(char) for char in row[1:]]
    
    X = (X-np.average(X)) / np.std(X)  #data normalization
    X = X.reshape(42000, 28, 28, 1)    #reshape for ConvNet

    # initialize a model 
    class ConvNet:
        def __init__(self):
            
            self.params = { # 2 / (# of inputs from last layer) for He initialization
                'f1' : Tensor(np.random.randn(5, 5, 1, 6) * np.sqrt(2 / (28*28*1))),
                'f2' : Tensor(np.random.randn(5, 5, 6, 16) * np.sqrt(2 / (12*12*6))),

                # 'f1d' : Tensor(np.random.randn(5, 5, 1) * np.sqrt(2 / (28*28*1))),
                # 'f1p' : Tensor(np.random.randn(1, 1, 1, 6) * np.sqrt(2 / (24*24*1))),
                # 'f2d' : Tensor(np.random.randn(5, 5, 6) * np.sqrt(2 / (12*12*6))),
                # 'f2p' : Tensor(np.random.randn(1, 1, 6, 16) * np.sqrt(2 / (8*8*6))),

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
            l1 = x.conv2d(self.params['f1']).relu().maxPool2d()
            l2 = l1.conv2d(self.params['f2']).relu().maxPool2d()
            
            #l1 = x.dconv2d(self.params['f1d']).conv2d(self.params['f1p']).relu().maxPool2d() #depthwise separable convolution
            #l2 = l1.dconv2d(self.params['f2d']).conv2d(self.params['f2p']).relu().maxPool2d() #depthwise separable convolution
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
    
    model = ConvNet()
    learning_rate, beta1, beta2, epsilon, weight_decay = 0.001, 0.9, 0.999, 1e-10, 0.01 #NOTE: cost will not converge if learning rate is too high
    print('TRAINING BEGINS')
    startTime = time.time()

    # optimization
    for k in range(epochs):
        
        # forward
        total_loss, acc = loss(X, y, model, batch_size = batch_size, regularization = regularization)

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
            p.data -= learning_rate * v_dp_corrected / (np.sqrt(s_dp_corrected) + epsilon)
        
        if k % 1 == 0:
            print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")

    endTime = time.time()
    print('TRAINING COMPLETE (in', round((endTime - startTime) / 60, 3), 'min)')
    plot_kaggle_data(X, y, model, predict = True)
    #print('BEGINNING TEST SET INFERENCE')
    #write_kaggle_submission(model)
    #print('TEST SET INFERENCE COMPLETE')

#############################################################################################

kaggle_training(epochs = 10, batch_size = 10, regularization = False)