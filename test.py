import torch
import numpy as np
from littlegrad2.engine import Tensor
import matplotlib.pyplot as plt
import csv

def my_test():
    a = [2, 3]
    b = [5, 10]
    c = 4

    #a = -5
    #b = 3
    #c = 1

    aVal = Tensor(a)
    bVal = Tensor(b)

    a = np.array(a).reshape((1,-1))
    b = np.array(b).reshape((1,-1))

    print("a:      | Passed == ", type(aVal) == Tensor)
    print("b:      | Passed == ", type(bVal) == Tensor)
    print("a.data: | Passed == ", aVal.data == a)
    print("b.data: | Passed == ", bVal.data == b)
    print("a+c:    | Passed == ", (aVal+c).data == a+c)
    print("c+a:    | Passed == ", (c+aVal).data == c+a)
    print("a-c:    | Passed == ", (aVal-c).data == a-c)
    print("c-a:    | Passed == ", (c-aVal).data == c-a)
    print("a*c:    | Passed == ", (aVal*c).data == a*c)
    print("c*a:    | Passed == ", (c*aVal).data == c*a)
    print("-a:     | Passed == ", (-aVal).data == -a)
    print("a/c:    | Passed == ", (aVal/c).data == a/c)
    print("c/a:    | Passed == ", (c/aVal).data == c/a)
    print("a**c:   | Passed == ", (aVal**c).data == a**c)
    print("c**a:   | Passed == ", (c**aVal).data == c**a)
    print()

# from karpathy/micrograd test_engine.py:
def test_sanity_check():

    x = Tensor(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backprop()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    #print('xmg.grad: ', xmg.grad)
    #print('xpt.grad.item(): ', xpt.grad.item())
    assert xmg.grad == xpt.grad.item()
    print('Karpathy #1: Passed == True')

def test_more_ops():

    a = Tensor(-4.0)
    b = Tensor(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backprop()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
    print('Karpathy #2: Passed == True')
    print()

#based on a practice assignment from Andrew Ng's "Advanced Learning Algorithms" Coursera course
def plot_kaggle_data(X, y, model, predict=False):
    m, n = X.shape

    fig, axes = plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)
        
        # Select rows corresponding to the random indices and reshape the image
        X_random_reshaped = X[random_index].reshape((28,28))
        
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        yhat = None
        # Predict using the Neural Network
        if predict:
            #inputs = [list(map(Value, X[random_index]))] # pick random image and convert pixel ints to Value, 
            #outputs = list(map(model, inputs))[0]        # then plug pixel Values into model for fwd pass ([0] because list(map) returns a list of lists which breaks softmax)             
            #probs, log_softmax = softmax(outputs)
            probs, log_softmax = softmax(model(Tensor(X[random_index])))
            yhat = np.argmax(probs)
        
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
    with open('digit-recognizer/submission.csv', newline='\n') as csvfile:
        digitwriter = csv.writer(csvfile, delimiter=',')
        digitwriter.writerow(['ImageId','Label'])
        for i in range(X.shape[0]):
            #inputs = list(map(Value, X[i]))  #Value(X[i])
            #outputs = model(inputs)  #forward pass
            #probs, log_softmax = softmax(outputs)  #softmax layer
            probs, log_softmax = softmax(model(Tensor(X[i])))
            digitwriter.writerow([i+1, np.argmax(probs)])  #take most likely digit as guess

#from karpathy's micrograd_exercises.ipynb
#subtracting by max to avoid overflow (rounding logit.exp() to inf)
#adding 1 to prevent log(0) due to underflow
def softmax(logits):
  counts = [(logit-np.max(logits)+1).exp() for logit in logits.data] #DELETE .DATA?
  #denominator = sum(counts)
  probs = [count / sum(counts) for count in counts]
  log_softmax = [(logit-np.max(logits)+1) - sum(counts).log() for logit in logits]
  return probs, log_softmax

#modified from karpathy's demo.ipynb
def loss(X, y, model, batch_size=None):

    if batch_size is None:  #dataloader
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size] #shuffles the X indexes and returns the first 10
        Xb, yb = X[ri], y[ri]

    losses, accuracy = [], []
    for (xrow, yrow) in zip(Xb, yb):
        #inputs = list(map(Value, xrow))  #Value(xrow[i])
        #outputs = model(inputs)  #forward pass
        #probs, log_softmax = softmax(outputs)
        probs, log_softmax = softmax(model(Tensor(xrow)))
        losses.append(-sum([log_softmax[index] * int(index == yrow) for index in range(len(log_softmax))])) 
        # ^ cross entropy loss (can't just take log_softmax[yrow] or else you lose track of gradients and backward() doesn't work)
        accuracy.append(yrow == np.argmax(probs))
    total_loss = sum(losses) / len(losses)
    return total_loss, sum(accuracy) / len(accuracy)

def kaggle_training(epochs = 10, batch_size = None):
    X = np.empty((42000, 28*28), dtype = int)
    y = np.empty(42000, dtype = int)
    with open('digit-recognizer/train.csv', newline='\n') as csvfile:
        digitreader = csv.reader(csvfile, delimiter=',')
        for row in digitreader:
            if digitreader.line_num != 1: #line_num starts at 1, not 0
                y[digitreader.line_num-2] = int(row[0])
                X[digitreader.line_num-2] = [int(char) for char in row[1:]]
    
    X = (X-np.average(X)) / np.std(X)  #data normalization

    # initialize a model 
    #model = MLP(28*28, [25, 15, 10])
    #print(model, "\n", "NUMBER OF PARAMTERS:", len(model.parameters()), "\n")
    class LinearNet:
        def __init__(self):
            self.params = {
            'w1' : Tensor(2 * np.random.random_sample((28*28, 25)) - 1),
            'b1' : Tensor(np.zeros((1, 25))),
            'w2' : Tensor(2 * np.random.random_sample((25, 15)) - 1),
            'b2' : Tensor(np.zeros((1, 15))),
            'w3' : Tensor(2 * np.random.random_sample((15, 10)) - 1),
            'b3' : Tensor(np.zeros((1, 10)))
            }

        def parameters(self):
            return self.params.values()

        def __call__(self, x:Tensor) -> Tensor:
            return ((x@self.params['w1']+self.params['b1']).relu()@self.params['w2']+self.params['b2']).relu()@self.params['w3']+self.params['b3']
    
    model = LinearNet()

    #index_list = [batch_size*3]
    #index_list = [64, 128, 128] # TODO: CHANGE THIS
    index_list = [-1]
    for index in index_list:
        print(index)
        # optimization
        for k in range(epochs):
            
            # forward
            total_loss, acc = loss(X[:index], y[:index], model, batch_size = batch_size)

            # backward
            #model.zero_grad()
            total_loss.backprop()
            
            # update (sgd)
            #learning_rate = 0.1 - 0.0999*k/epochs
            learning_rate = 0.01
            for p in model.parameters():
                p.data -= learning_rate * p.grad
            
            if k % 1 == 0:
                print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")

        #if index != X.shape[0]:
        #    index_list.append(index*10 if index*10 < X.shape[0] else X.shape[0]) #TODO: change 10
        #    print()

    print('TRAINING COMPLETE')
    plot_kaggle_data(X, y, model, predict = True)
    #print('BEGINNING TEST SET INFERENCE')
    #write_kaggle_submission(model)
    #print('TEST SET INFERENCE COMPLETE')

#############################################################################################

#my_test()
#test_sanity_check()
#test_more_ops()

kaggle_training(epochs = 12, batch_size = 32)