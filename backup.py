# -*- coding: utf-8 -*-

import os
import torch
import numpy
from torch.autograd import Variable
import matplotlib.pyplot as plt

workdir = "./"

L, R, step = -1.2, -0.8, 0.005
x_L, x_R = 34., 46.
N_test = 10

def produce(w):
    numpy.savetxt(workdir + "frompython.dat", [w])
    os.system(workdir + "cosmomc "+ "bkground.ini")
    Mockdata = numpy.loadtxt(workdir + "Mockdata.dat")
    return Mockdata.transpose()[1]
temp_x = produce(L)
temp_y = [L]

temp = numpy.random.rand()*(R-L)+L
temp_x_test = produce(temp)
temp_y_test = [temp]

for i in numpy.arange(L+step, R, step):
    temp_x = numpy.vstack((temp_x, produce(i)))
    temp_y = numpy.vstack((temp_y, [i]))
for i in numpy.random.rand(N_test-1)*(R-L)+L:
    temp_x_test = numpy.vstack((temp_x_test, produce(i)))
    temp_y_test = numpy.vstack((temp_y_test, [i]))


#normalize x and y
nmlz_temp_x = (temp_x - x_L) / (x_R - x_L)
nmlz_temp_y = (temp_y - L) / (R - L)
nmlz_temp_x_test = (temp_x_test - x_L) / (x_R - x_L)
nmlz_temp_y_test = (temp_y_test - L) / (R - L)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N = temp_y.size
D_in = temp_x[0].size
D_out = 1
H = 70
HH= 8

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = Variable(torch.FloatTensor(nmlz_temp_x), requires_grad=False)
y = Variable(torch.FloatTensor(nmlz_temp_y), requires_grad=False)
x_test = Variable(torch.FloatTensor(nmlz_temp_x_test), requires_grad=False)
y_test = Variable(torch.FloatTensor(nmlz_temp_y_test), requires_grad=False)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, HH),
    torch.nn.ReLU(),
    torch.nn.Linear(HH, D_out)
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-5
for t in range(2000):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Variable of input data to the Module and it produces
    # a Variable of output data.
    y_pred = model(x)
    y_pred_test = model(x_test)

    # Compute and print loss. We pass Variables containing the predicted and true
    # values of y, and the loss function returns a Variable containing the
    # loss.
    loss = loss_fn(y_pred, y)
    loss_test = loss_fn(y_pred_test, y_test)
    print(t, loss.data[0])

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Variables with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()
    plt.plot(t, loss.data[0]/N, "r.")
    plt.plot(t, loss_test.data[0]/N_test, "b.")

    # Update the weights using gradient descent. Each parameter is a Variable, so
    # we can access its data and gradients like we did before.
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data
plt_title = "simple_one_hidden_layer_polynomial: iteration v.s. loss\N\n"
plt_title +="N, D_in, H, D_out = {0}, {1}, {2}, {3}".format(N, D_in, H, D_out)
plt.title(plt_title)
plt.show()