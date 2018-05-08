# -*- coding: utf-8 -*-

import os
import torch
import numpy
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time

workdir = "./"
vol, vol_wa = 50, 50
L, R = -1.2, -0.8
x_L, x_R = 34., 46.
wa_L, wa_R = -0.5, 0.5
# N is batch size;
N = vol * vol_wa
N_test = 20

def readsn():
	res = numpy.loadtxt(workdir + "sn.dat")[:,1]
	nmlz_res = (res - x_L) / (x_R - x_L)
	return Variable(torch.FloatTensor( nmlz_res ), requires_grad=False)

def produce(w, wa):
    numpy.savetxt(workdir + "frompython.dat", [w, wa])
    os.system(workdir + "cosmomc "+ "bkground.ini")
    Mockdata = numpy.loadtxt(workdir + "Mockdata.dat")
    return Mockdata[:,1]
temp_x = produce(L, wa_L)
temp_y = [L, wa_L]

temp = numpy.random.rand()*(R-L)+L
temp_wa = numpy.random.rand()*(wa_R-wa_L)+wa_L
temp_x_test = produce(temp, temp_wa)
temp_y_test = [temp, temp_wa]

temp_count = 0
for i in range(0, vol):
    w = L+(R-L)/(vol-1)*i
    for j in range(0, vol_wa):
        wa = wa_L+(wa_R-wa_L)/(vol-1)*j
        temp_x = numpy.vstack((temp_x, produce(w, wa)))
        temp_y = numpy.vstack((temp_y, [w, wa]))
	print "generate {0}/{1}".format(temp_count, N)
	temp_count += 1
temp_x = numpy.delete(temp_x, 0, 0)
temp_y = numpy.delete(temp_y, 0, 0)
for i in numpy.random.rand(N_test-1):
    w = L+(R-L)*i
    j = numpy.random.rand()
    wa = wa_L+(wa_R-wa_L)*j
    temp_x_test = numpy.vstack((temp_x_test, produce(w, wa)))
    temp_y_test = numpy.vstack((temp_y_test, [w, wa]))


#normalize x and y
nmlz_temp_x = (temp_x - x_L) / (x_R - x_L)
nmlz_temp_y = temp_y.copy()
nmlz_temp_y[:,0] = (nmlz_temp_y[:,0] - L)/ (R-L)
nmlz_temp_y[:,1] = (nmlz_temp_y[:,1] - wa_L)/ (wa_R-wa_L)
nmlz_temp_x_test = (temp_x_test - x_L) / (x_R - x_L)
nmlz_temp_y_test = temp_y_test.copy()
nmlz_temp_y_test[:,0] = (nmlz_temp_y_test[:,0] - L)/ (R-L)
nmlz_temp_y_test[:,1] = (nmlz_temp_y_test[:,1] - wa_L)/ (wa_R-wa_L)

# D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in = temp_x[0].size
D_out = 2
H = 300
HH= 50

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = Variable(torch.FloatTensor(nmlz_temp_x), requires_grad=False)
y = Variable(torch.FloatTensor(nmlz_temp_y), requires_grad=False)
x_test = Variable(torch.FloatTensor(nmlz_temp_x_test), requires_grad=False)
y_test = Variable(torch.FloatTensor(nmlz_temp_y_test), requires_grad=False)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.
lf1 = torch.nn.Linear(D_in, H)
lf2 = torch.nn.Linear(H, HH)
lf3 = torch.nn.Linear(HH, D_out)
model = torch.nn.Sequential(
    lf1,
    torch.nn.ReLU(),
    lf2,
    torch.nn.ReLU(),
    lf3,
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(400):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Variable of input data to the Module and it produces
    # a Variable of output data.
    y_pred = model(x)
    y_pred_test = model(x_test)

    # Compute and print loss. We pass Variables containing the predicted and true
    # values of y, and the loss function returns a Variable containing the
    # loss.
	#import pdb; pdb.set_trace()
    loss = loss_fn(y_pred, y)
    loss_test = loss_fn(y_pred_test, y_test)
    print(t, loss.data[0]/N, loss_test[0]/N_test)

    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Variables with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()
    plt.plot(t, loss.data[0]/N, "r.", label="train")
    plt.plot(t, loss_test.data[0]/N_test, "b.", label="test")

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
numpy.savetxt("lf1_weight",lf1.weight.data)
numpy.savetxt("lf2_weight",lf2.weight.data)
numpy.savetxt("lf3_weight",lf3.weight.data)
numpy.savetxt("lf1_bias",lf1.bias.data)
numpy.savetxt("lf2_bias",lf2.bias.data)
numpy.savetxt("lf3_bias",lf3.bias.data)

w0w1 = model(readsn()).data
# denormalization
w0w1[0] = w0w1[0] * (R-L)+L
w0w1[1] = w0w1[1] * (wa_R-wa_L)+wa_L
print "w0w1 = {0},{1}".format(w0w1[0], w0w1[1])
plt_title = "simple_two_hidden_layer: iteration v.s. loss\N\n"
plt_title +="N, D_in, H1, H2, D_out = {0}, {1}, {2}, {3}, {4}".format(N, D_in, H, HH, D_out)
plt.title(plt_title)
plt.savefig(time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))+".png", format="png")
plt.show()
