# -*- coding: utf-8 -*-

import os
import torch
import numpy
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time

workdir = "./"
vol, vol_wa = 10, 10
L, R = -1.2, -0.8
x_L, x_R = 34., 46.
wa_L, wa_R = -0.5, 0.5
# N is batch size;
N = vol * vol_wa

dot_num = 500
w0w1 = numpy.zeros([dot_num,2])

def readsn():
	res = numpy.loadtxt(workdir + "sn.dat")[:,1]
	nmlz_res = (res - x_L) / (x_R - x_L)
	return Variable(torch.FloatTensor( nmlz_res ), requires_grad=False)

def produce(w, wa):
    numpy.savetxt(workdir + "frompython.dat", [w, wa])
    os.system(workdir + "cosmomc "+ "bkground.ini")
    Mockdata = numpy.loadtxt(workdir + "Mockdata.dat")
    return Mockdata[:,1]
for dot_iter in range(dot_num):
	### BEGINNING ###
	temp_x = produce(L, wa_L)
	temp_y = [L, wa_L]

	temp = numpy.random.rand()*(R-L)+L
	temp_wa = numpy.random.rand()*(wa_R-wa_L)+wa_L
	

	temp_count = 0
	for i in range(0, vol):
		w = L+(R-L)/(vol-1)*i
		for j in range(0, vol_wa):
		    wa = wa_L+(wa_R-wa_L)/(vol-1)*j
		    temp_x = numpy.vstack((temp_x, produce(w, wa)))
		    temp_y = numpy.vstack((temp_y, [w, wa])); print "generate {0}/{1}, dot_iter = {2}/{3}".format(temp_count, N, dot_iter, dot_num);temp_count += 1
	temp_x = numpy.delete(temp_x, 0, 0)
	temp_y = numpy.delete(temp_y, 0, 0)
	
	#normalize x and y
	nmlz_temp_x = (temp_x - x_L) / (x_R - x_L)
	nmlz_temp_y = temp_y.copy()
	nmlz_temp_y[:,0] = (nmlz_temp_y[:,0] - L)/ (R-L)
	nmlz_temp_y[:,1] = (nmlz_temp_y[:,1] - wa_L)/ (wa_R-wa_L)

	# D_in is input dimension;
	# H is hidden dimension; D_out is output dimension.
	D_in = temp_x[0].size
	D_out = 2
	H = 300
	HH= 50

	# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
	x = Variable(torch.FloatTensor(nmlz_temp_x), requires_grad=False)
	y = Variable(torch.FloatTensor(nmlz_temp_y), requires_grad=False)

	# Use the nn package to define our model as a sequence of layers. nn.Sequential
	# is a Module which contains other Modules, and applies them in sequence to
	# produce its output. Each Linear Module computes output from input using a
	# linear function, and holds internal Variables for its weight and bias.
	model = torch.nn.Sequential(
		torch.nn.Linear(D_in, H),
		torch.nn.ReLU(),
		torch.nn.Linear(H, HH),
		torch.nn.ReLU(),
		torch.nn.Linear(HH, D_out),
	)

	# The nn package also contains definitions of popular loss functions; in this
	# case we will use Mean Squared Error (MSE) as our loss function.
	loss_fn = torch.nn.MSELoss(size_average=False)

	learning_rate = 1e-4
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	for t in range(80):
		# Forward pass: compute predicted y by passing x to the model. Module objects
		# override the __call__ operator so you can call them like functions. When
		# doing so you pass a Variable of input data to the Module and it produces
		# a Variable of output data.
		y_pred = model(x)

		# Compute and print loss. We pass Variables containing the predicted and true
		# values of y, and the loss function returns a Variable containing the
		# loss.
		#import pdb; pdb.set_trace()
		loss = loss_fn(y_pred, y)

		# Zero the gradients before running the backward pass.
		optimizer.zero_grad()

		# Backward pass: compute gradient of the loss with respect to all the learnable
		# parameters of the model. Internally, the parameters of each Module are stored
		# in Variables with requires_grad=True, so this call will compute gradients for
		# all learnable parameters in the model.
		loss.backward()
		
		# Calling the step function on an Optimizer makes an update to its
		# parameters
		optimizer.step()
	w0w1[dot_iter] = model(readsn()).data
### ENDING ###

name = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + "net_error"


# denormalization
w0w1[:,0] = w0w1[:,0] * (R-L)+L
w0w1[:,1] = w0w1[:,1] * (wa_R-wa_L)+wa_L
numpy.savetxt(name, w0w1)
plt.plot(w0w1[:,0], w0w1[:,1], ".")
plt.plot([-1, -1],[w0w1[:,1].min(), w0w1[:,1].max()],"--",color="gray")
plt.plot([w0w1[:,0].min(), w0w1[:,0].max()],[0, 0],"--",color="gray")
plt.xlabel("w0")
plt.ylabel("w1")


plt_title = "Different (Training Set & Initial Weights)\n"
plt_title +="dot_num, N, D_in, H1, H2, D_out = {0}, {1}, {2}, {3}, {4}".format(dot_num, N, D_in, H, HH, D_out)
plt.title(plt_title)
plt.savefig(name+".png", format="png")
plt.show()
