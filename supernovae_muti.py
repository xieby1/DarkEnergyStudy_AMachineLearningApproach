# -*- coding: utf-8 -*-
# devide training set into small groups

import os
import torch
import numpy
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
from scipy import stats

workdir = "./"
vol, vol_wa = 10, 10
L, R = -1.2, -0.8
x_L, x_R = 34., 46.
wa_L, wa_R = -0.5, 0.5
# N is batch size;
N = vol * vol_wa

def readsn():
	return (numpy.loadtxt(workdir + "sn.dat")[:,1] - x_L) / (x_R - x_L)

def produce(w, wa):
    numpy.savetxt(workdir + "frompython.dat", [w, wa])
    os.system(workdir + "cosmomc "+ "bkground.ini")
    Mockdata = numpy.loadtxt(workdir + "Mockdata.dat")
    return Mockdata[:,1]
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
        temp_y = numpy.vstack((temp_y, [w, wa]))
	print "generate {0}/{1}".format(temp_count, N)
	temp_count += 1
temp_x = numpy.delete(temp_x, 0, 0)
temp_y = numpy.delete(temp_y, 0, 0)


#normalize x and y
nmlz_temp_x = (temp_x - x_L) / (x_R - x_L)
nmlz_temp_y = temp_y.copy()
nmlz_temp_y[:,0] = (nmlz_temp_y[:,0] - L)/ (R-L)
nmlz_temp_y[:,1] = (nmlz_temp_y[:,1] - wa_L)/ (wa_R-wa_L)

#group info, totally 580
grp_sz = 1 #size
grp_nm = temp_x[0].size / grp_sz #number
trn_grp_nm = 400 #training group number each time

# D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in = grp_sz * trn_grp_nm
D_out = 2
H = D_in / 2
HH= H / 2

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = nmlz_temp_x # Variable(torch.FloatTensor(nmlz_temp_x), requires_grad=False)
y = Variable(torch.FloatTensor(nmlz_temp_y), requires_grad=False)

w0w1 = numpy.zeros([grp_nm, 2])
sndata_hst = numpy.hstack([readsn(), readsn()])
# not to break boundary
x_hst = numpy.hstack([x,x])
for grp_iter in range(grp_nm):
	print "grp_iter: {0}/{1}".format(grp_iter, grp_nm)
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
	
	grp_sndata = Variable( torch.FloatTensor( sndata_hst[grp_iter*grp_sz : (grp_iter+trn_grp_nm)*grp_sz] ) , requires_grad=False)
	grp_x = Variable( torch.FloatTensor( x_hst[:, grp_iter*grp_sz : (grp_iter+trn_grp_nm)*grp_sz] ) , requires_grad=False)
	learning_rate = 1e-4
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	for t in range(400):
		# Forward pass: compute predicted y by passing x to the model. Module objects
		# override the __call__ operator so you can call them like functions. When
		# doing so you pass a Variable of input data to the Module and it produces
		# a Variable of output data.
		y_pred = model(grp_x)

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
		#plt.plot(t, loss.data[0]/N, "r.", label="train")

		# Calling the step function on an Optimizer makes an update to its
		# parameters
		optimizer.step()
	#end for t in range
	w0w1[grp_iter] = model(grp_sndata).data
#end for grp_iter in range(grp_nm)
# denormalization
w0w1[:,0] = w0w1[:,0] * (R-L)+L
w0w1[:,1] = w0w1[:,1] * (wa_R-wa_L)+wa_L
plt.plot(w0w1[:,0], w0w1[:,1], ".")
plt.plot([-1, -1],[w0w1[:,1].min(), w0w1[:,1].max()],"--",color="gray")
plt.plot([w0w1[:,0].min(), w0w1[:,0].max()],[0, 0],"--",color="gray")
plt.xlabel("w0")
plt.ylabel("w1")
plt_title = "Overlapped Division: distribution\n"
plt_title +="grp_sz, grp_nm, N = {0}, {1}, {2},".format(grp_sz, grp_nm, N) + "D_in, H1, H2, D_out = {0}, {1}, {2}, {3}".format(D_in, H, HH, D_out)
plt.title(plt_title)
name = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))+"Distri"
plt.savefig(name+".png", format="png")
numpy.savetxt(name, w0w1)
plt.show()
