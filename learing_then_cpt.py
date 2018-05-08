# -*- coding: utf-8 -*-

import torch
import numpy
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

workdir = "./"
L, R = -1.2, -0.8
x_L, x_R = 34., 46.
wa_L, wa_R = -0.5, 0.5
N_test = 3
# produce dot num
dot_num = 500

obs_set_numpy = numpy.loadtxt(workdir + "sn.dat")
#nmlz_obs_set_numpy = (obs_set_numpy - x_L) / (x_R - x_L)

#
lf1_weight = numpy.loadtxt("lf1_weight")
lf2_weight = numpy.loadtxt("lf2_weight")
lf3_weight = numpy.loadtxt("lf3_weight")
lf1_bias = numpy.loadtxt("lf1_bias")
lf2_bias = numpy.loadtxt("lf2_bias")
lf3_bias = numpy.loadtxt("lf3_bias")

D_in = obs_set_numpy[:,1].size
H = lf1_bias.size
HH = lf2_bias.size
D_out = lf3_bias.size
print "D_in = {0}, H = {1}, HH = {2}, D_out = {3}".format(D_in, H, HH, D_out)

lf1 = torch.nn.Linear(D_in, H)
lf2 = torch.nn.Linear(H, HH)
lf3 = torch.nn.Linear(HH, D_out)

lf1.weight.data = torch.FloatTensor(lf1_weight)
lf2.weight.data = torch.FloatTensor(lf2_weight)
lf3.weight.data = torch.FloatTensor(lf3_weight)
lf1.bias.bias = torch.FloatTensor(lf1_bias)
lf2.bias.bias = torch.FloatTensor(lf2_bias)
lf3.bias.bias = torch.FloatTensor(lf3_bias)

model = torch.nn.Sequential(
    lf1,
    torch.nn.ReLU(),
    lf2,
    torch.nn.ReLU(),
    lf3,
)

w0w1 = numpy.zeros([dot_num, 2])
plt_title = "Observation Error for a network\n"
plt_title +="dot_num, D_in, H1, H2, D_out = {0}, {1}, {2}, {3}, {4}".format(dot_num, D_in, H, HH, D_out)
plt.title(plt_title)
plt.xlabel("w0")
plt.ylabel("w1")
mu = obs_set_numpy[:,1]
for i in range(dot_num):
	x = numpy.zeros(D_in)
	for j in range(obs_set_numpy[:,2].size):
		x[j] = mu[j] + numpy.random.normal(scale=obs_set_numpy[j,2])
	# end for j in obs_set_numpy[:,2]
	# normalization
	x = (x - x_L) / (x_R - x_L)
	y = model(Variable(torch.FloatTensor(x), requires_grad=False))
	w0w1[i] = y.data
# end for i in range

# denormalization
w0w1[:,0] = w0w1[:,0] * (R-L)+L
w0w1[:,1] = w0w1[:,1] * (wa_R-wa_L)+wa_L
plt.plot(w0w1[:,0], w0w1[:,1], ".")
#plt.plot([-1, -1],[w0w1[:,1].min(), w0w1[:,1].max()],"--",color="gray")
#plt.plot([w0w1[:,0].min(), w0w1[:,0].max()],[0, 0],"--",color="gray")

name = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))+"sigma"
plt.savefig(name+".png", format="png")
numpy.savetxt(name, w0w1)
plt.show()











