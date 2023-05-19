import numpy as np
from scipy.stats import norm, entropy

def SiLU(x):
	return x/(1+np.exp(-x))

def get_out_expect(model,i = 0,act = 'SiLU'): # i +1 +2 就是bn层id
	out_expect = []
	state_dict = model.state_dict()
	names = list(state_dict.keys())
	norm_weight = state_dict[names[i+1]]  # bn层id视情况调整
	norm_bias   = state_dict[names[i+2]]
	num = len(norm_bias)
	for i in range(num):
		if norm_weight[i].item() < 0.: 
			norm_weight[i] = 1e-7
		_norm = norm(norm_bias[i].item(),norm_weight[i].item())
		out_expect.append(_norm.expect(SiLU))
	return out_expect