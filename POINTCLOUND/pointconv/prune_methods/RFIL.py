import numpy as np
import math
from scipy.stats import norm, entropy


from .common import *

def get_out_expect(model,i = 0): # i  bn-layer weight
	out_expect = []
	state_dict = model.state_dict()
	names = list(state_dict.keys())
	norm_weight = state_dict[names[i]]  # bn层id视情况调整
	norm_bias   = state_dict[names[i+1]]
	num = len(norm_bias)
	for i in range(num):
		if norm_weight[i].item() < 0.: 
			norm_weight[i] = 1e-7
		_norm = norm(norm_bias[i].item(),norm_weight[i].item())
		out_expect.append(_norm.expect(lb=0))
	return out_expect

def get_effect_2next(model,lid,nid,nfid,out_expect):  # 模型 本层id 下层id 下层特征id 本层输出期望
	state_dict = model.state_dict()
	names = list(state_dict.keys())
	state_dict_n = state_dict[names[nid]]
	num  = len(state_dict[names[lid]])  # i层通道个数
	eps  = 1e-7
	efts = []   # effect
	_sum = 0.
	_abs_sum = 0.
	for i in range(num):
		eft = state_dict_n[nfid][i].sum().item() * out_expect[i]
		_sum += eft
		_abs_sum += abs(eft)
		efts.append(eft)
	efts = [abs(x/(_abs_sum+eps)) for x in efts]
	return efts


def RFIL(model,i = 0,rate=0.,n = -1):   # i  bn-layer weight # n next conv layer
	rank = []
	out_expect = get_out_expect(model,i=i)  # output this layer
	num = len(out_expect)
	state_dict = model.state_dict()
	if n != -1:
		num_n = len(state_dict[list(state_dict.keys())[n]])  # width of layer n
		scores = [0.] * num
		for nfid in range(num_n):
			_scores = get_effect_2next(model,i,n,nfid,out_expect)
			scores = [x+y for x,y in zip(scores,_scores)]
		rank = list(np.argsort(scores))
	else :
		out_expect = [abs(x) for x in out_expect]
		rank = list(np.argsort(out_expect))
	_num = int(num*(1.-rate))
	rank = rank[num-_num:] 
	return rank