import numpy as np
import math
from scipy.stats import norm, entropy

from .common import *

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


def RFIL(model,i = 0,rate=0.,n = -1):
	rank = []
	out_expect = get_out_expect(model,i=i)
	num = len(out_expect)
	if n != -1:
		num_n = len(get_out_expect(model,i=n))
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