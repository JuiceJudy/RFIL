import argparse
import os
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils.utils import * # test, save_checkpoint, get_logger
from model.pointconv import PointConvDensityClsSsg as PointConvClsSsg
from model.pointconv_prune import PointConvDensityClsSsg as PointConvClsSsgPrune
import provider
import numpy as np 

from prune_methods import *
from utils.utils import *

def parse_args():
	'''PARAMETERS'''
	parser = argparse.ArgumentParser('PointConv')
	parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
	parser.add_argument('--batchsize', type=int, default=32, help='batch size in training')
	parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
	parser.add_argument('--num_workers', type=int, default=16, help='Worker Number [default: 16]')
	parser.add_argument('--pretrain', type=str, default='experiment/pretrained/pointconv_modelnet40-0.921799.pth',help='whether use pretrain model')
	parser.add_argument('--compress_rate', type=str, default='[0.2]*9', help='compress rate of each conv')
	parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information [default: False]')
	parser.add_argument('--data_dir', default='G:\\modelnet40_normal_resampled', type=str, metavar='DIR', help='path to dataset')
	return parser.parse_args()

def load_pruned_weights(model, old_state_dict, ranks=[]):
	logger.info('load pruned weights')
	# 需要修改的各层下标 ;(
	sa1_conv	  = [0,2,4]
	sa1_bn		  = [6,11,16]
	sa1_linear	  = [42]
	sa1_linear_bn = [42+2]

	sa2_conv	  = [x+70 for x in sa1_conv]
	sa2_bn		  = [x+70 for x in sa1_bn]
	sa2_linear	  = [sa1_linear[0]+70]
	sa2_linear_bn = [sa1_linear_bn[0] + 70]

	sa3_conv	  = [x+70 for x in sa2_conv]
	sa3_bn		  = [x+70 for x in sa2_bn]
	sa3_linear	  = [sa2_linear[0]+70]
	sa3_linear_bn = [sa2_linear_bn[0] + 70]

	fc1		   = [210]

	sa_conv	  = sa1_conv + sa1_linear + sa2_conv + sa2_linear + sa3_conv + sa3_linear
	sa_bn	  = sa1_bn + sa1_linear_bn + sa2_bn + sa2_linear_bn + sa3_bn + sa3_linear_bn

	if len(ranks) < 9: ranks = ranks + ['no_pruned'] * (9 - len(ranks))
	ranks = ranks[0:3] + ranks[2:3] + ranks[3:6] + ranks[5:6] + ranks[6:9] + ranks[8:9]

	# print([len(x) for x in ranks])
	# print(len(sa_conv),sa_conv)

	all_changed = []
	all_changed = sa_conv + sa_bn + fc1

	lastrank   = None
	layernames = all_layers(model)  # list(state_dict.keys())

	state_dict = model.state_dict()

	for _id,convid in enumerate(sa_conv):  # _id  下标  convid 卷积层下标
		#conv & linear layer
		if _id % 4 == 3: # linear layer
			weight_name = layernames[convid]
			for _i,i in enumerate(lastrank):
				tmp_rank = lastrank
				tmp_rank.sort()
				for _ii,ii in enumerate(tmp_rank):
					state_dict[weight_name][_i][_ii*16:(_ii+1)*16] = old_state_dict[weight_name][i][ii*16:(ii+1)*16]
		else : # conv layer
			conv_weight_name = layernames[convid]
			rank = ranks[_id]
			if rank == 'no_pruned': rank = list(range(len(state_dict[conv_weight_name])))
			#卷积层
			if lastrank is None:
				for _i,i in enumerate(rank):
					state_dict[conv_weight_name][_i] = old_state_dict[conv_weight_name][i]
			else :
				for _i,i in enumerate(rank):
					for _j,j in enumerate(lastrank):
						state_dict[conv_weight_name][_i][_j] = old_state_dict[conv_weight_name][i][j]
					if _id % 4 == 0:
						state_dict[conv_weight_name][_i][-3:] = old_state_dict[conv_weight_name][i][-3:]
			lastrank = rank
		
		# 偏置层
		conv_bias_name = layernames[convid+1]
		all_changed.append(convid+1)
		for _i,i in enumerate(lastrank):
			state_dict[conv_bias_name][_i] = old_state_dict[conv_bias_name][i]
		# bn层
		for bnid in range(sa_bn[_id],sa_bn[_id]+5):
			all_changed.append(bnid)
			_name = layernames[bnid]
			if bnid == sa_bn[_id]+5-1:
				state_dict[_name] = old_state_dict[_name]
			else:
				for _i,i in enumerate(lastrank):
					state_dict[_name][_i] = old_state_dict[_name][i]
	# fc1
	fc1_weight_name = layernames[fc1[0]]
	fc1_rank = list(range(len(state_dict[fc1_weight_name])))
	for _i,i in enumerate(fc1_rank):
		for _j,j in enumerate(lastrank):
			state_dict[fc1_weight_name][_i][_j] = old_state_dict[fc1_weight_name][i][j]
	#other layers
	for k,name in enumerate(old_state_dict):
		if k not in all_changed:
			state_dict[name] = old_state_dict[name]

	model.load_state_dict(state_dict, strict=False)
	return model

def loadmodel():

	# TRAIN_DATASET = ModelNetDataLoader(root=args.data_dir, npoint=args.num_point, split='train', normal_channel=args.normal)
	# TEST_DATASET = ModelNetDataLoader(root=args.data_dir, npoint=args.num_point, split='test', normal_channel=args.normal)
	# trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)
	# testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

	num_class = 40
	classifier = PointConvClsSsg(num_class).cuda()
	checkpoint = torch.load(args.pretrain)
	# print(checkpoint.keys())
	classifier.load_state_dict(checkpoint['model_state_dict'])

	# acc = test(classifier, testDataLoader)

	# logger.info('Load Model: '+args.pretrain)
	# logger.info('Top-1 Accuracy: '+str(acc))

	# old_state_dict = classifier.state_dict()
	# model = PointConvClsSsg(num_class).cuda()
	# model = load_pruned_weights(model, old_state_dict)

	# acc = test(model, testDataLoader)
	# logger.info('Top-1 Accuracy: '+str(acc))

	return classifier

def prune_pointconv():

	TEST_DATASET = ModelNetDataLoader(root=args.data_dir, npoint=args.num_point, split='test', normal_channel=args.normal)
	testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

	rate = format_compress_rate(args.compress_rate)
	rate = [0.8]*10
	model = loadmodel()
	ranks = []
	sa1_conv  = [0,2,4]
	sa1_bn	= [6,11,16]
	sa2_conv  = [x+70 for x in sa1_conv]
	sa2_bn	= [x+70 for x in sa1_bn]
	sa3_conv  = [x+70 for x in sa2_conv]
	sa3_bn	= [x+70 for x in sa2_bn]
	conv	  = sa1_conv + sa2_conv + sa3_conv
	bn		= sa1_bn + sa2_bn + sa3_bn
	calc_n	= [0,0,-1] * 3  # whether to calc next conv
	
	for i,conv_id in enumerate(conv):
		bn_id = bn[i]
		if rate[i] > 0.:
			if calc_n[i] == 0:
				rank = RFIL(model,bn_id,rate[i],n = conv[i+1])
				# rank = RFIL(model,bn_id,rate[i],n = -1)
			else :
				rank = RFIL(model,bn_id,rate[i],n = -1)
		else :
			rank = 'no_pruned'
		
		ranks.append(rank)

	# print(ranks)
	print([len(x) for x in ranks])

	# pruned_model = PointConvClsSsgPrune(num_classes=40,pruning_rate=rate).cuda()
	pruned_model = PointConvClsSsgPrune(num_classes=40,pruning_rate=rate).cuda()

	# print(pruned_model)

	old_state_dict = model.state_dict()
	pruned_model = load_pruned_weights(pruned_model, old_state_dict, ranks = ranks)

	logger.info(pruned_model)

	flops, params = model_info(pruned_model,npoints=1024)
	acc = test(pruned_model, testDataLoader)
	logger.info('Top-1 Accuracy: '+str(acc))
	logger.info('Model Summary: {} FLOPs, {} parameters'.format(flops,params))


	# save model
	ckpt = {'pruning_rate': rate,
			'test_accuracy': acc,
			'FLOPs': flops,
			'parameters': params, 
			'model_state_dict': pruned_model.state_dict()
			}
	save_dir = 'prune_runs/prune_test%.pt'
	torch.save(ckpt, save_dir)
	logger.info('Save ' + save_dir)

	return pruned_model

def test_load_pruned_model():

	num_class = 40
	model_dir  = 'prune_runs/prune_50%.pt'
	checkpoint = torch.load(model_dir)
	TEST_DATASET = ModelNetDataLoader(root=args.data_dir, npoint=args.num_point, split='test', normal_channel=args.normal)
	testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

	rate = checkpoint['pruning_rate']
	state_dict = checkpoint['model_state_dict']

	model = PointConvClsSsgPrune(num_classes=num_class,pruning_rate=rate).cuda()

	model.load_state_dict(state_dict)
	acc = test(model, testDataLoader)

	print(acc)

if __name__ == '__main__':
	args = parse_args()
	logger = get_logger('logs/log_prune')

	# loadmodel() 95 layers, 
	# original    Model Summary: 1165255552.0 FLOPs, 19569657 parameters 
	# 0.1 * 9     Model Summary: 944583490.0 FLOPs, 15903820 parameters   0.18937653772277413
	# 0.2 * 9     Model Summary: 751075318.0 FLOPs, 12649270 parameters   0.35544154523796684
	# 0.3 * 9     Model Summary: 578487960.0 FLOPs, 9753210 parameters    0.5035527108134302
	# 0.4 * 9     Model Summary: 430225164.0 FLOPs, 7252964 parameters    0.6307890030975798
	# 0.5 * 9     Model Summary: 310435712.0 FLOPs, 5135801 parameters    0.7335900168274847
	# 0.6 * 9     Model Summary: 204074818.0 FLOPs, 3371564 parameters    0.8248668992396271
	# 0.7 * 9     Model Summary: 122649590.0 FLOPs, 2001206 parameters    0.8947444706103405
	# 0.8 * 9     Model Summary: 63177368.0 FLOPs, 1001434 parameters     0.9457823926334744
	prune_pointconv()

	# test_load_pruned_model()
