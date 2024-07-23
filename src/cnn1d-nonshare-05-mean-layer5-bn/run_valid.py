import copy
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
from dataset import *
from model import *

import numpy as np
import random
from timeit import default_timer as timer
import torch

from my_lib.runner import *
from my_lib.file import *
from my_lib.net.rate import get_learning_rate
from my_lib.draw import *

from sklearn.metrics import roc_curve
import sklearn.metrics
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score, precision_recall_curve


### start validation here! ################################################

def run_valid(cfg):
	cfg.name = 'valid-' + cfg.resume_from.checkpoint.split('/')[-1][:-4]

	# --- setup ---
	seed_everything(cfg.seed)
	os.makedirs(cfg.fold_dir, exist_ok=True)
	for f in ['checkpoint', 'train', 'valid', 'etc']:
		os.makedirs(cfg.fold_dir + '/' + f, exist_ok=True)

	log = Logger()
	log.open(cfg.fold_dir + '/log.valid.txt', mode='a')
	log.write(f'\n--- [START {log.timestamp()}] {"-" * 64}')
	log.write(f'__file__ = {__file__}\n')
	log.write(f'cfg:\n{format_dict(cfg)}')
	log.write(f'')

	# --- net ---
	net = Net(cfg)
	net.cuda()
	f = torch.load(cfg.resume_from.checkpoint, map_location=lambda storage, loc: storage)
	state_dict = f['state_dict']
	print(net.load_state_dict(state_dict, strict=False))  # True


	net.eval()
	net.output_type = ['loss', 'infer']

	# --- dataset ---

	train_idx, valid_idx, nonshare_idx, unused_idx = load_kaggle_idx4(fold=cfg.fold)
	token_id, bind = load_kaggle_data()
	log.write('dataset:kaggle')
	log.write(data_to_text(token_id, bind))
	log.write(index_to_text3(train_idx, valid_idx, nonshare_idx))
	log.write(bind_to_text(bind[train_idx]))
	log.write(bind_to_text(bind[valid_idx]))
	log.write(bind_to_text(bind[nonshare_idx]))
	log.write('')

 	#---------------------
	result = dotdict(
		prob =[],
		truth=[],
	)
	start_timer = timer()

	num_valid = 0
	for t, index in enumerate(np.arange(0,len(valid_idx),cfg.valid_batch_size)):
		index = valid_idx[index:index+cfg.valid_batch_size]

		B = len(index)
		batch = dotdict(
			smiles_token_id   = torch.from_numpy(token_id[index]).byte().cuda(),
			bind = torch.from_numpy(bind[index]).float().cuda(),
		)
		with torch.cuda.amp.autocast(enabled=cfg.is_amp):
			with torch.no_grad():#output['bind']
				output = net(batch)   # data_parallel(net,batch)#
		result.truth.append(batch['bind'].data.cpu().numpy())
		result.prob.append(output['bind'].data.cpu().numpy())
		num_valid +=B

		print(f'\r validation: {num_valid}/{len(valid_idx)}', time_to_str(timer() - start_timer, 'min'),
			  end='', flush=True)

	#-----
	result1 = dotdict(
		prob=[],
		truth=[],
	)
	num_valid = 0
	for t, index in enumerate(np.arange(0, len(nonshare_idx), cfg.valid_batch_size)):
		index = nonshare_idx[index:index + cfg.valid_batch_size]

		B = len(index)
		batch = dotdict(
			smiles_token_id=torch.from_numpy(token_id[index]).byte().cuda(),
			bind=torch.from_numpy(bind[index]).float().cuda(),
		)
		with torch.cuda.amp.autocast(enabled=cfg.is_amp):
			with torch.no_grad():
				output = net(batch)   # data_parallel(net,batch)#
		result1.truth.append(batch['bind'].data.cpu().numpy())
		result1.prob.append(output['bind'].data.cpu().numpy())
		num_valid +=B

		print(f'\r validation: {num_valid}/{len(nonshare_idx)}', time_to_str(timer() - start_timer, 'min'),
			  end='', flush=True)

	# ----
	t = np.concatenate(result.truth)
	p = np.concatenate(result.prob)
	p = np.nan_to_num(p,nan=0,posinf=0,neginf=0)
	#np.savez_compressed('result.valid_idx.npz', prob=p, truth=t, index = valid_idx )

	#PROTEIN_NAME=['BRD4', 'HSA', 'sEH']
	bce_loss = np_binary_cross_entropy_loss(p,t)
	micro = average_precision_score(t, p, average='micro')
	BRD4 = average_precision_score(t[:,0], p[:,0])
	HSA  = average_precision_score(t[:,1], p[:,1])
	sEH  = average_precision_score(t[:,2], p[:,2])
	avg = (BRD4+HSA+sEH)/3

	# ----
	t = np.concatenate(result1.truth)
	p  = np.concatenate(result1.prob)
	p  = np.nan_to_num(p,nan=0,posinf=0,neginf=0)
	#np.savez_compressed('result.nonshare_idx.npz', prob=p, truth=t, index = nonshare_idx )

	BRD4_non = average_precision_score(t[:,0], p[:,0])
	HSA_non  = average_precision_score(t[:,1], p[:,1])
	sEH_non  = average_precision_score(t[:,2], p[:,2])
	avg_non = (BRD4_non+HSA_non+sEH_non)/3
	lb = (avg+avg_non)/2


	print('')
	log.write(f'bce_loss  : {bce_loss}')
	log.write(f'lb : {lb}')
	log.write(f'micro : {micro}')
	log.write(f'avg  : {avg}')
	log.write(f'BRD4 : {BRD4}')
	log.write(f'HSA  : {HSA}')
	log.write(f'sEH  : {sEH}')
	log.write(f'avg_non  : {avg_non}')
	log.write(f'BRD4_non : {BRD4_non}')
	log.write(f'HSA_non  : {HSA_non}')
	log.write(f'sEH_non  : {sEH_non}')


# main #################################################################
if __name__ == '__main__':
	from configure import default_cfg

	# validate fold 0 ===============================
	if 0:
		fold = 0
		cfg = deepcopy(default_cfg)
		cfg.fold=fold
		cfg.fold_dir=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'
		cfg.resume_from.checkpoint = f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}/checkpoint/00400000.pth'
		run_valid(cfg)

	# validate fold 1 ===============================
	if 0:
		fold = 1
		cfg = deepcopy(default_cfg)
		cfg.fold=fold
		cfg.fold_dir=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'
		cfg.resume_from.checkpoint = f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}/checkpoint/00550000.pth'
		run_valid(cfg)

	# validate fold 3 ===============================
	if 3:
		fold = 3
		cfg = deepcopy(default_cfg)
		cfg.fold=fold
		cfg.fold_dir=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'
		cfg.resume_from.checkpoint = f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}/checkpoint/00415000.pth'
		run_valid(cfg)