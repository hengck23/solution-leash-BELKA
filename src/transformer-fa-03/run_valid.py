import warnings
import pandas as pd
warnings.filterwarnings("ignore")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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


###############################################################################

def run_valid(cfg):
	cfg.name='valid-'+cfg.resume_from.checkpoint.split('/')[-1][:-4]

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

	# --- dataset ---
	train_idx, valid_idx, nonshare_idx, unused_idx = load_kaggle_idx4(fold=cfg.fold)
	token_id, token_mask, bind = load_kaggle_data()

	def my_collate(index):
		max_length = token_mask[index].sum(1).max()
		batch = dotdict(
			smiles_token_mask=torch.from_numpy(token_mask[index][:, :max_length]).byte().cuda(),
			smiles_token_id=torch.from_numpy(token_id[index][:, :max_length]).byte().cuda(),
			bind=torch.from_numpy(bind[index]).float().cuda(),
			batch_size=len(index)
		)
		return batch

	log.write(f'fold = {cfg.fold}')
	log.write(bind_to_text(bind[valid_idx]))
	log.write('\n')


	# --- net ---
	net = Net(cfg)
	if cfg.resume_from.checkpoint is not None:
		f = torch.load(cfg.resume_from.checkpoint, map_location=lambda storage, loc: storage)
		state_dict = f['state_dict']
		print(net.load_state_dict(state_dict, strict=False))  # True

	net.cuda()
	net.eval()
	net.output_type = ['loss', 'infer']

	#### start validation here ######################
	result = dotdict(
		prob =[],
		truth=[],
		bce_loss=0,
	)
	num_valid = 0
	start_timer = timer()
	for t, index in enumerate(np.arange(0, len(valid_idx), cfg.valid_batch_size)):
		index = valid_idx[index:index + cfg.valid_batch_size]
		batch = my_collate(index)

		with torch.cuda.amp.autocast(enabled=cfg.is_amp):
			with torch.no_grad():
				output = data_parallel(net,batch)

		result.truth.append(batch['bind'].data.cpu().numpy())
		result.prob.append(output['bind'].data.cpu().numpy())
		num_valid += batch.batch_size

		print(f'\r validation: {num_valid}/{len(valid_idx)}', time_to_str(timer() - start_timer, 'min'),
			  end='', flush=True)

	# ----
	truth = np.concatenate(result.truth)
	prob  = np.concatenate(result.prob)
	# np.savez_compressed(
	# 	f'{cfg.fold_dir}/valid/prob-{cfg.name}.npz',
	# 	prob=prob.astype(np.float16),
	# 	truth=truth.astype(np.uint8),
	# )

	#PROTEIN_NAME=['BRD4', 'HSA', 'sEH']
	bce_loss = np_binary_cross_entropy_loss(prob,truth)
	micro = average_precision_score(truth, prob, average='micro')
	BRD4 = average_precision_score(truth[:,0], prob[:,0])
	HSA  = average_precision_score(truth[:,1], prob[:,1])
	sEH  = average_precision_score(truth[:,2], prob[:,2])
	avg = (BRD4+HSA+sEH)/3

	print('')
	log.write(f'bce_loss  : {bce_loss}')
	log.write(f'all micro : {micro}')
	log.write(f'avg  : {avg}')
	log.write(f'BRD4 : {BRD4}')
	log.write(f'HSA  : {HSA}')
	log.write(f'sEH  : {sEH}')
	log.write(f'copytext  : {bce_loss}\t{micro}\t{avg}\t{BRD4}\t{HSA}\t{sEH}')

# main #################################################################
if __name__ == '__main__':
	from configure import default_cfg
	default_cfg.valid_batch_size=5000
	default_cfg.valid_num_worker=32

	# validate fold 2 ===============================
	if 1:
		fold = 2
		cfg = deepcopy(default_cfg)
		cfg.fold=fold
		cfg.fold_dir=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'
		cfg.resume_from.checkpoint = f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}/checkpoint/00264000.pth'
		run_valid(cfg)

	# validate fold 4 ===============================
	if 1:
		fold = 4
		cfg = deepcopy(default_cfg)
		cfg.fold=fold
		cfg.fold_dir=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'
		cfg.resume_from.checkpoint = f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}/checkpoint/00264000.pth'
		run_valid(cfg)
