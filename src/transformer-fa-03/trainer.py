import gc
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

def run_train(cfg):

	#--- setup ---
	start_mem = get_used_mem()
	seed_everything(cfg.seed)
	os.makedirs(cfg.fold_dir, exist_ok=True)
	for f in ['checkpoint', 'train', 'valid', 'etc']:
		os.makedirs(cfg.fold_dir + '/' + f, exist_ok=True)


	log = Logger()
	log.open(cfg.fold_dir + '/log.train.txt', mode='a')
	log.write(f'\n--- [START {log.timestamp()}] {"-" * 64}')
	log.write(f'__file__ = {__file__}\n')
	log.write(f'cfg:\n{format_dict(cfg)}')
	log.write(f'')


	#=== dataset ============================
	train_idx, valid_idx, nonshare_idx, unused_idx = load_kaggle_idx4(fold=cfg.fold)
	valid_idx = valid_idx[:500_000]#subsample for speed
	train_idx = np.concatenate([train_idx, nonshare_idx, unused_idx])

	token_id, token_mask, bind = load_kaggle_data()
	log.write('dataset:kaggle')
	log.write(data_to_text(token_id, token_mask,  bind))
	log.write(index_to_text3(train_idx, valid_idx, nonshare_idx))
	log.write(bind_to_text(bind[train_idx]))
	log.write(bind_to_text(bind[valid_idx]))
	log.write(bind_to_text(bind[nonshare_idx]))
	log.write('')

	num_train_batch = int(len(train_idx)/cfg.train_batch_size)

	#---model ---
	scaler = torch.cuda.amp.GradScaler(enabled=cfg.is_amp)
	net = Net(cfg)
	net.cuda()

	optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.lr)
	log.write(f'optimizer:\n\t{str(optimizer)}')
	log.write('')


	#--- loop ---
	start_iteration = 0
	start_epoch = 0
	if cfg.resume_from.checkpoint is not None:
		f = torch.load(cfg.resume_from.checkpoint, map_location=lambda storage, loc: storage)
		state_dict = f['state_dict']
		print(net.load_state_dict(state_dict, strict=False))  # True
		if cfg.resume_from.iteration<0:
			start_iteration = f.get('iteration', 0)
			start_epoch = f.get('epoch', 0)

	if cfg.is_torch_compile:
		net = torch.compile(net, dynamic=True)

	iter_save  = cfg.iter_save
	iter_valid = cfg.iter_valid
	iter_log   = cfg.iter_log
	train_loss = MyMeter(None, min(100,num_train_batch))#window must be less than num_train_batch
	valid_loss = [0, 0, ]


	# logging
	def message_header():
		text = ''
		text += f'** start training here! **\n'
		text += f'   experiment_name = {cfg.experiment_name} \n'
		text += f'                          |--------------------- VALID--------------------------------------------------------------|--- TRAIN/BATCH -----------------\n'
		text += f'rate      iter      epoch |  loss    lb     micro    avg    BRD4    HSA     sEH    avg_non   BRD4    HSA     sEH    |   loss   |   time  \n'
		text += f'------------------------------------------------------------------------------------------------------------------------------------------------------\n'
				 #1.00e-4  00002000    0.17 | 0.0183  0.4818  0.4012  0.4818  0.4012  0.3317  0.1802  0.6919  0.0107  0.0291  0.0110  |  0.0190  |  0 hr 03 min :  37 gb
		text = text[:-1]
		return text

	def message(mode='print'):
		if mode == 'print':
			loss = batch_loss
		if mode == 'log':
			loss = train_loss

		if (iteration % iter_save == 0):
			asterisk = '*'
		else:
			asterisk = ' '

		lr =  get_learning_rate(optimizer)[0]
		lr =  short_e_format(f'{lr:0.2e}')

		timestamp =  time_to_str(timer() - start_timer, 'min')
		text = ''
		text += f'{lr}  {iteration:08d}{asterisk} {epoch:6.2f} | '

		for v in valid_loss :
			text += f'{v:6.4f}  '
		text += f'| '

		for v in loss :
			text += f'{v:6.4f}  '
		text += f'| '

		text += f'{timestamp} : '
		text += f'{get_used_mem():3d} gb'
		return text


	### start training here! ################################################

	def do_valid():

		net.cuda()
		net.eval()
		net.output_type = ['loss', 'infer']
		start_timer = timer()

		result = dotdict(
			prob =[],
			truth=[],
		)
		num_valid = 0
		for t, index in enumerate(np.arange(0,len(valid_idx),cfg.valid_batch_size)):
			index = valid_idx[index:index+cfg.valid_batch_size]

			B = len(index)
			max_length = token_mask[index].sum(1).max()
			batch = dotdict(
				smiles_token_mask = torch.from_numpy(token_mask[index][:,:max_length]).byte().cuda(),
				smiles_token_id = torch.from_numpy(token_id[index][:,:max_length]).byte().cuda(),
				bind = torch.from_numpy(bind[index]).float().cuda(),
			)
			with torch.cuda.amp.autocast(enabled=cfg.is_amp):
				with torch.no_grad():
					output = net(batch)
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
			max_length = token_mask[index].sum(1).max()
			batch = dotdict(
				smiles_token_mask = torch.from_numpy(token_mask[index][:,:max_length]).byte().cuda(),
				smiles_token_id = torch.from_numpy(token_id[index][:,:max_length]).byte().cuda(),
				bind=torch.from_numpy(bind[index]).float().cuda(),
			)
			with torch.cuda.amp.autocast(enabled=cfg.is_amp):
				with torch.no_grad():
					output = net(batch)
			result1.truth.append(batch['bind'].data.cpu().numpy())
			result1.prob.append(output['bind'].data.cpu().numpy())
			num_valid +=B

			print(f'\r validation: {num_valid}/{len(nonshare_idx)}', time_to_str(timer() - start_timer, 'min'),
				  end='', flush=True)

		# ----
		t = np.concatenate(result.truth)
		p  = np.concatenate(result.prob)
		p  = np.nan_to_num(p,nan=0,posinf=0,neginf=0)

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

		BRD4_non = average_precision_score(t[:,0], p[:,0])
		HSA_non  = average_precision_score(t[:,1], p[:,1])
		sEH_non  = average_precision_score(t[:,2], p[:,2])
		avg_non = (BRD4_non+HSA_non+sEH_non)/3
		lb = (avg+avg_non)/2

		valid_loss = [
			bce_loss, lb,micro,
			avg,BRD4,HSA,sEH,
			avg_non, BRD4_non,HSA_non,sEH_non,
		]
		return valid_loss




	iteration   = start_iteration
	epoch       = start_epoch
	start_timer = timer()
	log.write(message_header())

	while epoch<cfg.num_epoch:
		shuffled_idx = train_idx.copy()
		np.random.shuffle(shuffled_idx)

		for t, index in enumerate(np.arange(0,len(shuffled_idx),cfg.train_batch_size)):
			index = shuffled_idx[index:index+cfg.train_batch_size]
			if len(index)!=cfg.train_batch_size: continue

			# --- start of callback ---
			if iteration % iter_save == 0:
				if (iteration != start_iteration):
					torch.save({
						#'state_dict': net.state_dict(),
						#'state_dict': clean_compile_state_dict(net.state_dict()), #old bug
						'state_dict': getattr(net, '_orig_mod', net).state_dict(),
						'iteration': iteration,
						'epoch': epoch,
					}, f'{cfg.fold_dir}/checkpoint/{iteration:08d}.pth')
					pass
 
			if iteration % iter_valid == 0:
				#if iteration != start_iteration:
					valid_loss = do_valid()

			if (iteration % iter_log == 0) or (iteration % iter_valid == 0):
				print('\r', end='', flush=True)
				log.write(message(mode='log'))

			# --- end of callback ----
			B = len(index)
			max_length = token_mask[index].sum(1).max()
			batch = dotdict(
				smiles_token_mask = torch.from_numpy(token_mask[index][:,:max_length]).byte().cuda(),
				smiles_token_id = torch.from_numpy(token_id[index][:,:max_length]).byte().cuda(),
				bind = torch.from_numpy(bind[index]).float().cuda(),
			)

			net.train()
			net.output_type = ['loss', 'infer']

			with torch.cuda.amp.autocast(enabled=cfg.is_amp):
				output = net(batch)
				bce_loss = output['bce_loss']

			optimizer.zero_grad()
			if (not torch.isnan(bce_loss)):
				if cfg.is_amp:
					scaler.scale(bce_loss).backward()
					scaler.step(optimizer)
					scaler.update()
				else:
					bce_loss.backward()
					optimizer.step()
			else:
				print('error')

			torch.clear_autocast_cache()

			# print---
			batch_loss = [bce_loss.item()]
			train_loss.step(batch_loss)

			print('\r', end='', flush=True)
			print(message(mode='print'), end='', flush=True)
			iteration +=  1
			epoch += 1/num_train_batch

# main #################################################################
if __name__ == '__main__':
	from configure import default_cfg as cfg

	run_train(cfg)
