import pandas as pd

from common import *
from _current_dir_ import *


#---------------------------
import psutil

def np_non_neg_mean(x, axis=0):
	mask = x>=0
	mean = (mask*x).sum(axis)/mask.sum(axis=axis, keepdims=True)
	return mean

def np_sigmoid(z):
	return 1/(1 + np.exp(-z))

def np_binary_cross_entropy_loss(logit_or_prob, truth, is_prob=True):
	if is_prob == False:
		logit = logit_or_prob.astype(np.float64)
		prob = np_sigmoid(logit)
	else:
		prob = logit_or_prob.astype(np.float64)

	epsilon = 1e-6  # small value to avoid division by zero
	prob = np.clip(prob,epsilon,1-epsilon)
	loss = truth * np.log(prob) + (1 - truth) * np.log(1 - prob)
	loss = (-loss).mean()
	return loss


def get_used_mem():
	memory = psutil.virtual_memory()
	total = memory.total / 1024.0 / 1024.0 / 1024.0
	available = memory.available / 1024.0 / 1024.0 / 1024.0 #in gb
	used = int(total-available)
	return used

###---------------------------
import _pickle as  cPickle
import bz2
def save_compressed_bz2_pickle(file, data):
	with bz2.BZ2File(file , 'w') as f:
		cPickle.dump(data, f)

def load_compressed_bz2_pickle(file):
	data = bz2.BZ2File(file, 'rb')
	data = cPickle.load(data)
	return data


import indexed_bzip2 as ibz2
def load_compressed_ibz2_pickle(file):
	with ibz2.open(file, parallelization=os.cpu_count()) as f:
		data = cPickle.load(f)
	return data

import mgzip
def save_compressed_mgzip_pickle(file, data):
	with mgzip.open(file , 'wb', thread=42, blocksize=2*10**8) as f:
		cPickle.dump(data, f)

###########################################################################
#split
def load_kaggle_idx4(fold=1):
	if fold==-1:
		df_group = pd.read_csv(f'{PROCESSED_DATA_DIR}/group_df.csv')
		nonshare_idx = np.where(df_group == 1)[0]
		share_idx    = np.where(df_group == 2)[0]
		unused_idx   = np.where(df_group == 0)[0]
	else:
		df_group = pd.read_parquet(f'{PROCESSED_DATA_DIR}/5fold_df.parquet')
		df_group = df_group[f'fold{fold}']
		nonshare_idx = np.where(df_group == 2)[0]
		share_idx    = np.where(df_group == 1)[0]
		unused_idx   = np.where(df_group == 0)[0]
		zz=0

	rng = np.random.RandomState(123)
	rng.shuffle(share_idx)
	train_idx, valid_idx = share_idx[:-4_000_000], share_idx[-4_000_000:]

	rng = np.random.RandomState(567)
	rng.shuffle(unused_idx)

	return train_idx, valid_idx, nonshare_idx, unused_idx

'''

# train_idx, valid_idx, nonshare_idx, unused_idx = load_kaggle_idx4(fold=1)
# print('len(train_idx)', len(train_idx))
# print('len(valid_idx)', len(valid_idx))
# print('len(nonshare_idx)', len(nonshare_idx))
# print('len(unused_idx)', len(unused_idx))

len(train_idx)    59_393_831
len(valid_idx)     4_000_000
len(nonshare_idx)    226_134
len(unused_idx)   34_795_645
all               98_415_610

'''

###################################################################
# helper print function
def index_to_text2(train_idx, valid_idx):
	t,v = train_idx,valid_idx
	text =''
	text += f'\tidx overlap train/valid: {set(train_idx).intersection(set(valid_idx))}\n'    # set()
	text += f'\ttrain_idx: {len(train_idx):06d}, [{t[0]},{t[1]} ... {t[-1]}]\n'  # print some index for debug
	text += f'\tvalid_idx: {len(valid_idx):06d}, [{v[0]},{v[1]} ... {v[-1]}]\n'  #
	return text[:-1]

def index_to_text3(train_idx, valid_idx, nonshare_idx):
	t,v,ns = train_idx, valid_idx, nonshare_idx

	text =''
	text += f'\tidx overlap train/valid: {len(set(train_idx) & set(valid_idx))}\n'
	text += f'\tidx overlap train/nonshare: {len(set(train_idx) & set(nonshare_idx))}\n'
	text += f'\ttrain_idx: {len(train_idx):06d}, [{t[0]},{t[1]} ... {t[-1]}]\n'  # print some index for debug
	text += f'\tvalid_idx: {len(valid_idx):06d}, [{v[0]},{v[1]} ... {v[-1]}]\n'  #
	text += f'\tnonshare_idx: {len(nonshare_idx):06d}, [{ns[0]},{ns[1]} ... {ns[-1]}]\n'  #
	return text[:-1]


def bind_to_text(bind, tag=''):
	text =''
	text += f'\t{tag} bind: {str(bind.shape)}, mean{np.array2string(np_non_neg_mean(bind,0), formatter={"float_kind":lambda x: "%.5f" % x},)}\n'  #
	return text[:-1]












###########################################333
#make_leashbio_fold()

'''
train_index 93415610 [86020762 88009475 98205198]
valid_index  5000000 [56351746 17705641 38522366]
'''

def plot_prc(name, labels, predictions, **kwargs):
	precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

	plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
	plt.xlabel('Precision')
	plt.ylabel('Recall')
	plt.grid(True)
	ax = plt.gca()
	ax.set_aspect('equal')


def run_make_probe_csv():
	submit_file=\
	'/home/hp/work/2024/kaggle/leash-belka/result/VERSION2/final-dnn-ecfp/nonshare-bb-fold-0-01/ver2-dnn-fp-fold0-00040000.submit.csv'
	#'/home/hp/work/2024/kaggle/leash-belka/result/VERSION2/final-mamba512/share-fold-3-fix-01/ver2-mamba512-fold3-00020000.submit.csv'
	#'/mnt/md0/2024/kaggle/leash-belka/result/submit.ensemble034f.csv'
		#'/mnt/md0/2024/kaggle/leash-belka/result/scratch/exact-512-00/fold-0/cnn1d-exact1-00170000.submit.csv'
		#'/mnt/md0/2024/kaggle/leash-belka/result/all-old-01/ChemBERTa/xx4b-from-fake-dti-add-layers/fold-0/chemberta-replace-c.00090000.submit.csv'
		#'/mnt/md0/2024/kaggle/leash-belka/result/scratch/extern-experiments-debug/fold-0/cnn1d-extern-1m-00185000.submit.csv'
		#'/mnt/md0/2024/kaggle/leash-belka/result/gcn5/simple-dim128-ly5-100m-fine-00/fold-0/graph-nn-train-ly5-00035000.submit.csv'
		#'/mnt/md0/2024/kaggle/leash-belka/result/gcn1/simple-70m-00-fine/fold-0/grahp-nn-train-50m-00115000.submit.csv'
	save_file = submit_file
	mode=['keep-nonshare']

	submit_df = pd.read_csv(submit_file)
	print('submit_df', submit_df.binds.mean())
	print('')
	#---
	test_df = pd.read_parquet(f'{kaggle_dir}/test.parquet')
	sharing_df = pd.read_csv(f'{reduced_dir}/submit_sharing.csv')
	print('check id',(submit_df['id'] != sharing_df['id']).sum())
	print('share count', sharing_df['is_share'].sum())
	print('nonshare count',len(sharing_df)-sharing_df['is_share'].sum())
	print('all',len(sharing_df))

	for m in mode:
		if 'keep-nonshare' == m:
			print(f'{m}...')
			nonshare_submit_df = submit_df.copy()
			nonshare_submit_df.loc[sharing_df.is_share==1,'binds']=0
			nonshare_submit_df.to_csv(save_file.replace('.csv',f'.{m}.csv'),index=False)
			print(nonshare_submit_df.binds.mean())
			print('')

		if 'keep-share' == m:
			print(f'{m}...')
			nonshare_submit_df = submit_df.copy()
			nonshare_submit_df.loc[sharing_df.is_share == 0, 'binds'] = 0
			nonshare_submit_df.to_csv(save_file.replace('.csv', f'.{m}.csv'), index=False)
			print(nonshare_submit_df.binds.mean())
			print('')

		# if 'boost-nonshare0.5' == m:
		# 	print(f'{m}...')
		# 	nonshare_submit_df = submit_df.copy()
		# 	nonshare_submit_df.loc[sharing_df.is_share == 0, 'binds'] **= 0.5
		# 	nonshare_submit_df.to_csv(save_file.replace('.csv', f'.{m}.csv'), index=False)
		# 	print(nonshare_submit_df.binds.mean())
		# 	print('')

		if 'keep-nonshare-BRD4' == m:
			print(f'{m}...')
			nonshare_submit_df = submit_df.copy()
			nonshare_submit_df.loc[~((sharing_df.is_share == 0) & (test_df.protein_name == 'BRD4')), 'binds'] = 0
			nonshare_submit_df.to_csv(save_file.replace('.csv', f'.{m}.csv'), index=False)
			print(nonshare_submit_df.binds.mean())
			print('')

		if 'keep-nonshare-sEH' == m:
			print(f'{m}...')
			nonshare_submit_df = submit_df.copy()
			nonshare_submit_df.loc[~((sharing_df.is_share == 0) & (test_df.protein_name == 'sEH')), 'binds'] = 0
			nonshare_submit_df.to_csv(save_file.replace('.csv', f'.{m}.csv'), index=False)
			print(nonshare_submit_df.binds.mean())
			print('')
	exit(0)

# main #################################################################
if __name__ == '__main__':
	run_make_probe_csv()



