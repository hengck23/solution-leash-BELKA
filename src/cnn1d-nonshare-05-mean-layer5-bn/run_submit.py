import warnings
warnings.filterwarnings("ignore")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dataset import *
from model import *

import numpy as np
import random
from timeit import default_timer as timer

from my_lib.runner import *
from my_lib.file import *
from my_lib.draw import *

from sklearn.metrics import roc_curve
import sklearn.metrics
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score, precision_recall_curve


###############################################################################
def run_submit(cfg):

	#--- setup ---
	seed_everything(cfg.seed)
	os.makedirs(cfg.fold_dir, exist_ok=True)
	for f in ['checkpoint', 'train', 'valid', 'etc', 'submit']:
		os.makedirs(cfg.fold_dir + '/' + f, exist_ok=True)


	log = Logger()
	log.open(cfg.fold_dir + '/log.submit.txt', mode='a')
	log.write(f'\n--- [START {log.timestamp()}] {"-" * 64}')
	log.write(f'__file__ = {__file__}\n')
	log.write(f'cfg:\n{format_dict(cfg)}')
	log.write(f'')


	#--- dataset ---
	token_id, bind = load_submit_data()
	valid_idx = np.arange(len(token_id))
	log.write(f'fold = {cfg.fold}')
	log.write(f'valid_idx : \n{len(valid_idx)}')

	#---model ---
	net = Net(cfg)
	if cfg.resume_from.checkpoint is not None:
		f = torch.load(cfg.resume_from.checkpoint, map_location=lambda storage, loc: storage)
		state_dict = f['state_dict']
		print(net.load_state_dict(state_dict, strict=False))  # True

	### start inference here! ################################################

	if 1:
		result = dotdict(
			prob =[],
		)
		num_valid = 0

		net.cuda()
		net.eval()
		net.output_type = [ 'infer']
		start_timer = timer()
		for t, index in enumerate(np.arange(0,len(valid_idx),cfg.valid_batch_size)):
			index = valid_idx[index:index+cfg.valid_batch_size]

			B = len(index)
			batch = dotdict(
				smiles_token_id   = torch.from_numpy(token_id[index]).byte().cuda(),
				#bind = torch.from_numpy(bind[index]).float().cuda(),
			)

			with torch.cuda.amp.autocast(enabled=cfg.is_amp):
				with torch.no_grad():
					output = data_parallel(net,batch)

			num_valid += B
			result.prob.append(output['bind'].data.cpu().numpy())
			print(f'\r validation: {num_valid}/{len(valid_idx)}', time_to_str(timer() - start_timer, 'min'),
				  end='', flush=True)

		# ----
		probability  = np.concatenate(result.prob)
		print(probability.shape)
		np.savez_compressed(
			f'{cfg.fold_dir}/{cfg.name}.npz',
			prob=(probability).astype(np.float32)
		)

		reduced_df = pd.read_parquet(f'/{PROCESSED_DATA_DIR}/test.reduced.parquet')
		id = reduced_df[['id_BRD4', 'id_HSA', 'id_sEH']].to_numpy()
		assert(probability.shape==id.shape)

		id = id.reshape(-1)
		probability = probability.reshape(-1)
		mask = id!=-1
		assert(mask.sum()==1674896)
		submit_df = pd.DataFrame({
			'id':id[mask],
			'binds':probability[mask],
		})
		print(submit_df)
		submit_df.to_csv(f'{cfg.fold_dir}/{cfg.name}.submit.csv',index=False)

#######################################################################################

if __name__ == '__main__':
	from configure import default_cfg
	default_cfg.valid_batch_size = 10000

	# submit fold 0 ===============================
	if 1:
		fold = 0
		cfg = deepcopy(default_cfg)
		cfg.fold=fold
		cfg.fold_dir=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'
		cfg.resume_from.checkpoint = f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}/checkpoint/00400000.pth'
		cfg.name = f'final-cnn1d-ly5-bn-mean-pool-fold{cfg.fold}-' + cfg.resume_from.checkpoint.split('/')[-1][:-4]
		run_submit(cfg)

	# submit fold 1 ===============================
	if 0:
		fold = 1
		cfg = deepcopy(default_cfg)
		cfg.fold=fold
		cfg.fold_dir=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'
		cfg.resume_from.checkpoint = f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}/checkpoint/00550000.pth'
		cfg.name = f'final-cnn1d-ly5-bn-mean-pool-fold{cfg.fold}-' + cfg.resume_from.checkpoint.split('/')[-1][:-4]
		run_submit(cfg)


	# submit fold 3 ===============================
	if 0:
		fold = 3
		cfg = deepcopy(default_cfg)
		cfg.fold=fold
		cfg.fold_dir=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'
		cfg.resume_from.checkpoint = f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}/checkpoint/00415000.pth'
		cfg.name = f'final-cnn1d-ly5-bn-mean-pool-fold{cfg.fold}-' + cfg.resume_from.checkpoint.split('/')[-1][:-4]
		run_submit(cfg)



