from common import *
from leashbio import *

def data_to_text(token_id, mask_id, bind):
	text =''
	text += f'\ttoken_id: {str(token_id.shape)} \n'
	text += f'\tmask_id: {str(mask_id.shape)} \n'
	text += f'\tbind: {str(bind.shape)}, mean{np.array2string(np_non_neg_mean(bind,0), formatter={"float_kind":lambda x: "%.5f" % x},)}\n'  #
	return text[:-1]



def load_kaggle_data():
	print('load_kaggle_data()...')
	start_timer = timer()

	bind = np.load(f'{PROCESSED_DATA_DIR}/train.bind.npz')['bind']
	tokenized_file = \
		f'{PROCESSED_DATA_DIR}/tokenized/my36-token/train-replace-c.padded_token_id.npz'
		#f'/mnt/md0/2024/kaggle/leash-belka/data/other/reduced-2/tokenized/molformer-token/train-replace-c.padded_token_id.npz'

	tokenized = np.load(tokenized_file)
	padded_token_id   = tokenized['padded_token_id']
	padded_token_mask = tokenized['padded_token_mask']
	token_length = tokenized['token_length']

	print('end:', time_to_str(timer() - start_timer, 'min'))
	return padded_token_id, padded_token_mask, bind


def load_submit_data():
	print('load_kaggle_data()...')
	start_timer = timer()

	tokenized = np.load(
		f'{PROCESSED_DATA_DIR}/tokenized/my36-token/test-replace-c.padded_token_id.npz')
	bind = np.zeros((878022,3),np.uint8)

	padded_token_id   = tokenized['padded_token_id']
	padded_token_mask = tokenized['padded_token_mask']
	token_length = tokenized['token_length']
	print('end:', time_to_str(timer() - start_timer, 'min'))
	return padded_token_id, padded_token_mask, bind




def run_check_dataset():
	train_idx, valid_idx, nonshare_idx, unused_idx = load_kaggle_idx4(fold = 0)
	token_id, token_mask, bind = load_kaggle_data()

	start_timer = timer()
	print('dataset:kaggle')
	print(data_to_text(token_id, token_mask,  bind))
	print(index_to_text3(train_idx, valid_idx, nonshare_idx))
	print(bind_to_text(bind[train_idx]))
	print(bind_to_text(bind[valid_idx]))
	print(bind_to_text(bind[nonshare_idx]))
	print(f'end:', time_to_str(timer() - start_timer,'min'), get_used_mem(), 'gb')


	# 	#i = np.random.choice(len(dataset))
	for i in range(10):
		i = np.random.choice(len(train_idx))
		print(i, '--------------------')
		k,v = 'token_id', token_id[i]
		print(k)
		print('\t', 'dtype:', v.dtype)
		print('\t', 'shape:', v.shape)
		print('\t', 'min/max:', v.min(), '/', v.max())
		print('\t', 'sum:', v.sum())
		print('\t', 'values:')
		print('\t\t', v.reshape(-1)[:3].tolist(), '...',  v.reshape(-1)[-3:])
		print('')


		k,v = 'token_mask', token_mask[i]
		print(k)
		print('\t', 'dtype:', v.dtype)
		print('\t', 'shape:', v.shape)
		print('\t', 'min/max:', v.min(), '/', v.max())
		print('\t', 'sum:', v.sum())
		print('\t', 'values:')
		print('\t\t', v.reshape(-1)[:3].tolist(), '...',  v.reshape(-1)[-3:])
		print('')

		k,v = 'bind', bind[i]
		print(k)
		print('\t', 'dtype:', v.dtype)
		print('\t', 'shape:', v.shape)
		print('\t', 'min/max:', v.min(), '/', v.max())
		print('\t', 'sum:', v.sum())
		print('\t', 'values:')
		print('\t\t', v.reshape(-1)[:3].tolist(), '...',  v.reshape(-1)[-3:])
		print('')

# main #################################################################
if __name__ == '__main__':
	run_check_dataset()
