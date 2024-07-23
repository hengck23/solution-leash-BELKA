
import pandas as pd

from _current_dir_ import *
test_df = pd.read_parquet(f'{KAGGLE_DATA_DIR}/leash-BELKA/test.parquet')

def read_and_average_df(base_file):

	L=len(base_file)
	probability=0
	id = None
	w_sum = 0
	for i in range(L):
		df = pd.read_csv(base_file[i])
		assert((df['id']!=test_df['id']).sum()==0)
		if i==0:
			id = df['id'].to_numpy()
		print(i,(id!=df['id'].to_numpy()).sum())
		probability += df['binds'].to_numpy()
		w_sum += 1
	probability = probability/w_sum
	return pd.DataFrame({'id':id,'binds':probability})

result_dir = RESULT_DIR
#result_dir = '/home/hp/work/2024/kaggle/leash-belka/deliver/result'
submit_df = read_and_average_df([
	f'{result_dir}/cnn1d-mean-pool-ly5-bn-01/fold-3/final-cnn1d-ly5-bn-mean-pool-fold3-00415000.submit.csv',
	f'{result_dir}/cnn1d-mean-pool-ly5-bn-01/fold-0/final-cnn1d-ly5-bn-mean-pool-fold0-00400000.submit.csv',
	f'{result_dir}/cnn1d-mean-pool-ly5-bn-01/fold-1/final-cnn1d-ly5-bn-mean-pool-fold1-00550000.submit.csv',
	f'{result_dir}/transfomer-fa-03/fold-2/final-transfomer-fa-03-fold2-00264000.submit.csv',
	f'{result_dir}/transfomer-fa-03/fold-4/final-transfomer-fa-03-fold4-00264000.submit.csv',
	f'{result_dir}/mamba-03/final-mamba-03-fold0-00255000.submit.csv',
])
ensemble_file=f'{result_dir}/final-3fold-tx2a-mamba-fix.submit.csv'

print(submit_df)
submit_df.to_csv(ensemble_file,index=False)
