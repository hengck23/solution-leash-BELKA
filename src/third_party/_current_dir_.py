import sys, os

# project root directory
# ROOT_DIR = '.'
ROOT_DIR = '/'.join(__file__.split('/')[:-4])
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print('ROOT_DIR:', ROOT_DIR)
#sys.path.append(f'{ROOT_DIR}/src')
#sys.path.append(f'{ROOT_DIR}/src/third_party')

# kaggle downloaded data
# KAGGLE_DATA_DIR = f'{ROOT_DIR}/data/kaggle'
# #  prepocessed data
# PROCESSED_DATA_DIR =  f'{ROOT_DIR}/data/processed'
# # training resulst (model weights, train log) and inference results
# RESULT_DIR =  f'{ROOT_DIR}/result'

##for debug
KAGGLE_DATA_DIR = f'/home/hp/work/2024/kaggle/leash-belka/data/kaggle'
PROCESSED_DATA_DIR = f'/home/hp/work/2024/kaggle/leash-belka/data/processed'
RESULT_DIR = f'/home/hp/work/2024/kaggle/leash-belka/result'
