from _current_dir_ import *
from my_lib.other import *
import numpy as np

PROTEIN_NAME=['BRD4', 'HSA', 'sEH']


# ========= cnn1d tokenization ========================================
if 1:
	# https://www.ascii-code.com/
	MOLECULE_DICT = {
		'l': 1, 'y': 2, '@': 3, '3': 4, 'H': 5, 'S': 6, 'F': 7, 'C': 8, 'r': 9, 's': 10, '/': 11, 'c': 12, 'o': 13,
		'+': 14, 'I': 15, '5': 16, '(': 17, '2': 18, ')': 19, '9': 20, 'i': 21, '#': 22, '6': 23, '8': 24, '4': 25,
		'=': 26, '1': 27, 'O': 28, '[': 29, 'D': 30, 'B': 31, ']': 32, 'N': 33, '7': 34, 'n': 35, '-': 36
	}
	MAX_MOLECULE_ID = np.max(list(MOLECULE_DICT.values()))
	VOCAB_SIZE = MAX_MOLECULE_ID + 3
	UNK = 255  # will cuase error
	BOS = MAX_MOLECULE_ID + 1
	EOS = MAX_MOLECULE_ID + 2
	# rest are reserved
	PAD = 0
	MAX_LENGTH = 160

	MOLECULE_LUT = np.full(256, fill_value=UNK, dtype=np.uint8)
	for k, v in MOLECULE_DICT.items():
		ascii = ord(k)
		MOLECULE_LUT[ascii] = v

#====================================================================================

default_cfg = dotdict(

	VOCAB_SIZE=VOCAB_SIZE,
	MAX_LENGTH=MAX_LENGTH,
	PAD=PAD,
	ROOT_DIR = ROOT_DIR,

	# --- dataset ---
	train_num_worker=0,
	train_batch_size=2_000,

	valid_num_worker=0,
	valid_batch_size=1_000,

	# --- model ---
	lr=0.00001,

	# --- loop ---
	resume_from=dotdict(
		iteration=-1,
		checkpoint=None,
	),
	num_epoch=30,
	iter_save=4000,
	iter_valid=1000,
	iter_log=1000,

	is_amp=True,  # fp16
	is_torch_compile=True,

	# --- experiment ---
	fold=2,
	seed=1234,
	experiment_name = 'transfomer-fa-03',
	fold_dir=f'{RESULT_DIR}/xxx', #f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'
)










