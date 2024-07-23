from common import *
from leashbio import *
from multiprocessing import Pool
import gc

from _current_dir_ import *
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

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



def make_token_id(s):
	#t = np.frombuffer(str.encode(s), np.uint8)
	t = np.frombuffer(s, np.uint8)
	t = MOLECULE_LUT[t]
	t = t.tolist()
	return t

def make_tokenized():
    for mode in ['test', 'train']:
        clean_smiles = load_compressed_bz2_pickle(f'{PROCESSED_DATA_DIR}/smiles-string/{mode}-replace-c.smiles.bytestring.bz2')

        start_timer = timer()
        with Pool(processes=60) as pool:
            token_id = pool.map(make_token_id, clean_smiles)
        print(f'make_tokenized', time_to_str(timer() - start_timer, 'min'), flush=True)
        print('clean_smiles', len(clean_smiles))
        print(clean_smiles[0], token_id[0])
        print(clean_smiles[-1], token_id[-1])
        del clean_smiles
        gc.collect()

        N = len(token_id)
        padded_token_id = np.full((N, MAX_LENGTH), fill_value=PAD, dtype=np.uint8)
        padded_token_mask = np.full((N, MAX_LENGTH), fill_value=0, dtype=np.uint8)

        token_length = []
        for i, s in enumerate(token_id):
            if i % 200_000 == 0: print('\r', i, end='', flush=True)
            t = token_id[i]
            L = len(t) + 2
            padded_token_id[i, :L] = [BOS] + t + [EOS]
            padded_token_mask[i, :L] = 1
            # padded_token_id.append( [BOS] + t + [EOS] + [PAD] * (MAX_LENGTH - L))
            # padded_token_mask.append([1] * L + [0] * (MAX_LENGTH - L))
            token_length.append(L)
        print('')
        del token_id
        gc.collect()

        print('np save')
        token_length = np.array(padded_token_id, np.uint8)
        os.makedirs(f'{PROCESSED_DATA_DIR}/tokenized/my36-token', exist_ok=True)
        np.savez_compressed(f'{PROCESSED_DATA_DIR}/tokenized/my36-token/{mode}-replace-c.padded_token_id.npz',
                            padded_token_id=padded_token_id,
                            padded_token_mask=padded_token_mask,
                            token_length=token_length,
                            PAD=PAD, MAX_LENGTH=MAX_LENGTH)

def run_check_tokenized():
    for mode in ['train', 'test']:
        tokenized = np.load(f'{PROCESSED_DATA_DIR}/tokenized/my36-token/{mode}-replace-c.padded_token_id.npz')
        reference = np.load(f'/home/hp/work/2024/kaggle/leash-belka/data/other/reduced-2/tokenized/my36-token/{mode}-replace-c.padded_token_id.npz')
        tokenized = tokenized['padded_token_id']
        reference = reference['padded_token_id']
        print(mode,(tokenized!=reference).sum())

    print('all ok!')

# main #################################################################
if __name__ == '__main__':
    #make_tokenized()
    run_check_tokenized()

'''
train:
make_tokenized  0 hr 06 min
clean_smiles 98415610
b'C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1' [8, 22, 8, 8, 28, 12, 27, 12, 12, 12, 17, 8, 33, 12, 18, 35, 12, 17, 33, 8, 8, 4, 8, 8, 8, 33, 4, 12, 4, 12, 12, 12, 35, 35, 4, 19, 35, 12, 17, 33, 29, 8, 3, 3, 5, 32, 17, 8, 8, 22, 8, 19, 8, 8, 17, 26, 28, 19, 33, 8, 19, 35, 18, 19, 12, 12, 27]
b'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1' [8, 33, 8, 17, 26, 28, 19, 29, 8, 3, 5, 32, 17, 8, 8, 8, 33, 26, 29, 33, 14, 32, 26, 29, 33, 36, 32, 19, 33, 12, 27, 35, 12, 17, 33, 12, 18, 35, 13, 12, 4, 12, 12, 12, 17, 7, 19, 12, 12, 18, 4, 19, 35, 12, 17, 33, 12, 18, 35, 13, 12, 4, 12, 12, 12, 17, 7, 19, 12, 12, 18, 4, 19, 35, 27]
 98400000
np save

test:
make_tokenized  0 hr 00 min
clean_smiles 878022
b'C#CCCC[C@H](Nc1nc(Nc2ccc(C=C)cc2)nc(Nc2ccc(C=C)cc2)n1)C(=O)NC' [8, 22, 8, 8, 8, 8, 29, 8, 3, 5, 32, 17, 33, 12, 27, 35, 12, 17, 33, 12, 18, 12, 12, 12, 17, 8, 26, 8, 19, 12, 12, 18, 19, 35, 12, 17, 33, 12, 18, 12, 12, 12, 17, 8, 26, 8, 19, 12, 12, 18, 19, 35, 27, 19, 8, 17, 26, 28, 19, 33, 8]
b'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(NCc2cccs2)nc(Nc2noc3ccc(F)cc23)n1' [8, 33, 8, 17, 26, 28, 19, 29, 8, 3, 5, 32, 17, 8, 8, 8, 33, 26, 29, 33, 14, 32, 26, 29, 33, 36, 32, 19, 33, 12, 27, 35, 12, 17, 33, 8, 12, 18, 12, 12, 12, 10, 18, 19, 35, 12, 17, 33, 12, 18, 35, 13, 12, 4, 12, 12, 12, 17, 7, 19, 12, 12, 18, 4, 19, 35, 27]
 800000
np save


'''