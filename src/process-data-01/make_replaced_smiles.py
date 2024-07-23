from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool
from leashbio import *

import pandas as pd

from common import *

from _current_dir_ import *
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


#replace linker [Dy]
def make_clean_smiles(s):
	m = Chem.MolFromSmiles(s)
	Dy, C = Chem.MolFromSmiles('[Dy]'), Chem.MolFromSmiles('C')
	m = AllChem.ReplaceSubstructs(m, Dy, C)[0]
	Chem.SanitizeMol(m)
	normalized = Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
	normalized = str.encode(normalized)
	return normalized

def make_replaced():
    for mode in ['train', 'test']:
        reduced_df = pd.read_parquet(f'{PROCESSED_DATA_DIR}/{mode}.reduced.parquet')
        smiles = reduced_df['molecule_smiles'].values
        print('smiles', len(smiles))

        start_timer = timer()
        with Pool(processes=60) as pool:
            # Pool.map results are ordered
            clean_smiles = pool.map(make_clean_smiles, smiles)
        print(f'make_clean_smiles', time_to_str(timer() - start_timer, 'min'), flush=True)
        print('clean_smiles', len(clean_smiles))
        print(clean_smiles[0])
        print(clean_smiles[-1])
        # --------------------------------------
        os.makedirs(f'{PROCESSED_DATA_DIR}/smiles-string', exist_ok=True)
        save_compressed_bz2_pickle(f'{PROCESSED_DATA_DIR}/smiles-string/{mode}-replace-c.smiles.bytestring.bz2', clean_smiles)

def run_check_replaced():
    for mode in ['train', 'test']:
        clean_smiles = load_compressed_bz2_pickle(f'{PROCESSED_DATA_DIR}/smiles-string/{mode}-replace-c.smiles.bytestring.bz2')
        reference_smiles = load_compressed_bz2_pickle(f'/home/hp/work/2024/kaggle/leash-belka/data/other/reduced-2/smiles-string/{mode}-replace-c.smiles.bytestring.bz2')
        N=len(clean_smiles)
        for i in tqdm(range(N)):
            if clean_smiles[i]!=reference_smiles[i]:
                print('error',mode,i)

    print('all ok!')

# main #################################################################
if __name__ == '__main__':
    #make_replaced()
    run_check_replaced()

'''
train:
smiles 98415610
make_clean_smiles  0 hr 25 min
clean_smiles 98415610
b'C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1'
b'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1'

test:
smiles 878022
make_clean_smiles  0 hr 00 min
clean_smiles 878022
b'C#CCCC[C@H](Nc1nc(Nc2ccc(C=C)cc2)nc(Nc2ccc(C=C)cc2)n1)C(=O)NC'
b'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(NCc2cccs2)nc(Nc2noc3ccc(F)cc23)n1'


'''