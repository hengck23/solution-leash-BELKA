import pandas as pd

from common import *

from _current_dir_ import *
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# make id for unique bb in train and test
def make_bb_df():
    train_df = pd.read_parquet(f'{KAGGLE_DATA_DIR}/leash-BELKA/train.parquet')
    test_df = pd.read_parquet(f'{KAGGLE_DATA_DIR}/leash-BELKA/test.parquet')

    bb_df={}
    for i in tqdm([1, 2, 3]):
        train_bb = train_df[f'buildingblock{i}_smiles'].to_numpy()
        train_bb_unique = np.unique(train_bb)
        train_bb_unique = list(train_bb_unique)

        test_bb = test_df[f'buildingblock{i}_smiles'].to_numpy()
        test_bb_unique = np.unique(test_bb)
        test_bb_unique = list(test_bb_unique)

        all = set(test_bb_unique).union(set(train_bb_unique))
        test_only = set(test_bb_unique) - set(train_bb_unique)
        test_only = list(test_only)

        #print(i, len(all), len(test_only) + len(train_bb), len(test_only), len(train_bb))
        bb_df[i] = pd.DataFrame({
            'id': np.arange(len(all)),
            'smiles':train_bb_unique + test_only,
            'is_train':[1]*len(train_bb_unique)+[0]*len(test_only),
        })

    #-----------------------------
    # reorganize the id such that train goes first

    train_bb = []
    test_bb = []
    for i in [1, 2, 3]:
        train_bb.append(bb_df[i][bb_df[i].is_train == 1].smiles.tolist())
        test_bb.append(bb_df[i][bb_df[i].is_train == 0].smiles.tolist())

    all_train_bb = list(set(train_bb[0]).union(set(train_bb[1]).union(set(train_bb[2]))))
    all_test_bb  = list(set(test_bb[0]).union(set(test_bb[1]).union(set(test_bb[2]))))

    bb_df = pd.DataFrame({
        'id': np.arange(len(all_train_bb + all_test_bb)),
        'smiles': all_train_bb + all_test_bb,
        'is_train': [1] * len(all_train_bb) + [0] * len(all_test_bb),
    })
    print(bb_df)
    #bb_df.to_csv(f'{PROCESSED_DATA_DIR}/all_buildingblock.csv', index=False)
    #unfornately, the code uses set operation. hence the results cannot be repeated



def make_reduced_test():
    test_df = pd.read_parquet(f'{KAGGLE_DATA_DIR}/leash-BELKA/test.parquet')
    bb_df = pd.read_csv(f'{PROCESSED_DATA_DIR}/all_buildingblock.csv')
    mapping = dict(zip(bb_df['smiles'], bb_df['id']))

    reduced = {}
    for t, d in test_df.iterrows():
        if d.molecule_smiles not in reduced:
            reduced[d.molecule_smiles] = {
                'molecule_smiles': d.molecule_smiles,
                'buildingblock1_id': mapping[d.buildingblock1_smiles],
                'buildingblock2_id': mapping[d.buildingblock2_smiles],
                'buildingblock3_id': mapping[d.buildingblock3_smiles],
                'id_BRD4': -1,
                'id_HSA': -1,
                'id_sEH': -1,

            }
        reduced[d.molecule_smiles]['id_' + d.protein_name] = d['id']

    reduced_df = pd.DataFrame.from_records(list(reduced.values()))
    # [878022 rows x 7 columns]

    reduced_df.to_parquet(f'{PROCESSED_DATA_DIR}/test.reduced.parquet', index=False)
    print(reduced_df)

    # check_df = pd.read_parquet('/home/hp/work/2024/kaggle/leash-belka/data/other/reduced-2/test.reduced.parquet')
    # print('check_df', reduced_df.equals(check_df[reduced_df.columns]))
    # print(check_df)


def make_reduced_train():
    train_df = pd.read_parquet(f'{KAGGLE_DATA_DIR}/leash-BELKA/train.parquet')
    bb_df = pd.read_csv(f'{PROCESSED_DATA_DIR}/all_buildingblock.csv')
    mapping = dict(zip(bb_df['smiles'], bb_df['id']))

    molecule_smiles = train_df['molecule_smiles'].to_numpy().reshape(-1,3)
    molecule_smiles = molecule_smiles[:,0]
    protein_name = train_df['protein_name'].to_numpy()
    protein_name0 = np.unique(protein_name[0::3])
    protein_name1 = np.unique(protein_name[1::3])
    protein_name2 = np.unique(protein_name[2::3])
    print(protein_name0)#BRD4
    print(protein_name1)#HSA
    print(protein_name2)#sEH

    binds = train_df['binds'].to_numpy()
    binds_BRD4 = binds[0::3]
    binds_HSA  = binds[1::3]
    binds_sEH  = binds[2::3]

    buildingblock1_smiles = train_df['buildingblock1_smiles'].to_numpy().reshape(-1,3)[:,0]
    buildingblock2_smiles = train_df['buildingblock2_smiles'].to_numpy().reshape(-1,3)[:,0]
    buildingblock3_smiles = train_df['buildingblock3_smiles'].to_numpy().reshape(-1,3)[:,0]
    buildingblock1_id = [mapping[x] for x in buildingblock1_smiles]
    buildingblock2_id = [mapping[x] for x in buildingblock2_smiles]
    buildingblock3_id = [mapping[x] for x in buildingblock3_smiles]

    reduced_df =pd.DataFrame({
        'molecule_smiles':molecule_smiles,
        'buildingblock1_id':buildingblock1_id,
        'buildingblock2_id':buildingblock2_id,
        'buildingblock3_id':buildingblock3_id,
        'binds_BRD4':binds_BRD4,
        'binds_HSA':binds_HSA,
        'binds_sEH':binds_sEH,
    })
    print(reduced_df)
    reduced_df.to_parquet(f'{PROCESSED_DATA_DIR}/train.reduced.parquet',index=False)

    #check_df = pd.read_parquet('/home/hp/work/2024/kaggle/leash-belka/data/other/reduced-2/train.reduced.parquet')
    #print('check_df', reduced_df.equals(check_df[reduced_df.columns]))
    #print(check_df)

def make_bind_train():
    reduced_df =pd.read_parquet(f'{PROCESSED_DATA_DIR}/train.reduced.parquet')
    bind = reduced_df[['binds_BRD4', 'binds_HSA', 'binds_sEH']].values
    bind = bind.astype(np.uint8)
    np.savez_compressed(f'{PROCESSED_DATA_DIR}/train.bind.npz', bind=bind)

    # check = np.load( f'/home/hp/work/2024/kaggle/leash-belka/data/other/reduced-2/train.bind.npz')
    # check = check['bind']
    # print ('check', (bind != check).sum())


# main #################################################################
if __name__ == '__main__':
    # unfornately, the make_bb_df() uses set operation. hence the results cannot be repeated
    #make_bb_df(cfg)

    #make_reduced_test()
    #make_reduced_train()
    make_bind_train()