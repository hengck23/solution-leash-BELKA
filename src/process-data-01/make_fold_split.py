from common import *
from leashbio import *
from _current_dir_ import *
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def make_fold():
    rng = np.random.RandomState(123)
    bb_df = pd.read_csv(f'{PROCESSED_DATA_DIR}/all_buildingblock.csv')
    reduced_df = pd.read_parquet(f'{PROCESSED_DATA_DIR}/train.reduced.parquet')

    # building block
    bb1 = set(reduced_df['buildingblock1_id'])  # len 271
    bb2 = set(reduced_df['buildingblock2_id'])  # 693
    bb3 = set(reduced_df['buildingblock3_id'])  # 872
    bb2_bb3 = bb2 | bb3

    print('overlap bb1&bb2', len(bb1 & bb2))  # 0
    print('overlap bb1&bb3', len(bb1 & bb3))  # 0
    print('overlap bb2&bb3', len(bb3 & bb3))  # 872

    bb1_id = np.sort(np.array(list(bb1)))
    bb2_bb3_id = np.sort(np.array(list(bb2_bb3)))
    rng.shuffle(bb1_id)
    rng.shuffle(bb2_bb3_id)
    split_by_bb1 = np.array_split(bb1_id, int(round(len(bb1_id) / 25)))
    split_by_bb2_bb3 = np.array_split(bb2_bb3_id, int(round(len(bb1_id) / 50)))
    print('split_by_bb1', len(split_by_bb1))
    print('bb2_bb3_id', len(bb2_bb3_id))

    def make_one_fold(nonshare_id, reduced_df):
        nonshare_mask = (
                reduced_df.buildingblock1_id.isin(nonshare_id)
                & reduced_df.buildingblock2_id.isin(nonshare_id)
                & reduced_df.buildingblock3_id.isin(nonshare_id)
        )
        share_mask = (
                (~reduced_df.buildingblock1_id.isin(nonshare_id))
                & (~reduced_df.buildingblock2_id.isin(nonshare_id))
                & (~reduced_df.buildingblock3_id.isin(nonshare_id))
        )
        group = np.zeros(len(reduced_df), dtype=np.uint8)
        group[share_mask] = 1
        group[nonshare_mask] = 2
        return group

    #---
    df_data = {}
    for fold in [0, 1, 2, 3, 4, ]:
        nonshare_id = split_by_bb1[fold].tolist()
        nonshare_id += split_by_bb2_bb3[fold].tolist()
        group = make_one_fold(nonshare_id, reduced_df)
        unused = (group == 0).sum()
        share = (group == 1).sum()
        nonshare = (group == 2).sum()
        # np.unique(group,return_counts=True)
        print('fold', fold)
        print('unused', unused, unused / len(group))
        print('share', share, share / len(group))
        print('nonshare', nonshare, nonshare / len(group))
        print('')
        df_data[f'fold{fold}'] = group
    df = pd.DataFrame(df_data)
    df.to_parquet(f'{PROCESSED_DATA_DIR}/5fold_df.parquet', index=False)


def run_check_fold():

    fold_df = pd.read_parquet(f'{PROCESSED_DATA_DIR}/5fold_df.parquet')
    check_df = pd.read_parquet(f'/home/hp/work/2024/kaggle/leash-belka/data/other/reduced-2/5fold_df.parquet')
    print('check_df', fold_df.equals(check_df))
    print(fold_df)
    print(check_df)


# main #################################################################
if __name__ == '__main__':
    #make_fold()
    run_check_fold()

'''
overlap bb1&bb2 0
overlap bb1&bb3 0
overlap bb2&bb3 872
split_by_bb1 11
bb2_bb3_id 874
fold 0
unused 40787692 0.41444331849388527
share 57265803 0.581877234719167
nonshare 362115 0.003679446786947721

fold 1
unused 40851197 0.41508859214508753
share 57200381 0.5812124824507007
nonshare 364032 0.0036989254042117913

fold 2
unused 40754582 0.4141068881247599
share 57299240 0.5822169877319259
nonshare 361788 0.003676124143314257

fold 3
unused 41112439 0.41774306941754463
share 56933363 0.5784993153017087
nonshare 369808 0.0037576152807466214

fold 4
unused 40819672 0.41476826694464425
share 57232479 0.5815386298982448
nonshare 363459 0.00369310315711095


'''