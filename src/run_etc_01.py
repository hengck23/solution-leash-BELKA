from common import *

f1 ='/home/hp/work/2024/kaggle/leash-belka/deliver/result/mamba-03/final-mamba-03-fold0-00255000.submit.csv'
df1 =pd.read_csv(f1)

f2 = '/home/hp/work/2024/kaggle/leash-belka/result/mamba-03/fold-0/final-transfomer-fa-03-fold0-00255000.submit.csv'
df2 =pd.read_csv(f2)

print(df1.equals(df2))
#df1.compare(df2, keep_shape=True)
mask = np.where(df1!=df2)[0]
print(df1.iloc[mask])
print(df2.iloc[mask])