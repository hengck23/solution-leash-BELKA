from make_reduced_df import *
from make_replaced_smiles import *
from make_my36_tokenized import *
from make_fold_split import *

#---
# unfornately, the make_bb_df() uses set operation. hence the results cannot be repeated
# make_bb_df(cfg)
make_reduced_test()
make_reduced_train()
make_bind_train()
make_replaced()
make_tokenized()

make_fold()