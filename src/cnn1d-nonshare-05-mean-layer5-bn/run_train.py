from trainer import *
from configure import default_cfg



#train fold 0 ===============================
fold=0
cfg = deepcopy(default_cfg)
cfg.fold=fold
cfg.fold_dir=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'

#train at  1e-3 ----
cfg.resume_from.checkpoint=None
cfg.lr=1e-3
cfg.num_epoch=9

run_train(cfg)

#finetune at  1e-4 ----
cfg.resume_from.checkpoint=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}/checkpoint/00310000.pth'
cfg.lr=1e-4
cfg.num_epoch=11

run_train(cfg)

#finetune at  5e-5 ----
cfg.resume_from.checkpoint=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}/checkpoint/00380000.pth'
cfg.lr=5e-5
cfg.num_epoch=11


#train fold 1 ===============================
fold=1
cfg = deepcopy(default_cfg)
cfg.fold=fold
cfg.fold_dir=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'

#train at  1e-3 ----
cfg.resume_from.checkpoint=None
cfg.lr=1e-3
cfg.num_epoch=11

run_train(cfg)

#finetune at  1e-4 ----
cfg.resume_from.checkpoint=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}/checkpoint/00405000.pth'
cfg.lr=1e-4
cfg.num_epoch=15

run_train(cfg)


#train fold 3 ===============================
fold=3
cfg = deepcopy(default_cfg)
cfg.fold=fold
cfg.fold_dir=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'

#train at  1e-3 ----
cfg.resume_from.checkpoint=None
cfg.lr=1e-3
cfg.num_epoch=10

run_train(cfg)

#finetune at  1e-4 ----
cfg.resume_from.checkpoint=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}/checkpoint/00375000.pth'
cfg.lr=1e-4
cfg.num_epoch=11

run_train(cfg)

