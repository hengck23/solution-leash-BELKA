from trainer import *
from configure import default_cfg




#train fold 2 ===============================
fold=2
cfg = deepcopy(default_cfg)
cfg.fold=fold
cfg.fold_dir=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'

#train at  1e-4 ----
cfg.resume_from.checkpoint=None
cfg.lr=1e-4
cfg.num_epoch=5

run_train(cfg)

#finetune at  1e-5 ----
cfg.resume_from.checkpoint=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}/checkpoint/00208000.pth'
cfg.lr=1e-5
cfg.num_epoch=6

run_train(cfg)


#train fold 4 ===============================
fold=4
cfg = deepcopy(default_cfg)
cfg.fold=fold
cfg.fold_dir=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'

#train at  1e-4 ----
cfg.resume_from.checkpoint=None
cfg.lr=1e-4
cfg.num_epoch=5

run_train(cfg)

#finetune at  1e-5 ----
cfg.resume_from.checkpoint=f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}/checkpoint/00220000.pth'
cfg.lr=1e-5
cfg.num_epoch=6

run_train(cfg)

