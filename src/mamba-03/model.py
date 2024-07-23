
from common import *
from pytorch_packbits import unpackbits as F_unpackbits
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling import *
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
from modeling import _init_weights
from mamba_ssm import Mamba,Mamba2

from functools import partial
BN = partial(nn.BatchNorm1d, eps=1e-3,momentum=0.1)

##############################################################################################
class LinearBnRelu(nn.Module):
	def __init__(self, in_dim, out_dim, is_bn=True):
		super().__init__()
		self.is_bn = is_bn
		self.linear = nn.Linear(in_dim, out_dim, bias=not is_bn)
		self.bn=BN(out_dim)

	def forward(self, x):
		x = self.linear(x)
		if self.is_bn:
			x=self.bn(x)
		x = F.relu(x,inplace=True)
		return x

class Conv1dBnRelu(nn.Module):
	def __init__(self, in_channel,out_channel,kernel_size,stride=1,padding=0, is_bn=True):
		super().__init__()
		self.is_bn = is_bn
		self.conv=nn.Conv1d(
			in_channel, out_channel, kernel_size=kernel_size,
			stride=stride, padding=padding, bias=not is_bn
		)
		self.bn=BN(out_channel)

	def forward(self, x):
		x=self.conv(x)
		if self.is_bn:
			x=self.bn(x)
		x = F.relu(x,inplace=True)
		return x

##############################################################################################


class Net(nn.Module):
	def __init__(self,cfg ):
		super().__init__()

		embed_dim=256
		num_layer=6
		self.output_type = ['infer', 'loss']
		self.embedding = nn.Embedding(cfg.VOCAB_SIZE, 64, padding_idx=cfg.PAD)

		self.conv_embedding = nn.Sequential(
			Conv1dBnRelu(64, embed_dim, kernel_size=3,stride=1,padding=1, is_bn=True),
		)

		self.mamba_encoder = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    d_intermediate=embed_dim//2,
                    ssm_cfg={'layer': 'Mamba1'},
                    attn_layer_idx=None,
                    attn_cfg=None,
                    norm_epsilon=1e-4,
                    rms_norm=1e-4,
                    residual_in_fp32=False,
                    fused_add_norm=True,
                    layer_idx=i,
                )
                for i in range(num_layer)
            ])

		self.norm_f = nn.LayerNorm ( #RMSNorm
			embed_dim, eps=1e-4
		)

		self.bind = nn.Sequential(
			nn.Linear(embed_dim, 3),
		)

		self.apply(
			partial(
				_init_weights,
				n_layer=num_layer,
				n_residuals_per_layer=2,
			)
		)

	def forward(self, batch):
		smiles_token_id = batch['smiles_token_id'].long()
		smiles_token_mask = batch['smiles_token_mask'].long()
		B, L  = smiles_token_id.shape

		x = self.embedding(smiles_token_id)
		x = x.permute(0,2,1).float()
		x = self.conv_embedding(x)
		x = x.permute(0,2,1).contiguous()
		#x = self.pe(x)

		hidden, residual = x, None
		for mamba in self.mamba_encoder:
			hidden, residual = mamba(
				hidden, residual, inference_params=None
			)
			hidden = F.dropout(hidden,p=0.1, training=self.training)

		#z=hidden
		z = layer_norm_fn(
			hidden,
			self.norm_f.weight,
			self.norm_f.bias,
			eps=self.norm_f.eps,
			residual=residual,
			prenorm=False,
			residual_in_fp32=False,
			is_rms_norm=isinstance(self.norm_f, RMSNorm)
		)

		#pool = z.mean(1)
		m = smiles_token_mask.unsqueeze(2).float()
		pool = (z*m).sum(1)/m.sum(1)
		bind = self.bind(pool)

		# --------------------------
		output = {}
		if 'loss' in self.output_type:
			target = batch['bind']
			output['bce_loss'] = F.binary_cross_entropy_with_logits(bind.float(), target.float())

		if 'infer' in self.output_type:
			output['bind'] = torch.sigmoid(bind)

		return output





def run_check_net():
	from configure import default_cfg as cfg

	batch_size = 500
	batch = {
		'smiles_token_id': torch.from_numpy(np.random.choice(cfg.VOCAB_SIZE, (batch_size, cfg.MAX_LENGTH))).byte().cuda(),
		'smiles_token_mask': torch.from_numpy(np.random.choice(2, (batch_size, cfg.MAX_LENGTH))).byte().cuda(),
		'bind': torch.from_numpy(np.random.choice(2, (batch_size, 3))).float().cuda(),
	}

	net = Net(cfg).cuda()
	with torch.no_grad():
		with torch.cuda.amp.autocast(enabled=True): # dtype=torch.float16):
			output = net(batch)

	# ---
	print('batch')
	for k, v in batch.items():
		if k=='idx':
			print(f'{k:>32} : {len(v)} ')
		else:
			print(f'{k:>32} : {v.shape} ')

	print('output')
	for k, v in output.items():
		if 'loss' not in k:
			print(f'{k:>32} : {v.shape} ')
	print('loss')
	for k, v in output.items():
		if 'loss' in k:
			print(f'{k:>32} : {v.item()} ')



# main #################################################################
if __name__ == '__main__':
	run_check_net()



