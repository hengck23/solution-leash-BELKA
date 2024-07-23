from common import *
from pytorch_packbits import unpackbits as F_unpackbits
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling import *
from functools import partial

BN = partial(nn.BatchNorm1d, eps=5e-3,momentum=0.1)

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


class PositionalEncoding(nn.Module):

	def __init__(self, d_model, max_len=256):
		super(PositionalEncoding, self).__init__()

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[ :,:x.size(1)]
		return x

class Net(nn.Module):
	def __init__(self, cfg):
		super().__init__()

		self.output_type = ['infer', 'loss']
		self.embedding = nn.Embedding(cfg.VOCAB_SIZE, 64, padding_idx=cfg.PAD)

		embed_dim=512
		self.pe = PositionalEncoding(embed_dim,max_len=256)
		self.conv_embedding = nn.Sequential(
			Conv1dBnRelu(64, embed_dim, kernel_size=3,stride=1,padding=1, is_bn=True),
		)

		self.tx_encoder = FlashAttentionTransformerEncoder(
			dim_model=embed_dim,
			num_heads=8,
			dim_feedforward=embed_dim*4,
			dropout=0.1,
			norm_first=False,
			activation=F.gelu,
			rotary_emb_dim=0,
			num_layers=7,
		)

		self.bind = nn.Sequential(
			nn.Linear(embed_dim, 3),
		)


	def forward(self, batch):
		smiles_token_id = batch['smiles_token_id'].long()
		smiles_token_mask = batch['smiles_token_mask'].long()
		B, L  = smiles_token_id.shape

		x = self.embedding(smiles_token_id)
		x = x.permute(0,2,1).float()
		x = self.conv_embedding(x)
		x = x.permute(0,2,1).contiguous()

		x = self.pe(x)
		z = self.tx_encoder(
			x=x,
			src_key_padding_mask=smiles_token_mask==0,
		)

		m = smiles_token_mask.unsqueeze(-1).float()
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

	batch_size = 32
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



