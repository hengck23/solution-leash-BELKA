from common import *
from pytorch_packbits import unpackbits as F_unpackbits
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##############################################################################################
class LinearBnRelu(nn.Module):
	def __init__(self, in_dim, out_dim, is_bn=True):
		super().__init__()
		self.is_bn = is_bn
		self.linear = nn.Linear(in_dim, out_dim, bias=not is_bn)
		self.bn=nn.BatchNorm1d(out_dim, eps=5e-3,momentum=0.2)
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
		self.bn=nn.BatchNorm1d(out_channel, eps=5e-3, momentum=0.2)
	def forward(self, x):
		x=self.conv(x)
		if self.is_bn:
			x=self.bn(x)
		x = F.relu(x,inplace=True)
		return x

##############################################################################################

class Net(nn.Module):
	def __init__(self, cfg):
		super().__init__()

		self.output_type = ['infer', 'loss']
		self.embedding = nn.Embedding(cfg.VOCAB_SIZE, 64, padding_idx=0)
		self.token_encoder = nn.Sequential(
			Conv1dBnRelu(64, 128, kernel_size=3,stride=1,padding=1, is_bn=True),
			Conv1dBnRelu(128,256, kernel_size=3,stride=1,padding=1, is_bn=True),
			Conv1dBnRelu(256,256, kernel_size=3,stride=1,padding=1, is_bn=True),
			Conv1dBnRelu(256,256, kernel_size=3,stride=1,padding=1, is_bn=True),
			Conv1dBnRelu(256,256, kernel_size=3,stride=1,padding=1, is_bn=True),
			#nn.Dropout(0.1),
		)

		# --------------------------------
		self.bind = nn.Sequential(
			LinearBnRelu(256, 1024, is_bn=True),
			nn.Dropout(0.1),
			LinearBnRelu(1024, 1024, is_bn=True),
			nn.Dropout(0.1),
			LinearBnRelu(1024, 512, is_bn=True),
			nn.Dropout(0.1),
			nn.Linear(512, 3),
		)


	def forward(self, batch):
		smiles_token_id = batch['smiles_token_id'].long()
		B, L  = smiles_token_id.shape

		x = self.embedding(smiles_token_id)
		x = x.permute(0,2,1).contiguous()
		x = self.token_encoder(x)
		last = F.adaptive_avg_pool1d(x,1).squeeze(-1)
		bind = self.bind(last)

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
		'bind': torch.from_numpy(np.random.choice(2, (batch_size, 3))).float().cuda(),
	}
	net = Net(cfg).cuda()
	#print(net)

	with torch.no_grad():
		with torch.cuda.amp.autocast(enabled=True):
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


'''



'''
# main #################################################################
if __name__ == '__main__':
	run_check_net()



