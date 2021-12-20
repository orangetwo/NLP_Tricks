# Author   : Orange
# Coding   : Utf-8
# @Time    : 2021/11/19 4:57 下午
# @File    : Mixup.py
import argparse

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def args():
	parser = argparse.ArgumentParser()
	args = parser.parse_args()
	args.embed_size = 300
	args.kernel_size = [3, 4, 5]
	args.num_channels = 100
	args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	args.num_class = 2
	args.lr = 1e-2

	return args


class TextCNN(nn.Module):
	def __init__(self, vocab_size, word_embeddings=None, fine_tune=True, dropout=0.5, args=args):
		super(TextCNN, self).__init__()

		# Embedding Layer
		self.embeddings = nn.Embedding(vocab_size, args.embed_size)
		if word_embeddings is not None:
			self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=fine_tune)

		# Conv layers
		self.convs = nn.ModuleList([nn.Conv2d(1, args.num_channels, [k, args.embed_size]) for k in args.kernel_size])

		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(args.num_channels * len(args.kernel_size), args.num_class)

	def forward(self, x):
		# (batch, seq_len, embed)
		x = self.embeddings(x).permute(1, 0, 2)
		# (batch, channel, seq_len, embed)
		x = torch.unsqueeze(x, 1)

		# (batch, channel, seq_len-k+1)
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

		# (batch, channel)
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

		# (batch, #filters * channel)
		x = torch.cat(x, 1)

		x = self.dropout(x)

		# (batch, #class)
		x = self.fc(x)
		return x

	def _forward_dense(self, x):
		x = self.embeddings(x).permute(1, 0, 2)
		x = torch.unsqueeze(x, 1)
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
		x = torch.cat(x, 1)
		return x

	def forward_mix_embed(self, x1, x2, lam):
		# (seq_len, batch) -> (batch, seq_len, embed)
		x1 = self.embeddings(x1).permute(1, 0, 2)
		x2 = self.embeddings(x2).permute(1, 0, 2)
		x = lam * x1 + (1.0 - lam) * x2

		x = torch.unsqueeze(x, 1)
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
		x = torch.cat(x, 1)
		x = self.dropout(x)
		x = self.fc(x)
		return x

	def forward_mix_sent(self, x1, x2, lam):
		y1 = self.forward(x1)
		y2 = self.forward(x2)
		y = lam * y1 + (1.0 - lam) * y2
		return y

	def forward_mix_encoder(self, x1, x2, lam):
		y1 = self._forward_dense(x1)
		y2 = self._forward_dense(x2)
		y = lam * y1 + (1.0 - lam) * y2
		y = self.fc(y)
		return y


def mixup_criterion_cross_entropy(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_perm(self, x, device):
	"""get random permutation"""
	batch_size = x.size()[0]
	index = torch.randperm(batch_size).to(device)
	return index


config = args()
model = TextCNN(10000, word_embeddings=None, fine_tune=True, args=config)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)
method = 'sent'


def train_mixup(self, epoch, args):
	train_loss = 0
	total = 0
	correct = 0

	for x, y in enumerate(self.dataloader):
		model.train()
		optimizer.zero_grad()
		x, y = x.to(args.device), y.to(args.device)
		# beta 分布采样
		lam = np.random.beta(1, 0, 1.0)
		index = get_perm(x, device=args.device)
		x1 = x[:, index]
		y1 = y[index]

		if method == 'embed':
			y_pred = model.forward_mix_embed(x, x1, lam)
		elif method == 'sent':
			y_pred = model.forward_mix_sent(x, x1, lam)
		elif method == 'encoder':
			y_pred = model.forward_mix_encoder(x, x1, lam)
		else:
			raise ValueError('invalid method name')

		loss = mixup_criterion_cross_entropy(criterion, y_pred, y, y1, lam)
		train_loss += loss.item() * y.shape[0]
		total += y.shape[0]
		_, predicted = torch.max(y_pred.data, 1)
		correct += ((lam * predicted.eq(y.data).cpu().sum().float()
		             + (1 - lam) * predicted.eq(y1.data).cpu().sum().float())).item()

		loss.backward()

		optimizer.step()
