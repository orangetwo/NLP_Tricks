# Author   : Orange
# Coding   : Utf-8
# @Time    : 2021/11/19 10:54 上午
# @File    : R-Drop.py


"""
R-Drop 使用 dropout为 0。3

"""
import torch.nn.functional as F

# define your task model, which outputs the classifier logits
from torch import nn

model = YourModel()


def compute_kl_loss(self, p, q, pad_mask = None):

	p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
	q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

	# pad_mask is for seq-level tasks
	if pad_mask is not None:
		p_loss.masked_fill_(pad_mask, 0.)
		q_loss.masked_fill_(pad_mask, 0.)

	# You can choose whether to use function "sum" and "mean" depending on your task
	p_loss = p_loss.sum()
	q_loss = q_loss.sum()

	loss = (p_loss + q_loss) / 2
	return loss

# keep dropout and forward twice
logits = model(x)

logits2 = model(x)

# cross entropy loss for classifier
cross_entropy_loss = nn.CrossEntropyLoss()
ce_loss = 0.5 * (cross_entropy_loss(logits, label) + cross_entropy_loss(logits2, label))

kl_loss = compute_kl_loss(logits, logits2)

# carefully choose hyper-parameters
α = 5
loss = ce_loss + α * kl_loss

