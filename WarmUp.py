"""
https://zhuanlan.zhihu.com/p/63982470
L2正则 和weight decay 的区别:
在训练神经网络的时候，由于Adam有着收敛快的特点被广泛使用。
但是在很多数据集上的最好效果还是用SGD with Momentum细调出来的。可见Adam的泛化性并不如SGD with Momentum。
Decoupled weight decay regularization,在这篇文章中指出了Adam泛化性能差的一个重要原因就是Adam中L2正则项并不像在SGD中那么有效，
并且通过Weight Decay的原始定义去修正了这个问题。文章表达了几个观点比较有意思。

1. L2正则和Weight Decay并不等价。这两者常常被大家混为一谈。
首先两者的目的都是想是使得模型权重接近于0。L2正则是在损失函数的基础上增加L2 norm。
而权重衰减则是在梯度更新时直接增加一项。
在标准SGD的情况下，通过对衰减系数做变换，可以将L2正则和Weight Decay看做一样。
但是在Adam这种自适应学习率算法中两者并不等价。
2. 使用Adam优化带L2正则的损失并不有效。
如果引入L2正则项，在计算梯度的时候会加上对正则项求梯度的结果。
那么如果本身比较大的一些权重对应的梯度也会比较大，由于Adam计算步骤中减去项会有除以梯度平方的累积，使得减去项偏小。
按常理说，越大的权重应该惩罚越大，但是在Adam并不是这样。
而权重衰减对所有的权重都是采用相同的系数进行更新，越大的权重显然惩罚越大。在常见的深度学习库中只提供了L2正则，并没有提供权重衰减的实现。
这可能就是导致Adam跑出来的很多效果相对SGD with Momentum偏差的一个原因。
3. 在Adam上，应该使用Weight decay，而不是L2正则。

"""
from transformers import AdamW, get_linear_schedule_with_warmup

# learning_rate: default 2e-5 for text classification
learning_rate = 2e-5
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters,
                  lr=learning_rate)

# https://huggingface.co/transformers/main_classes/optimizer_schedules.html
# 这里选用线性学习率

# num_warmup_steps (int) – The number of steps for the warmup phase.
# num_training_steps (int) – The total number of training steps.
# training steps 的数量: [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=total_steps*0.1,
                                            num_training_steps=total_steps)

# 用法：
## 在每个batch training 时
for batch in train_loader:
    ...
    optimizer.step()
    scheduler.step()

