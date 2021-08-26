import torch
import torch.nn as nn
from torch.nn import functional as F


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True, device='cpu'):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha: 阿尔法α,类别权重.
                    当α是列表时,为各类别权重；
                    当α为常数时,类别权重为[α, 1-α, 1-α, ....],
                    alpha默认为0.25
        :param gamma: 伽马γ,难易样本调节参数.默认为2
        :param num_classes: 类别数量
        :param size_average: 损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average

        # 精细化的对每个类别赋权重
        if isinstance(alpha, list):
            assert len(alpha) == num_classes, f"the alpha length don't match class num!"
            # α可以以list方式输入,
            # size:[num_classes] 用于对不同类别精细地赋予权重
            print(f"Focal_loss alpha = {alpha}, 将对每一类权重进行精细化赋值")
            self.alpha = torch.Tensor(alpha)
        else:
            assert 0 < alpha < 1, f"alpha should be less than 1 and more than 0!"  # 如果α为一个小于1的浮点数时
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma
        self.alpha = self.alpha.to(device)

    def forward(self, predict, labels):
        """
        focal_loss损失计算
        :param predict: 预测类别. size: [Batch size, Class num] ,
        :param labels:  实际类别. size: [Batch size]
        :return:
        """

        predict = predict.view(-1, predict.size(-1))

        # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然也可以使用log_softmax,然后进行exp操作)
        predict_softmax = F.softmax(predict, dim=1)
        # 对所求概率进行 clamp 操作，不然当某一概率过小时，进行 log 操作，会使得 loss 变为 nan!!!
        predict_softmax = predict_softmax.clamp(min=0.0001, max=1.0)
        predict_logsoft = torch.log(predict_softmax)

        # 这部分实现nll_loss ( Cross Entropy = log_softmax + nll )
        predict_softmax = predict_softmax.gather(1, labels.view(-1, 1))
        predict_logsoft = predict_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - predict_softmax), self.gamma), predict_logsoft)
        # torch.pow((1-predict_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
