# Author   : Orange
# coding   : Utf-8
# @Time    : 2021/9/5 3:04 下午
# @File    : BeamSearch.py
# @Software: PyCharm
import torch


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        # 束搜索的宽度
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9
        self.length_penalty = length_penalty

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            # 可更新的情况：数量未饱和或超过最差得分
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                # 数量饱和需要删掉一个最差的
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        相关样本是否已经完成生成。
        best_sum_logprobs是新的候选序列中的最高得分。
        """

        if len(self) < self.num_beams:
            return False
        else:
            """
            虽然这里transformers库中提供了cur_len,但是我人为这个cur_len应该永久被设定为None, 
            假设 len(self.beams) 此时已经满足了 等于 beam size, 在假设 beams有 这样一个序列 "我想回家"
            但是 如果cur_len 不为None,即为序列吃长度时,
            可能存在 best_sum_logprobs("我想回家")/ 4 比 best_sum_logprobs("我想回家睡觉")/6 小
            而此时 is_done 可能已经为 True(存在这种可能性) 就没办法更新 beam
            """
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            # 是否最高分比当前保存的最低分还差
            ret = self.worst_score >= cur_score
            return ret


def MaskForDecoder(x, pad_idx, mode='decoder'):
    # TODO: define your mask function for your decoder
    # for test
    def get_pad_mask(seq, pad_idx):
        # src_mask = [batch size, 1, src len]
        return (seq != pad_idx).unsqueeze(-2)

    def get_subsequent_mask(seq):
        ''' For masking out the subsequent info. '''
        # seq : [batch size, trg len]
        sz_b, len_s = seq.size()

        # subsequent_mask: [1, trg len, trg len]
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        return subsequent_mask

    if mode == 'decoder':
        return get_pad_mask(x, pad_idx) & get_subsequent_mask(x)

    elif mode == 'encoder':
        return get_pad_mask(x, pad_idx)


def beam_search_generate(encoder_output,
                         decoder,
                         vocab_size,
                         batch_size=3,
                         max_length=20,
                         min_length=2,
                         num_beams=2,
                         bos_token_id=101,
                         pad_token_id=0,
                         eos_token_id=102,
                         device=torch.device('cpu'),
                         length_penalty=0.1,
                         early_stopping=True
                         ):
    """
	encoder_output: Transformer output, (batch size * num beams, sequence length, hidden dimension)
			so, the input of Transformer encoder shape - > (batch size * num beams , sequence length)
				code like this: input4encoder - > (batch size, sequence length)
								input4encoder = torch.unsqueeze(input4encoder, 1)
								input4encoder = input4encoder.repeat(1, num beams, 1)
								input4encoder = input4encoder.reshape(batch size * num beams, sequence length)
								encoder_output = Transformer.encoder(input4encoder)

	"""

    encoder_output = torch.unsqueeze(encoder_output, 1).repeat(1, num_beams, 1, 1).\
        reshape(batch_size * num_beams, encoder_output.shape[1], encoder_output[2]).to(device)
    encoder_mask =

    # 建立beam容器，每个样本一个
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty)
        for _ in range(batch_size)
    ]

    # 每个beam容器的得分
    # beam_scores : (batch size * num beams)
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
    beam_scores = beam_scores.view(-1)

    # 每个样本是否完成生成，共batch_size个
    done = [False for _ in range(batch_size)]

    # 为了并行计算，一次生成batch_size*num_beams个序列
    # 第一步自动填入bos_token, decoder input 初始化
    input_ids = torch.full(
        (batch_size * num_beams, 1),
        bos_token_id,
        dtype=torch.long,
        device=device,
    )

    # 当前长度设为1
    cur_len = 1

    while cur_len < max_length:
        # 将编码器得到的上下文向量和当前结果输入解码器，这部分代码要结合到Transformer模型的具体实现
        # output：(batch size *num beams, current length , vocab size)
        mask4decoder = MaskForDecoder(input_ids, pad_token_id)
        output = decoder.decode_next_step(encoder_output, input_ids, mask4decoder,)

        # 取出最后一个时间步的各token概率，即当前条件概率
        # score : (batch size * num beams, vocab size),  = log + softmax + logit
        scores = next_token_logits = output[:, -1, :]

        ###########################
        # 这里可以做一大堆操作减少重复 #
        ###########################

        """
        计算序列条件概率的，因为取了log，所以直接相加即可。
        beam_scores : (batch size * num beams) 表示序列之前的log-softmax之和, 共batch * num beams 个序列
        beam_scores[:, None] : (batch size * num beams, 1)
        beam_scores[:, None].expand_as(scores) : (batch size * num beams , vocab size), 
        如((batch size * num beams 设定为5, vocab size 设定为4): 
        
        beam_score                     ->                beam_scores[:, None].expand_as(scores)
        tensor([[-1.4581],                               tensor([[-1.4581, -1.4581, -1.4581, -1.4581],
                [ 1.7265],                                       [ 1.7265,  1.7265,  1.7265,  1.7265],
                [-0.0455],                                       [-0.0455, -0.0455, -0.0455, -0.0455],
                [ 0.8145],                                       [ 0.8154,  0.8154,  0.8154,  0.8154],
                [ 1.7904]])                                      [ 1.7904,  1.7904,  1.7904,  1.7904]])
        """
        # next_scores: (batch_size * num_beams, vocab size)
        next_scores = scores + beam_scores[:, None].expand_as(scores)

        # 为了提速，将结果重排成图中3的形状,方便取top k
        # next_scores : (batch size, num beams * vocab_size)
        next_scores = next_scores.view(
            batch_size, num_beams * vocab_size
        )

        # 取出分数最高的token（图中黑点）和其对应得分
        # sorted=True，保证返回序列是有序的
        # 取的 tokens的个数要 >= 2*num beams, 如果小于 2*num beams, 假设其中含有 num beams个eos,则之后的score维度要发生改变
        next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        """
        下一个时间步整个batch的beam列表
        列表中的每一个元素都是三元组
        (分数, token_id, beam_id)
        """
        next_batch_beam = []

        # 对每一个样本进行扩展
        for batch_idx in range(batch_size):

            # 检查样本是否已经生成结束
            if done[batch_idx]:
                # 对于已经结束的句子，待添加的是pad token
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # 当前样本下一个时间步的beam列表
            """
            存在三元组, 假设当前词预测出为 "家", 则有 ("我想回家"的得分, "家"对应的词典的位置, "我想回"在input_ids中的位置)
            有了三元组 就可以更新"我想回"这个序列了
            """
            next_sent_beam = []

            # 对于还未结束的样本需要找到分数最高的num_beams个扩展
            # 注意，next_scores和next_tokens是对应的
            # 而且已经按照next_scores排好顺序
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                # get beam and word IDs
                # 这两行可参考图中3进行理解
                # 对于next_scores
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size

                # 用来判定是哪个序列,作用在 input_ids中, input_ids : (batch size * num beams, current length)
                effective_beam_id = batch_idx * num_beams + beam_id

                # 如果出现了EOS token说明已经生成了完整句子
                if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added

                    # eos token的排名是不是在num beams排名开外
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    # eos token的排名在前num beams排名开外(不在候选的vocab中的概率最大的前num beams个中)
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    # 往容器中添加这个序列
                    # input_ids 存储序列的token id
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id].clone(), beam_token_score.item(),
                    )
                else:
                    # add next predicted word if it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                # 扩展num_beams个就够了
                if len(next_sent_beam) == num_beams:
                    break

            # 检查这个样本是否已经生成完了，有两种情况
            # 1. 已经记录过该样本结束
            # 2. 新的结果没有使结果改善
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len=cur_len
            )

            # 把当前样本的结果添加到batch结果的后面
            next_batch_beam.extend(next_sent_beam)

        # 如果全部样本都已经生成结束便可以直接退出了
        if all(done):
            break

        # 把三元组列表再还原成三个独立列表
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        # 准备下一时刻的解码器输入
        # 取出实际被扩展的beam
        input_ids = input_ids[beam_idx, :]
        # 在这些beam后面接上新生成的token
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)

        # 更新当前长度
        cur_len = cur_len + 1
# end of length while
