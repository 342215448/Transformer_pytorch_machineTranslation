import torch
import torch.nn.functional as F
from fpn_transformer import Transformer
from fpn_process import nopeak_mask, Spacy_tokenizer
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda' if torch.backends.mps.is_available() else 'cpu')


def init_vars(src, model, SRC, TRG):
    # 起始token为<sos>
    init_tok = TRG['<sos>']
    # 创建src_mask
    src_mask = (src != SRC['<pad>']).unsqueeze(-2).to(device)
    # 拿到起始token的encoder output
    e_output = model.encoder(src, src_mask).to(device)

    # outputs = torch.LongTensor([[init_tok]], device=opt.device)
    outputs = torch.LongTensor([[init_tok]]).to(device)
    # 创建一个size为1的下三角矩阵
    trg_mask = nopeak_mask(1).to(device)
    # 拿到<sos>的解码结果
    out = model.out(model.decoder(outputs, e_output, src_mask, trg_mask))
    # 经过softmax层得到概率分布
    out = F.softmax(out, dim=-1)
    # 拿到最高的三个位置对应的概率以及索引(对应字典中的位置)
    probs, ix = out[:, -1].data.topk(3)
    # 每一个概率都取log
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    # 创建一个(top_k*max_length)维度的0矩阵用来作为输出
    outputs = torch.zeros(3, 50, device=device).long()
    # 将top_k中的每一个的第一个放置为起始字符的索引
    outputs[:, 0] = init_tok
    # 第二个放置为概率最大的字符的索引
    outputs[:, 1] = ix[0]
    # 创建一个top_k * src_length * d_model维度的全零矩阵
    e_outputs = torch.zeros(3, e_output.size(-2), e_output.size(-1), device=device)
    # 因为是top_k（这里是3），所以需要做三个一模一样的矩阵，相当于在一个list中放了三个一模一样的矩阵，维度为input_length*d_model
    e_outputs[:, :] = e_output[0]
    return outputs, e_outputs, log_scores


# 函数比较难看懂 建议大家可以反复观看多遍来理解
def k_best_outputs(outputs, out, log_scores, i, k):
    # 对out进行取top_k的操作，需要注意的是这里相当于对三个句子的下一个token进行top_k的取值
    probs, ix = out[:, -1].data.topk(k)
    # 对每个可能的值都进行概率对累加
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0, 1)
    # 根据概率累加的结果，把三句话分别对三个beam拉直后找到概率最大的那个 至于这里为什么要进行概率的累加，在扩充的部分会讲到
    k_probs, k_ix = log_probs.view(-1).topk(k)
    # 找出topk的那几个k_probs对应在ix中的位置 row中存分别在哪几行 col存在哪几列
    row = k_ix // k
    col = k_ix % k
    # 这一步是从top_k*top_k个候选中挑出概率最大的top_k个 所以这个地方实际上还是一个3*max_length的维度
    outputs[:, :i] = outputs[row, :i]
    # 将选出来的序列后面接上概率最大的索引
    outputs[:, i] = ix[row, col]
    # 将概率值升维后进行返回
    log_scores = k_probs.unsqueeze(0)
    # 将得到的分数内容进行返回
    return outputs, log_scores

def beam_search(src, model, SRC, TRG):
    # 用模型过一遍<sos>的tgt，同时拿到encoder的编码信息，需要注意的是，这里的ourputs会更新
    outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG)
    eos_tok = TRG['<eos>']
    # 此处稍微有些冗余，可以考虑这是第二遍生成mask矩阵（一模一样的内容），可以进行优化
    src_mask = (src != SRC['<pad>']).unsqueeze(-2)
    ind = None
    # 这里的第二个参数是最大输出长度
    for i in range(2, 50):
        # 此处因为已经有<sos>以及<sos>结合e_outputs得到的第二个token索引，所以编码需要从2开始
        trg_mask = nopeak_mask(i).to(device)
        # 使用encoder编码的信息e_outputs结合当前out中的内容再预测一波
        out = model.out(model.decoder(outputs[:, :i], e_outputs, src_mask, trg_mask))
        # 将得到的结果进行softmax（Transformer论文原文的最后一步）
        out = F.softmax(out, dim=-1)
        # 这里面做的事情实际上就是每次对top_k的变量进行调整 outputs是一个(3,50)的结果向量，log_scores是
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, 3)
        # 这一步是为了能够找到输出矩阵中eos的位置坐标
        ones = (outputs == eos_tok).nonzero()
        # sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).to(device)
        # 如果outputs中有存在预测出的eos
        for vec in ones:
            # i在这里是第一个的坐标，实际上就是滴几句话的意思
            i = vec[0]
            # 这里判断的意思是，如果当前这句话的句子长度本身就是0
            if sentence_lengths[i] == 0:
                # 将第一个结束字符的位置给到length
                sentence_lengths[i] = vec[1]  # Position of first end symbol
        # 统计一下beam=3的情况下已经结束了的句子的个数
        num_finished_sentences = len([s for s in sentence_lengths if s > 0])
        # 如果这三个句子都已经完全结束了 通过beamsize个分数的分数来找一句最可能的
        if num_finished_sentences == 3:
            alpha = 0.7
            # 将每句话叠加概率进行计算，找到最可能的那句话
            div = 1 / (sentence_lengths.type_as(log_scores) ** alpha)
            # 找到概率最大的那个
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    # 首先将词典进行反转
    TRG_trans = {v: k for k, v in TRG.items()}
    # 如果没有找到合适的那个ind
    if ind is None:
        # 找到结尾的那个eos_tok
        length = (outputs[0] == eos_tok).nonzero()[0]
        # length = (outputs[0] == eos_tok).nonzero()
        # 因为第一个是开始的那个字符，所以需要跳过第一个
        return ' '.join([TRG_trans[tok] for tok in outputs[0][1:length]])
    else:
        length = (outputs[ind] == eos_tok).nonzero()[0]
        return ' '.join([TRG_trans[tok] for tok in outputs[ind][1:length]])


if __name__ == '__main__':
    tk = Spacy_tokenizer('en')
    input_sentence = "I do love you"
    tokenized_sentence = tk.tokenizer(input_sentence)
    # input_sentence = torch.randint(1, 9, (1, 10))
    english_vocab = torch.load('/home/fpn/projects/scp_transformer/dict/English_dict.fpn')
    french_vocab = torch.load('/home/fpn/projects/scp_transformer/dict/French_dict.fpn')
    input_sentence = [english_vocab[temp] for temp in tokenized_sentence]
    input_sentence = torch.tensor(input_sentence, dtype=torch.long).unsqueeze(0)
    model = Transformer(len(english_vocab), len(french_vocab), 512, 6, 8, 0.1, 50).to(device)
    model.load_state_dict(torch.load('/home/fpn/projects/scp_transformer/weights/28_statedict.pth', map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    out = beam_search(input_sentence.to(device), model, english_vocab, french_vocab)
    print(out)
    # outputs, e_outputs, log_scores = init_vars(input_sentence.to(device), model, english_vocab, french_vocab)
    # BeamDecoder = BeamSearchDecoder(model, french_vocab)
