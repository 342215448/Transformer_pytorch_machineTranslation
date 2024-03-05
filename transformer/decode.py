import torch
import torch.nn.functional as F
from fpn_transformer import Transformer
from fpn_process import nopeak_mask
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    # 创建一个(top_k*max_length)维度的0矩阵用来作为输出
    outputs = torch.zeros(3, 50, device=device).long()
    # 将top_k中的每一个的第一个放置为起始字符的索引
    outputs[:, 0] = init_tok
    # 第二个放置为概率最大的字符的索引
    outputs[:, 1] = ix[0]
    # 创建一个top_k * src_length * d_model维度的全零矩阵
    e_outputs = torch.zeros(3, e_output.size(-2), e_output.size(-1), device=device)
    e_outputs[:, :] = e_output[0]

    return outputs, e_outputs, log_scores


class BeamSearchDecoder:
    def __init__(self, model, vocab, beam_size=3, max_seq_len=50):
        """
        初始化Beam Search解码器。
        :param model: Transformer模型，用于生成预测。
        :param vocab: 词汇表，用于ID和词汇的转换。
        :param beam_size: Beam宽度。
        :param max_seq_len: 生成序列的最大长度。
        """
        self.model = model
        self.vocab = vocab
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len

    def decode(self, encoder_outputs, src_mask):
        """
        使用Beam Search解码给定的encoder输出。
        :param encoder_outputs: Transformer编码器的输出。
        :param src_mask: 源序列的掩码。
        :return: 最佳翻译序列及其分数。
        """
        # 初始化beam_size个活跃候选项
        init_beam = [([self.vocab.start_token_id], 0.0)]  # 列表中的元素是(序列, 分数)
        # 确保模型处于评估模式
        self.model.eval()

        # 使用beam search进行解码
        current_beams = init_beam
        for _ in range(self.max_seq_len):
            next_beams = []
            for seq, score in current_beams:
                # 检查序列是否已经结束
                if seq[-1] == self.vocab.end_token_id:
                    next_beams.append((seq, score))
                    continue

                # 准备输入进行下一次预测
                seq_tensor = torch.tensor([seq], dtype=torch.long)
                logits = self.model(decoder_input=seq_tensor, encoder_outputs=encoder_outputs, src_mask=src_mask)
                log_probs = F.log_softmax(logits[:, -1], dim=-1)  # 取最后一个时间步的输出

                # 选择top-k个最可能的下一个词
                topk_log_probs, topk_ids = torch.topk(log_probs, self.beam_size, dim=-1)

                # 将新候选项加入到next_beams中
                for k in range(self.beam_size):
                    next_seq = seq + [topk_ids[0, k].item()]
                    next_score = score + topk_log_probs[0, k].item()
                    next_beams.append((next_seq, next_score))

            # 从next_beams中选出得分最高的beam_size个序列
            current_beams = sorted(next_beams, key=lambda x: x[1], reverse=True)[:self.beam_size]

        # 选择得分最高的序列
        best_seq, best_score = max(current_beams, key=lambda x: x[1])
        return self.vocab.convert_ids_to_tokens(best_seq), best_score


if __name__ == '__main__':
    input_sentence = torch.randint(1, 9, (1, 10))
    english_vocab = torch.load('dict/English_dict.fpn')
    french_vocab = torch.load('dict/French_dict.fpn')
    model = Transformer(len(english_vocab), len(french_vocab), 512, 6, 8, 0.1, 50).to(device)
    model.load_state_dict(torch.load('weights/21_statedict.pth'))
    model.to(device)
    model.eval()
    outputs, e_outputs, log_scores = init_vars(input_sentence.to(device), model, english_vocab, french_vocab)
    # BeamDecoder = BeamSearchDecoder(model, french_vocab)