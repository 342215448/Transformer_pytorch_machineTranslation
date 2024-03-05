import torch
from tqdm import tqdm
from transformer.transformer import Transformer
from transformer.process import create_dataloader, create_masks, create_folder_if_not_exists
import torch.nn.functional as F
import torch.optim as optim
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(src_lang, tgt_lang, src_path, tgt_path, batch_size, learning_rate, epoch_nums, weights_path):
    dataloader, src_vocab_size, tgt_vocab_size, max_seq_len = create_dataloader(
        src_lang,
        tgt_lang,
        src_path,
        tgt_path,
        batch_size
    )
    print('model initialization...')
    model = Transformer(src_vocab_size, tgt_vocab_size, 512, 6, 8, 0.1, max_seq_len).to(device)
    total_params = sum(
        p.numel() for p in tqdm(model.parameters(), desc='caculating trainable parameters...') if p.requires_grad)
    print(f'###### total trainable parameter: {total_params}({(total_params / 1000000000):.3f}B) ######')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.98, 0.9))
    for epoch_cur in range(epoch_nums):
        epoch_loss = 0
        step = 0
        for batch in tqdm(dataloader, desc=f'[{epoch_cur + 1}/{epoch_nums}] training...'):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_mask, tgt_mask = create_masks(src, tgt)
            src_mask.to(device)
            tgt_mask.to(device)
            pred = model(src, tgt, src_mask, tgt_mask)
            reshape_tgt = tgt.view(-1)
            reshape_pred = pred.view(len(reshape_tgt), -1)
            loss = F.cross_entropy(reshape_pred, reshape_tgt, ignore_index=0)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
        epoch_loss /= step
        print(f'[{epoch_cur + 1}/{epoch_nums}], train loss: {epoch_loss}, saving {epoch_cur + 1} state dict...')
        create_folder_if_not_exists(weights_path)
        torch.save(model.state_dict(), os.path.join(weights_path, f'{epoch_cur + 1}_statedict.pth'))


def for_test():
    # 'data/en2zh/old/version1/english.txt'
    train('en',
          'fr',
          'data/test-en2fr/english.txt',
          'data/test-en2fr/french.txt',
          64, 1e-6, 100, 'weights')
    # model = Transformer(10000, 10000, 1024, 5, 8, 0.1, 200).to(device)


if __name__ == '__main__':
    for_test()
    # test()
