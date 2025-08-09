import hydra
import logging
import os
import time

import torch
from torch import nn
from torch import tensor
import torch.optim as optim

from jiwer import wer
import editdistance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from dataset import TextDataset
from model import GPT
from conf.config import Config





def train_epoch(loader: torch.utils.data.DataLoader, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, device: str) -> tensor:
    model.train()
    losses = torch.zeros(len(loader))
    batch_times = []
    for i, (x, y) in enumerate(loader):
        start_time = time.time()
        optimizer.zero_grad()
        logits = model(x.to(device))
        loss = criterion(logits, y.to(device).view(-1,))
        losses[i] = loss.item()
        end_time = time.time()
        batch_time = end_time - start_time
        batch_times.append(batch_time)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Batch {i:4d}: Current Loss = {loss.item():.4f}")
    avg_loss = torch.mean(losses)
    avg_time = sum(batch_times) / len(batch_times)
    print(f"Average Loss: {avg_loss:.4f}, Average Batch Time: {avg_time:.4f}s")
    with open("logs.txt", "a") as f:
         f.write(str(losses.tolist()[:i+1]) + "\n")
    return avg_loss


@torch.inference_mode()
def test_epoch(loader, model, criterion, optimizer, device, idx_to_char):
    model.eval()

    losses = torch.zeros(len(loader))
    batch_times = []
    
    all_preds = []
    all_targets = []

    for i, (x, y) in enumerate(loader):
        start_time = time.time()
        optimizer.zero_grad()
        logits = model(x.to(device))
        loss = criterion(logits, y.to(device).view(-1,))
        losses[i] = loss.item()

        # Decode predictions and targets
        pred_ids = torch.argmax(logits, dim=-1)
        if pred_ids.ndim == 1:  # (B*T,) -> reshape to (B, T)
            T = y.size(1)
            pred_ids = pred_ids.view(y.size(0), T)

        target_ids = y  # already (B, T)

        pred_ids = pred_ids.cpu().numpy()
        target_ids = target_ids.cpu().numpy()

        for pred_seq, target_seq in zip(pred_ids, target_ids):
            pred_text = ''.join(idx_to_char[int(idx)] for idx in pred_seq)
            target_text = ''.join(idx_to_char[int(idx)] for idx in target_seq)
            all_preds.append(pred_text)
            all_targets.append(target_text)



        end_time = time.time()
        batch_time = end_time - start_time
        batch_times.append(batch_time)

    # Compute metrics
    cer_scores = [
        editdistance.eval(p, t) / len(t) if len(t) > 0 else 0
        for p, t in zip(all_preds, all_targets)
    ]
    avg_cer = sum(cer_scores) / len(cer_scores)

    wer_scores = [wer(t, p) for p, t in zip(all_preds, all_targets)]

    avg_wer = sum(wer_scores) / len(wer_scores)

    smoothie = SmoothingFunction().method1
    bleu_scores = [
        sentence_bleu([t.split()], p.split(), smoothing_function=smoothie)
        for p, t in zip(all_preds, all_targets)
    ]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    avg_loss = torch.mean(losses)
    avg_time = sum(batch_times[1:]) / len(batch_times[1:])

    print(f"Average Loss: {avg_loss:.4f}, CER: {avg_cer:.4f}, WER: {avg_wer:.4f}, BLEU: {avg_bleu:.4f}, Avg Batch Time: {avg_time:.4f}s")
    return avg_loss, avg_cer, avg_wer, avg_bleu


@hydra.main(config_name="config", version_base=None)
def main(cfg: Config):
    logging.basicConfig(level=logging.INFO)

    # >>> Setting configs, device, etc
    os.makedirs(cfg.data.checkpoints, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Device for training: {device}")

    # >>> Preparing the dataset
    with open(cfg.data.data_path, 'r', encoding='utf-8') as f:
        text = f.read()


    text = text[:len(text)]

    characters = sorted(list(set(text)))
    train_size = int(len(text) * cfg.data.train)

    train_text = text[:train_size]
    test_text = text[train_size:]
    
    train_set = TextDataset(train_text, characters, cfg.train.block_size, train=True)
    test_set = TextDataset(test_text, characters, cfg.train.block_size, train=False)
    
    print("Train size: ", len(train_set))
    print("Test size: ", len(test_set))

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.train.bsz, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg.train.bsz, shuffle=False
    )

    idx_to_char = {i: ch for i, ch in enumerate(characters)}

    # >>> Setting the model
    model = GPT(
        len(characters), 
        cfg.train.block_size, 
        cfg.train.n_embd, 
        cfg.train.n_head, 
        cfg.train.n_layer
    ).to(device)

    # loading checkpoint if provided
    if cfg.data.load_checkpoint:
        model.load_state_dict(torch.load(cfg.data.load_checkpoint_path))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    best_loss = 1e5
    for epoch in range(1, cfg.train.n_epoch+1):
        logging.info(f"Current epoch: {epoch}")
        
        mean_loss = train_epoch(
            train_dataloader, model, criterion, optimizer, device
        )
        logging.info(f"Epoch: {epoch}, train loss: {mean_loss:.2f}")
        
        mean_loss, cer, wer_score, bleu = test_epoch(
            test_dataloader, model, criterion, optimizer, device, idx_to_char
        )

        logging.info(f"Epoch: {epoch}, train loss: {mean_loss:.2f}")

        # save checkpoint
        if best_loss > mean_loss:
            torch.save(
                model.state_dict(), 
                cfg.data.checkpoints + "model_epoch_{}_loss_{:.2f}.pt".format(epoch, mean_loss)
            )
            best_loss = mean_loss

        

if __name__ == "__main__":
    main()
