import hydra
import logging
import os

import torch
from torch import nn
from torch import Tensor, tensor
import torch.optim as optim
import torchquantum as tq

from dataset import TextDataset
from model import GPT
from conf.config import Config



import time
def train_epoch(
        loader: torch.utils.data.DataLoader, 
        model: nn.Module, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer,
        device: str
    ) -> tensor:

    losses = torch.zeros(len(loader))
    batch_times = []  # To store the time for each batch

    for i, (x, y) in enumerate(loader):
        start_time = time.time()  # Start timing the batch

        optimizer.zero_grad()
        logits = model(x.to(device))
        loss = criterion(logits, y.to(device).view(-1,))
        losses[i] = loss.item()
        
        # End timing the batch
        end_time = time.time()
        batch_time = end_time - start_time
        batch_times.append(batch_time)

        loss.backward()
        optimizer.step()

        print(f"Batch {i:4d}: Loss = {loss.item():.4f}, Time = {batch_time:.4f}s ", losses.tolist()[:i+1])
        

    avg_loss = torch.mean(losses)
    avg_time = sum(batch_times) / len(batch_times)

    print(f"Average Loss: {avg_loss:.4f}, Average Batch Time: {avg_time:.4f}s")
    return avg_loss



@torch.inference_mode()
def test_epoch(
        loader: torch.utils.data.DataLoader, 
        model: nn.Module, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer,
        device: str
    ) -> tensor:

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
        print(f"Batch {i:4d}: Loss = {loss.item():.4f}, Time = {batch_time:.4f}s ", losses.tolist()[:i+1])


    info = torch.mean(losses)
    avg_loss = torch.mean(losses)
    avg_time = sum(batch_times) / len(batch_times)
    print(f"Average Loss: {avg_loss:.4f}, Average Batch Time: {avg_time:.4f}s")
    return info


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
        
        # testing
        mean_loss = test_epoch(
            test_dataloader, model, criterion, optimizer, device
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
