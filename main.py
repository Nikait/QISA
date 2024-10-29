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



def train_epoch(
        loader: torch.utils.data.DataLoader, 
        model: nn.Module, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer,
        device: str
    ) -> tensor:

    losses = torch.zeros(len(loader))

    for i, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        x.to(device); y.to(device)
        logits = model(x)
        loss = criterion(logits, y.view(-1,))
        losses[i] = loss.item()
        print(i, loss.item())
        loss.backward()
        optimizer.step()

    info = torch.mean(losses)

    return info


@torch.inference_mode()
def test_epoch(
        loader: torch.utils.data.DataLoader, 
        model: nn.Module, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer,
        device: str
    ) -> tensor:

    losses = torch.zeros(len(loader))

    for i, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        x.to(device); y.to(device)
        logits = model(x)
        loss = criterion(logits, y.view(-1,))
        losses[i] = loss.item()

    info = torch.mean(losses)

    return info



example = Tensor(
       [[[ 0.7044,  0.2483,  0.1535,  0.5878],
         [-0.1411, -0.1529,  0.1122, -0.3319],
         [-0.6108,  0.4578, -0.0164,  0.2785],
         [-0.0856,  0.0203, -0.4373, -0.2202],
         [-0.3578, -0.3051,  0.2508, -0.9166],
         [-0.2245, -0.2521,  0.7277, -0.0840],
         [-0.4278,  0.4434, -1.1972,  0.2474],
         [ 0.5434, -0.5020, -0.5242,  0.0658],
         [ 0.4942, -0.3882, -0.3520,  0.2547],
         [ 0.2411, -0.2406,  0.3084, -0.8022],
         [ 0.1785,  0.3281, -0.2977,  0.2702],
         [-0.1735, -0.3803, -0.1284, -0.4475],
         [ 0.3911,  0.0125,  0.0000,  0.0000]],

        [[ 0.4633,  0.1374,  0.3429,  0.1954],
         [-0.3175, -0.3780, -0.1383, -0.0918],
         [-0.6496,  0.2089, -0.1501, -0.1357],
         [ 0.0460,  0.1829,  0.0710, -0.2389],
         [-0.1784, -0.0932,  0.0784, -0.4668],
         [-0.0905, -0.4281,  0.3322,  0.1938],
         [-0.2218, -0.0911, -0.5818,  0.1080],
         [ 0.4531, -0.2249, -0.2654, -0.0616],
         [ 0.4836, -0.4430, -0.1051,  0.3080],
         [ 0.4164,  0.2105,  0.2831, -0.5753],
         [ 0.0246,  0.0785, -0.4280,  0.1627],
         [-0.0178, -0.2567, -0.2761, -0.4372],
         [-0.0421,  0.2470,  0.0000,  0.0000]],

        [[ 0.7998, -0.2493,  0.1360,  0.5713],
         [-0.0634, -0.2167,  0.0054, -0.0297],
         [-0.2917,  0.2224,  0.2153,  0.0133],
         [-0.0867,  0.1790, -0.1625, -0.4744],
         [-0.1247,  0.1510, -0.0651, -0.7477],
         [-0.0834, -0.1654,  0.3244,  0.3841],
         [-0.2189, -0.1031, -0.5856,  0.1294],
         [ 0.3526, -0.2429, -0.6476,  0.2244],
         [ 0.1442, -0.4297, -0.2274, -0.1343],
         [ 0.2589,  0.1034,  0.1524, -0.6588],
         [ 0.1386, -0.3446, -0.2303, -0.0735],
         [ 0.1622, -0.7334, -0.0785, -0.0039],
         [ 0.2207,  0.0403,  0.0000,  0.0000]],

        [[ 0.9094,  0.2046, -0.0663,  0.3822],
         [ 0.0201, -0.2234, -0.0314, -0.2823],
         [-0.3307,  0.3111,  0.0711,  0.0167],
         [-0.2645,  0.5164,  0.0024, -0.1014],
         [ 0.1373,  0.3681,  0.2459, -0.3409],
         [ 0.1182, -0.4725,  0.3823,  0.2341],
         [-0.0828,  0.0535, -0.7295,  0.2064],
         [-0.0200, -0.3469, -0.3892,  0.0912],
         [ 0.0057,  0.0347,  0.1169,  0.2641],
         [ 0.1341, -0.0740,  0.1009, -0.1946],
         [ 0.0204, -0.0302, -0.4706,  0.0564],
         [-0.0875, -0.1816, -0.3542, -0.5282],
         [-0.3329,  0.0554,  0.0000,  0.0000]],

        [[ 0.3994, -0.0411, -0.0016,  0.3766],
         [-0.1749, -0.1379, -0.0397, -0.0656],
         [-0.4586,  0.4183, -0.0014, -0.0015],
         [-0.1754,  0.2595, -0.0913, -0.2740],
         [-0.2588,  0.1391,  0.1093, -0.7207],
         [ 0.1923, -0.3494,  0.5314,  0.1067],
         [-0.2310,  0.0810, -0.6376,  0.2211],
         [ 0.5071, -0.3650, -0.0911,  0.1653],
         [ 0.0060, -0.2051, -0.3741, -0.0349],
         [ 0.2161,  0.0546,  0.0699, -0.4035],
         [-0.0115,  0.2116, -0.2233,  0.2550],
         [-0.3008, -0.3423, -0.4521, -0.3139],
         [ 0.1494,  0.1762,  0.0000,  0.0000]]]
    )

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
    
    train_set = TextDataset(train_text, characters, cfg.data.block_size, train=True)
    test_set = TextDataset(test_text, characters, cfg.data.block_size, train=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.train.bsz, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg.train.bsz, shuffle=False
    )


    # >>> Setting the model
    model = GPT(
        len(characters), 
        cfg.data.block_size, 
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
