import os, sys

from tqdm import tqdm
import numpy as np
import random

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from pytorch_lightning import Trainer, seed_everything
from asteroid.engine import System
import asteroid
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


np.random.seed(42)
random.seed(42)
seed_everything(42, workers=True)
sym_type = 'syncreal_4pamdfs'

sig_len = 4096
n_examples = 100000

np.random.seed(0)
random.seed(0)

import pickle

all_sig_noisy, all_sig = pickle.load(open(os.path.join('dataset', f'{sym_type}_data.pickle'),'rb'))
window_len = sig_len

n_train = int(len(all_sig)*0.9)

tensor_x = torch.Tensor(all_sig_noisy[:n_train])
tensor_y = torch.Tensor(all_sig[:n_train])

train_dataset = TensorDataset(tensor_x,tensor_y)
train_loader = DataLoader(train_dataset, batch_size=16, num_workers=40)

tensor_val_x = torch.Tensor(all_sig_noisy[n_train:])
tensor_val_y = torch.Tensor(all_sig[n_train:])

val_dataset = TensorDataset(tensor_val_x,tensor_val_y)
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=40)


from waveunet import Waveunet
from asteroid.models import SuDORMRFNet, SuDORMRFImprovedNet, DPTNet, DPRNNTasNet, ConvTasNet

all_k_sz = [15, 21, 31, 51, 63, 65, 71, 81, 91, 101, 151, 201]
def train_script(idx):
    k_sz = all_k_sz[idx]
    model = Waveunet(n_src=1, long_kernel_size=k_sz, n_first_filter=20, depth=5)
    model.cuda()
    model_name = f'waveunet_20filters_5depth_ksz{k_sz}'        
    
    print(sym_type, model_name)
    loss = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
    system = System(model, optimizer, loss, train_loader, val_loader)
    trainer = Trainer(max_epochs=2000, gpus=1, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=100), ModelCheckpoint(dirpath=os.path.join('models', f"{sym_type}_models_ksizes", model_name, f"sinsep_{window_len}"), monitor='val_loss', mode='min')])
    
    trainer.fit(system)
    
if __name__ == '__main__':
    train_script(int(sys.argv[1]))
