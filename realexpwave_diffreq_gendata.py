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
sym_type = 'syncreal_diffreq'

n_sc = 28
nfft = 64
osfactor = 1
sig_len = 4096 + 160
cos_idx = np.arange(n_sc) + 1
np.random.shuffle(cos_idx)

cos_waves1 = np.exp(1j*2*np.pi*osfactor*cos_idx[:n_sc//2].reshape(-1,1)/nfft*(np.arange(sig_len).reshape(1,-1)))
cos_waves1C = np.exp(1j*2*np.pi*osfactor*(-cos_idx[:n_sc//2]).reshape(-1,1)/nfft*(np.arange(sig_len).reshape(1,-1)))

cos_waves2 = np.exp(1j*2*np.pi*osfactor*cos_idx[n_sc//2:].reshape(-1,1)/nfft*(np.arange(sig_len).reshape(1,-1)))
cos_waves2C = np.exp(1j*2*np.pi*osfactor*(-cos_idx[n_sc//2:]).reshape(-1,1)/nfft*(np.arange(sig_len).reshape(1,-1)))

n_examples = 100000

def generate_sig(coeff=1, sigtype=1):
    cos_waves0 = cos_waves1 if sigtype==1 else cos_waves2
    cos_waves0C = cos_waves1C if sigtype==1 else cos_waves2C
    syms = coeff*(np.random.randn(cos_waves0.shape[0], 1))
    
    sig_comp = np.vstack((syms * cos_waves0, syms * cos_waves0C)) * 1/np.sqrt(n_sc)
    sig = sig_comp.sum(axis=0)
    return sig, sig_comp, syms

def add_noise(sig, noise_pow=0.01):
    noise = np.sqrt(noise_pow)*np.random.randn(len(sig))
    return sig + noise

def add_interference(sig):
    interference, _, _ = generate_sig(coeff=4, sigtype=2)
    return sig+interference

def reconstruct_sig(syms):
    sig_comp = syms * np.sqrt(2)/np.sqrt(n_sc) * cos_waves
    sig = sig_comp.sum(axis=0)
    return sig, sig_comp

np.random.seed(0)
random.seed(0)
all_sig, all_sig_noisy = [], [] 
for _ in tqdm(range(n_examples)):
    idx = 0
    sig, sig_comp, syms = generate_sig()
    sig_noisy = add_interference(sig)
    all_sig.append(sig[idx:idx+sig_len-160])
    all_sig_noisy.append(sig_noisy[idx:idx+sig_len-160])
    
window_len = sig_len - 160

all_sig = np.array(all_sig).reshape(-1, 1, window_len)
all_sig_noisy = np.array(all_sig_noisy).reshape(-1, 1, window_len)

import pickle
pickle.dump((all_sig_noisy, all_sig), open(f'{sym_type}_data.pickle','wb'), protocol=4)
