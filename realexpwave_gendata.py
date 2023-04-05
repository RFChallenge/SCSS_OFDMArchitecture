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

# Case 2
sym_type = 'syncreal_bpskdfs'
sigtype1, sigtype2 = 1, 1

# # Case 3
# sym_type = 'syncreal_bpsk4pammixeddfs'
# sigtype1, sigtype2 = 1, 2

# # Case 4
# sym_type = 'syncreal_4pamdfs'
# sigtype1, sigtype2 = 2, 2

sig_len = 4096
n_examples = 100000

nfft = 64
sig_len = 4096 + 160
osfactor = 1
cos_waves = np.exp(1j*2*np.pi*osfactor*np.arange(nfft).reshape(-1,1)/nfft*(np.arange(sig_len).reshape(1,-1)))

n_sc = 28
def generate_sig(coeff=1, sigtype=1):
    if sigtype == 1:
        syms = coeff*(2*np.random.randint(2, size=(nfft, 1)) - 1)
    else:
        syms = coeff*(2*np.random.randint(4, size=(nfft,1)) - 3)/np.sqrt(5)
    syms[0,:] = 0
    syms[n_sc+1:,:] = 0
    syms[nfft//2+1:,:] = np.flipud(syms[1:nfft//2,:])
    sig_comp = syms * 1/np.sqrt(2*n_sc) * cos_waves
    sig = sig_comp.sum(axis=0)
    return sig, sig_comp, syms

def add_noise(sig, noise_pow=0.01):
    noise = np.sqrt(noise_pow)*np.random.randn(len(sig))
    return sig + noise

def add_interference(sig, interference_sigtype=1):
    interference, _, _ = generate_sig(coeff=4, sigtype=interference_sigtype)
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
    sig, sig_comp, syms = generate_sig(sigtype=sigtype1)
    sig_noisy = add_interference(sig, interference_sigtype=sigtype2)
    
    all_sig.append(sig[idx:idx+sig_len-160])
    all_sig_noisy.append(sig_noisy[idx:idx+sig_len-160])
    
window_len = sig_len - 160

all_sig = np.array(all_sig).real.reshape(-1, 1, window_len)
all_sig_noisy = np.array(all_sig_noisy).real.reshape(-1, 1, window_len)

import pickle
pickle.dump((all_sig_noisy, all_sig), open(f'{sym_type}_data.pickle','wb'), protocol=4)
