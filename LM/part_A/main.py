import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
import math
import os
import copy

from utils import read_file, Lang, PennTreeBank, collate_fn
from functions import train_loop, eval_loop, init_weights
from model import LM_LSTM_Dropout

def main():
    # --- Hyperparameters (Optimized for AdamW) ---
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    TBS = 32
    LR = 0.0004           
    HID_DIM = 800
    EMB_SIZE = 600
    N_EPOCHS = 100
    EMB_DO = 0.3         
    OUT_DO = 0.65
    CLIP = 5
    
    print(f"Running Part A on: {DEVICE}")

    # --- Data Loading ---
    base_path = "dataset/PennTreeBank/"
    train_raw = read_file(os.path.join(base_path, "ptb.train.txt"))
    dev_raw = read_file(os.path.join(base_path, "ptb.valid.txt"))
    test_raw = read_file(os.path.join(base_path, "ptb.test.txt"))

    lang = Lang(train_raw, ["<pad>", "<eos>"])
    pad_index = lang.word2id["<pad>"]
    vocab_len = len(lang.word2id)

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=TBS, 
                              collate_fn=partial(collate_fn, pad_token=pad_index), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, 
                            collate_fn=partial(collate_fn, pad_token=pad_index))
    test_loader = DataLoader(test_dataset, batch_size=128, 
                             collate_fn=partial(collate_fn, pad_token=pad_index))

    # --- Model & Optimizer ---
    model = LM_LSTM_Dropout(EMB_SIZE, HID_DIM, vocab_len, pad_index=pad_index, 
                            emb_dropout=EMB_DO, hid_dropout=OUT_DO).to(DEVICE)
    model.apply(init_weights)

    # Requirement 3: Replace SGD with AdamW
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_index)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')

    # --- Training Loop ---
    patience = 5
    best_ppl = math.inf
    best_model = None

    pbar = tqdm(range(1, N_EPOCHS + 1))
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, CLIP, DEVICE)
        ppl_dev, _ = eval_loop(dev_loader, criterion_eval, model, DEVICE)
        
        pbar.set_description("PPL: %f" % ppl_dev)

        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = 5
        else:
            patience -= 1
            
        if patience <= 0:
            print("Early stopping")
            break

    best_model.to(DEVICE)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model, DEVICE)
    print('\nPart A Test PPL: ', final_ppl)
    torch.save(best_model.state_dict(), 'bin/best_model_partA.pt')

if __name__ == "__main__":
    main()