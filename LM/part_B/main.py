import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
import math
import os
import copy

from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
from functions import train_loop, eval_loop, init_weights
from model import LM_LSTM_VDO

def main():
    # --- Hyperparameters (Higher LR for ASGD) ---
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    TBS = 32
    LR = 3.0            # High LR for ASGD
    HID_DIM = 700
    EMB_SIZE = 700
    N_EPOCHS = 100
    EMB_DO = 0.4
    OUT_DO = 0.7
    CLIP = 5
    
    print(f"Running Part B on: {DEVICE}")

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
    model = LM_LSTM_VDO(EMB_SIZE, HID_DIM, vocab_len, pad_index=pad_index, 
                        emb_dropout=EMB_DO, hid_dropout=OUT_DO).to(DEVICE)
    model.apply(init_weights)

    # Requirement: ASGD
    optimizer = torch.optim.ASGD(model.parameters(), lr=LR, t0=0, lambd=0.0)

    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_index)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')

    # --- Training Loop with NT-ASGD Logic ---
    patience = 3
    best_ppl = math.inf
    best_model = None
    triggered = False  
    avg_params = {} 

    pbar = tqdm(range(1, N_EPOCHS + 1))
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, CLIP, DEVICE)
        ppl_dev, _ = eval_loop(dev_loader, criterion_eval, model, DEVICE)
        
        pbar.set_description("PPL: %f" % ppl_dev)

        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = 3
        else:
            patience -= 1

        # --- NT-ASGD Trigger ---
        if patience <= 0 and not triggered:
            print(f"\nEpoch {epoch}: >>> Triggering NT-ASGD averaging")
            triggered = True
            for group in optimizer.param_groups:
                for p in group['params']:
                    avg_params[id(p)] = p.detach().cpu().clone()
                group['lr'] *= 0.33 # Reduce LR
            patience = 3 # Reset patience

        if patience <= 0 and triggered:
            print(f"\nEpoch {epoch}: Early stopping after NT-ASGD")
            break

    # Load averaged weights
    if triggered:
        print(">>> Loading NT-ASGD averaged weights")
        for group in optimizer.param_groups:
            for p in group['params']:
                key = id(p)
                if key in avg_params:
                    p.data.copy_(avg_params[key].to(DEVICE))
        best_model = copy.deepcopy(model).to('cpu')

    best_model.to(DEVICE)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model, DEVICE)
    print('\nPart B Test PPL: ', final_ppl)
    torch.save(best_model.state_dict(), 'bin/best_model_partB.pt')

if __name__ == "__main__":
    main()