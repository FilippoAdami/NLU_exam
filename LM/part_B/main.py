import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
import math
import os
import copy

# Import utilities and model components
from utils import read_file, Lang, PennTreeBank, collate_fn
from functions import train_loop, eval_loop, init_weights
from model import LSTM_VDO # Model includes Weight Tying and Variational Dropout

def main():
    ## --- HYPERPARAMETERS (Part 1.B Configuration: VDO + WT + NT-ASGD) ---
    
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Training Batch Size
    TBS = 32
    
    # Learning Rate: High for the NT-AvSGD optimizer
    LR = 3.0
    
    # Hidden and Embedding Dimensions (same dimension because of Weight Tying)
    HID_DIM = 900
    EMB_SIZE = 900
    
    # Maximum number of training epochs
    N_EPOCHS = 100
    
    # Dropout rate applied after the embedding layer (input regularization)
    EMB_DO = 0.7        
    
    # Dropout rate applied before the final linear layer (regularizes hidden state output)
    OUT_DO = 0.7
    
    # Gradient Clipping threshold (prevents Exploding Gradients)
    CLIP = 5
    
    print(f"Running Part B on: {DEVICE}")

    ## --- DATA LOADING AND PREPROCESSING ---
    
    base_path = "dataset/PennTreeBank/"
    # Load raw text data
    train_raw = read_file(os.path.join(base_path, "ptb.train.txt"))
    dev_raw = read_file(os.path.join(base_path, "ptb.valid.txt"))
    test_raw = read_file(os.path.join(base_path, "ptb.test.txt"))

    # Initialize Lang class to build the word-to-ID vocabulary
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    pad_index = lang.word2id["<pad>"]
    vocab_len = len(lang.word2id)

    # Create PyTorch Dataset instances
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Create DataLoader instances with custom collate_fn for batching and padding
    train_loader = DataLoader(train_dataset, batch_size=TBS, 
                              collate_fn=partial(collate_fn, pad_token=pad_index), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, 
                            collate_fn=partial(collate_fn, pad_token=pad_index))
    test_loader = DataLoader(test_dataset, batch_size=128, 
                             collate_fn=partial(collate_fn, pad_token=pad_index))

    ## --- MODEL INITIALIZATION ---
    
    # Instantiate LSTM_VDO (includes Weight Tying and Variational Dropout)
    model = LSTM_VDO(EMB_SIZE, HID_DIM, vocab_len, pad_index=pad_index, 
                        emb_dropout=EMB_DO, hid_dropout=OUT_DO).to(DEVICE)
    model.apply(init_weights)

    # SDG Optimizer
    #optimizer = optim.SGD(model.parameters(), lr=LR)

    # AvSGD Optimizer
    optimizer = torch.optim.ASGD(model.parameters(), lr=LR, t0=0, lambd=0.0)

    # Loss Functions
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_index) # Used for backpropagation
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum') # Used for PPL calculation

    ## --- MAIN TRAINING LOOP with NT-ASGD LOGIC ---
    
    # Early Stopping parameters: Patience is the number of epochs to wait for improvement
    patience = 3
    best_ppl = math.inf # Initialize best Perplexity to infinity
    best_model = None
    triggered = False  # Flag to indicate if ASGD averaging has started
    avg_params = {}    # Dictionary to store the running average of parameters (for NT-ASGD)

    pbar = tqdm(range(1, N_EPOCHS + 1)) 
    for epoch in pbar:
        # 1. Training Step: Performs forward/backward pass and updates weights
        loss = train_loop(train_loader, optimizer, criterion_train, model, CLIP, DEVICE)
        
        # 2. Validation Step: Evaluate model performance on the dev set
        ppl_dev, _ = eval_loop(dev_loader, criterion_eval, model, DEVICE)
        
        # Update progress bar description with current PPL
        pbar.set_description("PPL: %f" % ppl_dev)

        # 3. Early Stopping Check
        if ppl_dev < best_ppl:
            # Improvement found: save model copy and reset patience
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu') # Save model state on CPU to save GPU memory
            patience = 3
        else:
            # No improvement: decrease patience counter
            patience -= 1

        # --- NON-MONOTONICALLY TRIGGERED ASGD (NT-ASGD) ---
        if patience <= 0 and not triggered:
            # Patience ran out while in the standard SGD phase -> Trigger Averaging
            print(f"\nEpoch {epoch}: >>> Triggering NT-ASGD averaging")
            triggered = True
            for group in optimizer.param_groups:
                for p in group['params']:
                    # Start storing the current weights as the initial averaged parameters
                    avg_params[id(p)] = p.detach().cpu().clone()
                # Reduce Learning Rate drastically (standard practice for ASGD stabilization)
                group['lr'] *= 0.33 
            patience = 3 # Reset patience to allow the ASGD phase to run

        # --- Early Stopping after NT-ASGD is active ---
        if patience <= 0 and triggered:
            # Patience ran out while in the ASGD phase -> Stop training
            print(f"\nEpoch {epoch}: Early stopping after NT-ASGD")
            break

    ## --- FINAL MODEL LOAD AND EVALUATION ---
    
    # Load averaged weights if the NT-ASGD phase was triggered
    if triggered:
        print(">>> Loading NT-ASGD averaged weights")
        for group in optimizer.param_groups:
            for p in group['params']:
                key = id(p)
                if key in avg_params:
                    # Replace current model parameters with the stored average
                    p.data.copy_(avg_params[key].to(DEVICE))
        # Update best_model to ensure it holds the final averaged state
        best_model = copy.deepcopy(model).to('cpu')

    # Move the best performing model back to the device for final test
    best_model.to(DEVICE)
    # Evaluate final PPL on the test set
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model, DEVICE)
    print('\nPart A Test PPL: ', final_ppl)
    
    # torch.save(best_model.state_dict(), 'bin/best_model_partB.pt') # Saving is commented out

if __name__ == "__main__":
    main()