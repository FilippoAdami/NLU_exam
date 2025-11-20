import torch
import torch.nn as nn
import math

## --- WEIGHT INITIALIZATION ---

def init_weights(mat):
    """
    Initializes the weights and biases of the neural network.
    """
    for m in mat.modules():
        if type(m) in [nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                # Initialize Input-Hidden weights ('weight_ih') using Xavier Uniform distribution
                if 'weight_ih' in name:
                    for idx in range(4): # LSTM weights are split into 4 parts (gates)
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                # Initialize Hidden-Hidden weights ('weight_hh') using Orthogonal initialization
                # Orthogonal initialization helps maintain gradient magnitude over time steps.
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                # Initialize biases to zero
                elif 'bias' in name:
                    param.data.fill_(0)
        # Check for Linear Layers
        else:
            if type(m) in [nn.Linear]:
                # Initialize Linear weights using a uniform distribution (small values)
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                # Initialize biases to a small positive value
                if m.bias != None:
                    m.bias.data.fill_(0.01)

## --- TRAINING LOOP ---

def train_loop(data, optimizer, criterion, model, clip, device):
    """
    Performs one epoch of training over the DataLoader.

    :param data: DataLoader iterable containing batches.
    :param optimizer: The optimization strategy (e.g., SGD, ASGD).
    :param criterion: The loss function (e.g., CrossEntropyLoss).
    :param model: The PyTorch model instance.
    :param clip: Gradient clipping threshold (to prevent exploding gradients).
    :param device: The target device ('cuda' or 'cpu').
    :return: The average loss normalized by the total number of non-padding tokens.
    """
    model.train() # Set the model to training mode (enables dropout/batchnorm)
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad() # Clear previous gradients
        
        # Move batch data to the target device (GPU)
        source = sample['source'].to(device)
        target = sample['target'].to(device)
        
        output = model(source) # Forward pass
        loss = criterion(output, target) # Compute loss
        
        # Store loss weighted by the actual number of tokens (to calculate normalized average later)
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        
        loss.backward() # Compute gradients
        
        # Gradient Clipping: Clamps the gradient norms to a maximum value
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step() # Update model parameters
        
    return sum(loss_array)/sum(number_of_tokens) # Return average loss per token

## --- EVALUATION LOOP ---

def eval_loop(data, eval_criterion, model, device):
    """
    Evaluates the model on a dataset (Validation or Test set).

    :param data: DataLoader iterable.
    :param eval_criterion: Loss function (using reduction='sum' for easy normalization).
    :param model: The PyTorch model instance.
    :param device: The target device ('cuda' or 'cpu').
    :return: Perplexity (PPL) and the total average loss per token.
    """
    model.eval() # Set the model to evaluation mode (disables dropout/batchnorm)
    loss_array = []
    number_of_tokens = []
    
    # Disable gradient tracking for efficiency and to save memory
    with torch.no_grad():
        for sample in data:
            # Move batch data to the target device
            source = sample['source'].to(device)
            target = sample['target'].to(device)
            
            output = model(source)
            loss = eval_criterion(output, target)
            
            # Store raw loss item and token count
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    # Perplexity (PPL) is calculated as e raised to the power of the normalized Cross-Entropy Loss (NLL)
    normalized_loss = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(normalized_loss)
    
    return ppl, normalized_loss