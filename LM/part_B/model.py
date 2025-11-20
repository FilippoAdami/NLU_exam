import torch.nn as nn

## --- VARIATIONAL DROPOUT IMPLEMENTATION ---

class VariationalDropout(nn.Module):
    """
    This technique applies the SAME dropout mask to the input 
    across all time steps for a given forward pass. 
    This is essential for robust regularization in recurrent networks.
    """
    def __init__(self):
        super(VariationalDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        # 1. Skip if not training or dropout is 0
        if not self.training or dropout == 0:
            return x
            
        # 2. Create the mask: 
        #    - Size (1, x.size(1), x.size(2)) ensures the mask is broadcast 
        #      across the time dimension (x.size(0) is Sequence Length, if batch_first=False, but here Batch size).
        #    - bernoulli_ creates a binary mask where (1 - dropout) probability = 1 (kept).
        mask = x.new_empty(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        
        # 3. Rescale: Divide the mask by (1 - dropout) to maintain the expected output magnitude
        mask = mask.div_(1 - dropout)
        
        # 4. Expand: Broadcast the single mask across the batch dimension (x.size(0))
        mask = mask.expand_as(x)
        
        # 5. Apply the mask
        return x * mask

## --- LSTM with Variational Dropout and Weight Tying ---

class LSTM_VDO(nn.Module):
    """
    Language Model combining LSTM with Weight Tying and Variational Dropout.
    This architecture is designed for high performance and regularization.
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index,
                 emb_dropout=0.1, hid_dropout=0.1, n_layers=1):
        super().__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # 2. Dropout Instance (LockedDropout is instantiated once)
        self.lockdrop = VariationalDropout()
        self.emb_dropout = emb_dropout # Dropout rate for embeddings
        self.hid_dropout = hid_dropout # Dropout rate for hidden states

        # 3. LSTM Layer
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers,
                           batch_first=True, bidirectional=False)

        # 4. Output Linear Layer (bias=False is common when implementing Weight Tying)
        self.output = nn.Linear(hidden_size, output_size, bias=False)
        
        # 5. Requirement: Weight Tying
        self.output.weight = self.embedding.weight 

    def forward(self, x):
        # 1. Embeddings
        emb = self.embedding(x)
        
        # 2. Variational Dropout on Embeddings
        emb = self.lockdrop(emb, self.emb_dropout) 
        
        rnn_out, _ = self.rnn(emb)
        
        # 4. Variational Dropout on Hidden States
        rnn_out = self.lockdrop(rnn_out, self.hid_dropout)
        
        # 5. Linear Projection and Permute
        out = self.output(rnn_out).permute(0, 2, 1)
        return out