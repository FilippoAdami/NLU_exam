import torch.nn as nn

## --- 1. Vanilla RNN ---

class RNN(nn.Module):
    """
    Simple Language Model using a standard PyTorch RNN layer.
    This serves as a common baseline for sequence modeling tasks.
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(RNN, self).__init__()
        # nn.Embedding: Converts token IDs (integers) into dense vectors (emb_size).
        # output_size is the vocabulary size. pad_index=0 ensures padding tokens are ignored.
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # nn.RNN: The core recurrent layer.
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, batch_first=True)
        
        # nn.Linear: Projects the hidden state (hidden_size) back to the vocabulary space (output_size).
        # The output represents the logits for the next predicted token.
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        # 1. Embedding: Convert input token IDs to vectors
        emb = self.embedding(input_sequence)
        
        # 2. RNN Pass: rnn_out contains the hidden states for every time step; _ is the final hidden state (h_n)
        rnn_out, _  = self.rnn(emb)
        
        # 3. Linear Projection: Apply the final linear layer
        output = self.output(rnn_out)
        
        # Permute: Change output shape from (Batch, Seq_Len, Vocab_Size) to (Batch, Vocab_Size, Seq_Len)
        # This is required by PyTorch's nn.CrossEntropyLoss for sequence classification tasks.
        output = output.permute(0,2,1)
        return output

## --- 2. LSTM standard, no Dropout ---
class LSTM(nn.Module):
    """
    Language Model using a standard PyTorch LSTM network with no dropout.
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LSTM, self).__init__()
        
        # 1. Embedding Layer: Maps token IDs to dense vectors.
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # 2. LSTM Core: The recurrent layer.
        #    bidirectional=False and no dropout arguments are passed (standard LSTM).
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        
        # 3. Output Layer: Projects hidden state back to the vocabulary size.
        self.output = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, input_sequence):
        # 1. Embedding Pass
        emb = self.embedding(input_sequence)
        
        # 2. LSTM Pass: rnn_out contains the hidden states for all time steps.
        rnn_out, _ = self.rnn(emb)
        
        # 3. Linear Projection
        output = self.output(rnn_out)
        
        # 4. Permute: Required format for PyTorch's nn.CrossEntropyLoss
        #    (Batch, Seq_Len, Vocab_Size) -> (Batch, Vocab_Size, Seq_Len)
        output = output.permute(0, 2, 1) 
        return output

## --- 3. LSTM with Standard Dropout ---

class LSTM_Dropout(nn.Module):
    """
    Language Model using LSTM for improved sequence modeling, incorporating
    standard Dropout layers for regularization (a Part 1.A mandatory step).
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0,
                 emb_dropout=0.1, hid_dropout=0.1, n_layers=1):
        super(LSTM_Dropout, self).__init__()
        
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # 1. Embedding Dropout: Standard Dropout applied immediately after the embedding layer.
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        
        # 2. Output Dropout: Applied to the hidden states (rnn_out) before the final linear layer.
        self.out_dropout = nn.Dropout(hid_dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        # 1. Embedding
        emb = self.embedding(input_sequence)
        
        # 2. Apply Dropout to Embeddings
        emb = self.emb_dropout(emb)
        
        # 3. LSTM Pass: Returns (output, (h_n, c_n)). 'rnn_out' is the output hidden state H for all steps.
        rnn_out, _ = self.rnn(emb)
        
        # 4. Apply Dropout to LSTM Output
        rnn_out = self.out_dropout(rnn_out)
        
        # 5. Linear Projection and Permute
        output = self.output(rnn_out).permute(0, 2, 1)
        return output