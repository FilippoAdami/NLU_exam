import torch
import torch.nn as nn

class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or dropout == 0:
            return x
        mask = x.new_empty(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = mask.div_(1 - dropout)
        mask = mask.expand_as(x)
        return x * mask

class LM_LSTM_VDO(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index,
                 emb_dropout=0.1, hid_dropout=0.1, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lockdrop = LockedDropout()
        self.emb_dropout = emb_dropout
        self.hid_dropout = hid_dropout

        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers,
                           batch_first=True, bidirectional=False)

        self.output = nn.Linear(hidden_size, output_size, bias=False)
        # Requirement: Weight Tying
        self.output.weight = self.embedding.weight 

    def forward(self, x):
        emb = self.embedding(x)
        # Variational Dropout on Embeddings
        emb = self.lockdrop(emb, self.emb_dropout) 
        rnn_out, _ = self.rnn(emb)
        # Variational Dropout on Hidden states
        rnn_out = self.lockdrop(rnn_out, self.hid_dropout)
        out = self.output(rnn_out).permute(0, 2, 1)
        return out