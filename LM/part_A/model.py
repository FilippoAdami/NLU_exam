import torch
import torch.nn as nn

class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_RNN, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output

class LM_LSTM_Dropout(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0,
                 emb_dropout=0.1, hid_dropout=0.1, n_layers=1):
        super(LM_LSTM_Dropout, self).__init__()
        # 1. Embeddings
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # 2. Dropout after embedding
        self.emb_dropout = nn.Dropout(emb_dropout)
        # 3. LSTM (Replacing RNN)
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        # 4. Dropout before Linear
        self.out_dropout = nn.Dropout(hid_dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)
        rnn_out, _ = self.rnn(emb)
        rnn_out = self.out_dropout(rnn_out)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output