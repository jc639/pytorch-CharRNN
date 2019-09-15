from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from torch import nn, optim

class CharRNN(nn.Module):
    
    def __init__(self, vocab_size, embed_size, hidden, 
                 layers, output_size, fc_size,
                rnn_drop=0.2, clas_drop=[0.5, 0.5]):
        super(CharRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.rnn_layers = layers
        self.hidden_size = hidden
        self.rnn_drop = rnn_drop
        self.clas_drop=clas_drop
        self.fc_size = fc_size
        
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden, dropout=rnn_drop, num_layers=layers,
                           batch_first=True)
        self.bn = nn.BatchNorm1d(hidden)
        self.drop1 = nn.Dropout(clas_drop[0])
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden, fc_size)
        self.drop2 = nn.Dropout(clas_drop[1])
        self.fc2 = nn.Linear(fc_size, output_size)
        
    def init_hidden(self, bs):
        h_o = torch.zeros(self.rnn_layers, bs, self.hidden_size)
        c_o = torch.zeros(self.rnn_layers, bs, self.hidden_size)
        if next(self.parameters()).is_cuda:
            h_o = h_o.cuda()
            c_o = c_o.cuda()
               
        return (h_o, c_o)
    
    def forward(self, xb, lens):
        
        hidden = self.init_hidden(xb.size()[0])
        xb = self.emb(xb)
        packed_input = pack_padded_sequence(xb, lens, batch_first=True)
        packed_output, (ht, ct) = self.lstm(packed_input, 
                                            hidden)
        xb = self.bn(ht[-1])
        xb = self.relu(self.fc1(self.drop1(xb)))
        
        out = self.fc2(self.drop2(xb))
        
        return out