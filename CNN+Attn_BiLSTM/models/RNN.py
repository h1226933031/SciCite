import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dropout_rate, vocab_size, embedding_dim, hidden_size, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(dropout_rate)
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_dim, bias=True)

    def forward(self, x: torch.Tensor):  # x shape: [batch, max_word_length, embedding_length]
        emb = self.embedding(x)  # input : [batch_size, seq_len, embedding_dim]
        emb = self.drop(emb)
        output, _ = self.rnn(emb)  # [batch_size, seq_len, hidden_size]
        output = self.linear(output)  # [batch_size, seq_len, hidden_size]
        return output[:, -1]  # (只需要最后一个RNN的输出！！！) only use the last out put of RNN!
