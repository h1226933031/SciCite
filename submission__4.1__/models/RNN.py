import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, embed_matrix, dropout_rate, n_hidden, num_classes):
        super().__init__()
        self.vocab_size = embed_matrix.shape[0]
        self.embedding_dim = embed_matrix.shape[-1]
        self.embedding = nn.Embedding.from_pretrained(embed_matrix, freeze=False)
        self.drop = nn.Dropout(dropout_rate)
        self.rnn = nn.GRU(self.embedding_dim, n_hidden, batch_first=True)
        self.linear = nn.Linear(in_features=n_hidden, out_features=num_classes, bias=True)

    def forward(self, x: torch.Tensor):  # x shape: [batch, max_word_length, embedding_length]
        emb = self.embedding(x)  # input : [batch_size, seq_len, embedding_dim]
        emb = self.drop(emb)
        output, _ = self.rnn(emb)  # [batch_size, seq_len, hidden_size]
        output = self.linear(output)  # [batch_size, seq_len, hidden_size]
        return output[:, -1]  # (只需要最后一个RNN的输出！！！) only use the last out put of RNN!
