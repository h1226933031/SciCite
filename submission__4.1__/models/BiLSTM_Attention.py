import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, embed_matrix, n_hidden, num_classes):
        super(Model, self).__init__()
        self.embedding_dim = embed_matrix.shape[-1]
        self.embedding = nn.Embedding.from_pretrained(embed_matrix, freeze=False)
        self.lstm = nn.LSTM(self.embedding_dim, n_hidden, bidirectional=True, batch_first=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)

    def attention_net(self, lstm_output, final_state):
        # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
        # final_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        batch_size = len(lstm_output)
        # hidden = final_state.view(batch_size,-1,1)
        hidden = torch.cat((final_state[0], final_state[1]), dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        # attn_weights : [batch_size,n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)

        # context: [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return context, soft_attn_weights

    def forward(self, X):
        '''
        :param X: [batch_size, seq_len]
        :return:
        '''
        input = self.embedding(X)  # input : [batch_size, seq_len, embedding_dim]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        # output : [batch_size, seq_len, n_hidden * num_directions(=2)]
        output, (final_hidden_state, final_cell_state) = self.lstm(input)

        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(
            attn_output), attention  # attn_output : [batch_size, num_classes], attention : [batch_size, n_step]
