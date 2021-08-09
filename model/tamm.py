import torch
from torch import nn

class TaMM(nn.Module):
    def __init__(self, hidden_size, key_size, val_size):
        super(TaMM, self).__init__()
        self.temper = hidden_size ** 0.5
        self.key_embedding = nn.Embedding(key_size, hidden_size)
        self.val_embedding = nn.Embedding(val_size, hidden_size)

    def forward(self, hidden_state, key_seq, value_matrix, key_mask_matrix):
        embedding_key = self.key_embedding(key_seq)
        embedding_val = self.val_embedding(value_matrix)

        key_seq_h = embedding_key.permute(0, 2, 1)
        u = torch.matmul(hidden_state.float(), key_seq_h.float()) / self.temper

        key_mask_matrix = torch.clamp(key_mask_matrix, 0, 1)

        exp_u = torch.exp(u)
        delta_exp_u = torch.mul(exp_u, key_mask_matrix.float())

        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)

        p = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)

        embedding_val = embedding_val.permute(3, 0, 1, 2)
        o = torch.mul(p, embedding_val.float()).type_as(hidden_state)

        o = o.permute(1, 2, 3, 0)
        o = torch.sum(o, 2)

        embedding_key_matrix = torch.stack([embedding_key] * p.shape[1], 1).permute(3, 0, 1, 2)
        ko = torch.mul(p, embedding_key_matrix.float()).type_as(hidden_state).permute(1, 2, 3, 0)
        ko = torch.sum(ko, 2)
        o = torch.add(o, ko)

        return torch.add(o, hidden_state)
