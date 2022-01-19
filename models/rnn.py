from torch import nn
from einops.layers.torch import Rearrange
from config import config
__all__ = ['SigLSTM']
import torch


class SigLSTM(nn.Module):

    def __init__(self, signal_size, patch_size=16, channels=13, hidden_dim=256, num_classes=18, dropout_rate=0.2):
        super().__init__()
        # num_patches = (signal_size // patch_size)
        # patch_dim = channels * patch_size
        self.embedding = nn.Sequential(
            Rearrange('b c s -> b s c')
        )
        self.rnn = nn.LSTM(input_size=13, hidden_size=hidden_dim,
                           batch_first=True, bidirectional=True)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, num_classes)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        x = self.dropout(self.embedding(inputs))
        _, (hidden, context) = self.rnn(x)
        hidden = torch.cat((hidden[0], hidden[-1]), dim=-1)
        return self.mlp_head(hidden)


def siglstm(**kwargs):
    model = SigLSTM(config.target_point_num)
    return model
