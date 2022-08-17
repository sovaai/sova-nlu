import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class SentenceBERT(nn.Module):
    def __init__(self, encoder: nn.Module, pooling_mode: str = 'cls'):
        super(SentenceBERT, self).__init__()
        self.encoder = encoder
        if pooling_mode not in {'cls', 'mean'}:
            raise ValueError(f'`pooling_mode` should be "cls" or "mean", got `{pooling_mode}`')
        self.pooling_mode = pooling_mode

        self.dequant = torch.quantization.DeQuantStub()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None
    ):
        hidden_states = self.encoder(input_ids, attention_mask, token_type_ids, position_ids,
                                     head_mask, output_attentions, output_hidden_states)[0]

        if self.pooling_mode == 'cls':
            sentence_embedding = hidden_states[:, 0, :]
            sentence_embedding = self.dequant(sentence_embedding)
        else:
            hidden_states = self.dequant(hidden_states)
            sentence_embedding = torch.mean(hidden_states, dim=1)

        return sentence_embedding
