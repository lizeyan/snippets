import random

import torch
import torch.nn as nn
from typing import *

from torch.nn.utils.rnn import pad_sequence


class Seq2SeqTrainer(object):
    def __init__(self, *, encoder: nn.Module, decoder: nn.Module,
                 eos_index: int = 1, sos_index: int = 0,
                 teach_forcing_prob: float = 0.5, ):
        self.teach_forcing_prob = teach_forcing_prob
        self.sos_token = sos_index
        self.eos_token = eos_index
        self.criterion = nn.NLLLoss(reduction="none")
        self.decoder = decoder
        self.encoder = encoder

    def step(self, source_tensors: List[torch.Tensor],
             target_tensors: List[torch.Tensor]) -> torch.Tensor():
        """
        :param source_tensors: list of tensor in shape (seq_length, )
        :param target_tensors: list of tensor in shape (seq_length, )
        :return: loss,
        """
        assert len(source_tensors) == len(target_tensors), "the number of source and target mismatch"
        batch_size = len(source_tensors)
        source_lengths = list(_.size(0) for _ in source_tensors)
        target_lengths = list(_.size(0) for _ in target_tensors)
        source = pad_sequence(source_tensors)
        assert source.size() == (max(source_lengths), batch_size)
        target = pad_sequence(target_tensors)
        assert target.size() == (max(target_lengths), batch_size)

        encoder_hidden = None
        source_length = source.size(0)
        target_length = target.size(0)

        encoder_outputs, encoder_hidden = self.encoder(source, encoder_hidden)
        assert encoder_outputs.size(0) == source_length
        assert encoder_outputs.size(1) == batch_size

        # noinspection PyCallingNonCallable
        sos_tensor = torch.tensor([[self.sos_token]], device=target.device).expand(-1, batch_size)
        assert sos_tensor.size() == (1, batch_size)
        decoder_hidden = encoder_hidden

        use_teach_forcing = True if random.random() < self.teach_forcing_prob else False
        decoder_input = sos_tensor
        if use_teach_forcing:
            decoder_input = torch.cat([sos_tensor, target[:-1]])
            decoder_outputs, _ = self.decoder(decoder_input, decoder_hidden,
                                              encoder_outputs=encoder_outputs)
        else:
            decoder_outputs = self.decoder.forward_n(decoder_input[0], decoder_hidden,
                                                     n_steps=target_length,
                                                     encoder_outputs=encoder_outputs)
        loss_matrix = self.criterion(decoder_outputs.permute(1, 2, 0),
                                     target.permute(1, 0))
        assert loss_matrix.size() == (batch_size, target_length)
        loss = 0
        for idx, length in enumerate(target_lengths):
            loss += torch.sum(loss_matrix[idx, :length]) / length
        return loss / batch_size
