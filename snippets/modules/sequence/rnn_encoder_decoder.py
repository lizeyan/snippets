import torch
import torch.nn as nn
import torch.nn.functional as func


class EncoderSeq(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, embedding_size: int, *,
                 n_layers: int = 1, reverse_input: bool=False, bidirectional: bool=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.reverse_input = reverse_input
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.n_direction = 2 if bidirectional else 1

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size,
                          num_layers=n_layers, bidirectional=bidirectional)

    def forward(self, x, hidden=None):
        """
        :param x: A tensor in shape (seq_length, batch_size, )
        :param hidden: Corresponding hidden state, keeping it None will use new hidden state
        :return: A tensor in shape (seq_length, batch_size, hidden_size * n_direction)
        """
        if self.reverse_input:
            x = torch.flip(x, dims=(0,))

        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)

        return output, hidden


class DecoderSeq(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, embedding_size: int, *,
                 n_layers: int = 1, dropout_p: float = 0.5,
                 bidirectional: bool=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_direction = 2 if bidirectional else 1

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.rnn = nn.GRU(embedding_size, hidden_size,
                          num_layers=n_layers, bidirectional=bidirectional)
        self.out = nn.Sequential(
            nn.Linear(hidden_size * self.n_direction, output_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, hidden=None, **kwargs):
        """
        :param x: A Tensor in shape (seq_length, batch_size, )
        :param hidden: Corresponding hidden state, keeping it None will use new hidden state
        :return: A Tensor in shape (seq_length, batch_size, output_size * n_direction)
        """
        embedded = self.embedding(x)
        embedded = self.input_dropout(embedded)
        output = func.relu(embedded)
        output, hidden = self.rnn(output, hidden)
        output = self.out(output)

        return output, hidden

    def forward_n(self, x, hidden=None, *, n_steps: int, **kwargs):
        """
        :param x: A Tensor in shape (batch_size,), representing an element in a sequence
        :param hidden: Corresponding hidden state, keeping it None will use new hidden state
        :param n_steps: maximum steps allowed
        :return: A Tensor in shape (n_steps, batch_size, output_size)
        """
        batch_size = x.size(0)
        decoder_input = x.view(1, -1)
        assert decoder_input.size() == (1, batch_size, )
        decoder_outputs = torch.zeros(n_steps, batch_size, self.output_size, device=x.device)
        output_length = 0
        for di in range(n_steps):
            decoder_output, decoder_hidden = self.forward(decoder_input, hidden, **kwargs)
            _, top_index = decoder_output.topk(1)
            decoder_input = top_index.squeeze(-1).detach()
            decoder_outputs[di] = decoder_output[0]
            output_length += 1
        return decoder_outputs
