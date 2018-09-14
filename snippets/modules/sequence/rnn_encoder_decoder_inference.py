from typing import List

from . import DecoderSeq, EncoderSeq, Lang
import torch


class Seq2SeqInference:
    def __init__(self, encoder: EncoderSeq, decoder: DecoderSeq, target_lang: Lang):
        self.encoder = encoder
        self.decoder = decoder
        self.target_lang = target_lang

    def __call__(self, seq, max_length: int, ) -> List[str]:
        """
        :param seq: A tensor in shape (seq_length,)
        :return:
        """
        device = seq.device
        encoder_outputs, hidden = self.encoder(seq.unsqueeze(1), None)

        # noinspection PyCallingNonCallable
        decoder_input = torch.tensor([[self.target_lang.SOS_INDEX]], device=device)
        decoder_outputs = torch.zeros(max_length, self.decoder.output_size, device=device)
        for di in range(max_length):
            decoder_output, decoder_hidden = \
                self.decoder.forward(decoder_input, hidden, encoder_outputs=encoder_outputs)
            _, top_index = decoder_output.topk(1)
            decoder_input = top_index.squeeze(-1).detach()
            decoder_outputs[di] = decoder_output[0, 0]
            if decoder_input.item() == self.target_lang.EOS_INDEX:
                break
        _, decoder_outputs = decoder_outputs.topk(1)
        output_words = self.target_lang.tensor_to_tokens(decoder_outputs[:, 0])
        return output_words
