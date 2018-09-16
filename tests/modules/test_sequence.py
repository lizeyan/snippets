from snippets.modules.sequence import *
import unittest


class TestSequence(unittest.TestCase):
    def test_encoder(self):
        seq_length = 7
        batch_size = 4
        n_layers = 2
        n_direction = 2
        hidden_size = 13
        vocab_size = 11
        encoder = EncoderSeq(input_size=vocab_size, hidden_size=hidden_size,
                             embedding_size=12, n_layers=n_layers,
                             bidirectional=True if n_direction > 1 else False,
                             reverse_input=True)
        input_tensor = torch.randint(low=0, high=vocab_size,
                                     size=(seq_length, batch_size), dtype=torch.long)
        encoder_outputs, hidden = encoder(input_tensor, None)
        self.assertEqual(encoder_outputs.size(),
                         (seq_length, batch_size, hidden_size * n_direction))
        self.assertEqual(hidden.size(), (n_layers * n_direction, batch_size, hidden_size))

    def test_decoder(self):
        seq_length = 7
        batch_size = 4
        n_layers = 2
        n_direction = 2
        hidden_size = 13
        vocab_size = 11
        max_steps = 12
        decoder = DecoderSeq(hidden_size=hidden_size, output_size=vocab_size,
                             embedding_size=12, n_layers=n_layers,
                             bidirectional=True if n_direction > 1 else False)
        input_tensor = torch.randint(low=0, high=vocab_size,
                                     size=(seq_length, batch_size), dtype=torch.long)
        decoder_outputs, hidden = decoder(input_tensor, None)
        self.assertEqual(decoder_outputs.size(),
                         (seq_length, batch_size, vocab_size))
        self.assertEqual(hidden.size(), (n_layers * n_direction, batch_size, hidden_size))
        decoder_outputs = decoder.forward_n(input_tensor[0], None, n_steps=max_steps)
        self.assertEqual(decoder_outputs.size(),
                         (max_steps, batch_size, vocab_size))

    def test_trainer(self):
        seq_length = 7
        batch_size = 4
        n_layers = 2
        n_direction = 2
        hidden_size = 13
        vocab_size = 11
        encoder = EncoderSeq(input_size=vocab_size, hidden_size=hidden_size,
                             embedding_size=12, n_layers=n_layers,
                             bidirectional=True if n_direction > 1 else False,
                             reverse_input=True)
        decoder = DecoderSeq(hidden_size=hidden_size, output_size=vocab_size,
                             embedding_size=12, n_layers=n_layers,
                             bidirectional=True if n_direction > 1 else False)
        trainer = Seq2SeqTrainer(encoder=encoder, decoder=decoder)
        trainer.teach_forcing_prob = 1
        input_tensors = list(torch.randint(low=0, high=vocab_size,
                                           size=(random.randint(5, seq_length + 1),),
                                           dtype=torch.long) for _ in range(batch_size))
        trainer.step(input_tensors, input_tensors)
        trainer.teach_forcing_prob = 0
        trainer.step(input_tensors, input_tensors)

    def test_inference(self):
        seq_length = 7
        batch_size = 4
        n_layers = 2
        n_direction = 2
        hidden_size = 13
        vocab_size = 11
        encoder = EncoderSeq(input_size=vocab_size, hidden_size=hidden_size,
                             embedding_size=12, n_layers=n_layers,
                             bidirectional=True if n_direction > 1 else False,
                             reverse_input=True)
        decoder = DecoderSeq(hidden_size=hidden_size, output_size=vocab_size,
                             embedding_size=12, n_layers=n_layers,
                             bidirectional=True if n_direction > 1 else False)
        target_lang = Lang(name="lang")
        target_lang.add_sentence("a b c d e f g h i g k l m n")
        inference = Seq2SeqInference(encoder=encoder, decoder=decoder, target_lang=target_lang)
        input_tensor = torch.randint(low=0, high=vocab_size,
                                     size=(seq_length,), dtype=torch.long)
        inference(input_tensor, max_length=2)
