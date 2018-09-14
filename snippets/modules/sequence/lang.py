from typing import Callable, Iterable


class Lang(object):
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    SOS_INDEX = 0
    EOS_INDEX = 1

    def __init__(self, name: str):
        self.name = name
        self.token2index = {self.SOS_TOKEN: self.SOS_INDEX, self.EOS_TOKEN: self.EOS_INDEX}
        self.token2count = {}
        self.index2token = {self.EOS_INDEX: self.EOS_TOKEN, self.SOS_INDEX: self.SOS_TOKEN}
        self.n_words = 2

    def add_sentences(self, sentences: Iterable[str], tokenizer=None):
        for sentence in sentences:
            self.add_sentence(sentence, tokenizer)

    def add_sentence(self, sentence: str, tokenizer: Callable[[str], Iterable[str]] = None):
        if tokenizer is None:
            tokenizer = lambda x: x.split(" ")
        for token in tokenizer(sentence):
            self.add_token(token)

    def add_token(self, token):
        if token not in self.token2count:
            self.token2index[token] = self.n_words
            self.index2token[self.n_words] = token
            self.n_words += 1
            self.token2count[token] = 1
        else:
            self.token2count[token] += 1

    def sentence_to_tensor(self, sentence, tokenizer=lambda x: x.split(" ")):
        import torch
        indexes = [self.token2index[_] for _ in tokenizer(sentence)]
        indexes.append(self.EOS_INDEX)
        # noinspection PyCallingNonCallable
        return torch.tensor(indexes, dtype=torch.long)

    def tensor_to_tokens(self, x):
        assert len(x.size()) == 1
        tokens = ([self.index2token[_.item()] for _ in x])
        return tokens
