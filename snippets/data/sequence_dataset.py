from torch.utils.data import Dataset
from typing import *
from snippets.modules.sequence.lang import Lang


class SequenceDataset(Dataset):
    def __init__(self, data: Iterable[str], lang: Lang=""):
        self.lang = lang
        self.tensors = list([self.lang.sentence_to_tensor(seq) for seq in data])

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        return self.tensors[index]
