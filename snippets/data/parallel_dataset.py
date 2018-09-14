from torch.utils.data import Dataset
from ..utilities import in_same_length


class ParallelDataset(Dataset):
    def __init__(self, *args):
        assert len(args) >= 1, "At least one dataset should be given"
        self.dataset_list = args
        assert in_same_length(self.dataset_list), "All dataset should be in the same length."
        self.length = len(self.dataset_list[0])

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return tuple(_[item] for _ in self.dataset_list)
