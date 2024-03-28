import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
import sys

sys.path.append("..")


class Loader:
    def __init__(self, dataset, dataset_type, device, batch_size, shuffle=True):
        self.device = device
        if dataset_type == "n-caltech101":
            self.loader = DataLoader(dataset, drop_last=True, shuffle=shuffle, batch_size=batch_size,  num_workers=8, collate_fn=collate_events_ncal, pin_memory=True)
        elif dataset_type == "gen1":
            self.loader = DataLoader(dataset, drop_last=True, shuffle=shuffle, batch_size=batch_size,  num_workers=8, collate_fn=collate_events_gen, pin_memory=True)

    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)


def collate_events_ncal(data):
    labels = []
    events = []
    for i, d in enumerate(data):
        labels.append(d[1])
        ev = np.concatenate([d[0], i * np.ones((len(d[0]), 1), dtype=np.float32)], 1)
        events.append(ev)
    events = torch.from_numpy(np.concatenate(events, 0))
    labels = default_collate(labels)
    return events, labels


def collate_events_gen(data):
    labels = []
    events = []
    for i, d in enumerate(data):
        labels.append(d[1])
        ev = np.concatenate([d[0], i * np.ones((len(d[0]), 1), dtype=np.float32)], 1)
        events.append(ev)
    events = torch.from_numpy(np.concatenate(events, 0))
    DATA = list(map(lambda x: torch.tensor(x), labels))
    labels = pad_sequence(DATA, batch_first=True)
    return events, labels
