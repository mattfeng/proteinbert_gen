import torch
import csv

class SwissProtDataset(torch.utils.data.Dataset):
    def __init__(self, csvfile):
        self.csvfile = csvfile
        self.data = self._read_data()

    def _read_data(self):
        data = []
        with open(self.csvfile, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                _, prot_id, seq, desc = row
                data.append({
                    "prot_id": self._nullable(prot_id),
                    "seq": self._nullable(seq),
                    "desc": self._nullable(desc)
                })
        return data

    def _nullable(self, x):
        NO_DATA_SENTINEL = "<NO_DATA>"
        if x == NO_DATA_SENTINEL:
            return None
        return x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def calculate_splits(train_pct, val_pct, test_pct, total):
    val = int(val_pct * total)
    test = int(test_pct * total)
    train = total - val - test
    return train, val, test
