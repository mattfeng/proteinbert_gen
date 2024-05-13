import torch
import csv

class SwissProtDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.csvfile = "../data/uniprot_sprot_seq_desc.csv"
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

sprot_all = SwissProtDataset()
sprot_train, sprot_val, sprot_test = torch.utils.data.random_split(
        sprot_all,
        calculate_splits(0.7, 0.2, 0.1, len(sprot_all))
        )

