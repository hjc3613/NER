from dataset.data_utils import CMeEEDataset, collate_fn
from torch.utils.data import DataLoader

if __name__ == '__main__':
    ds_train = CMeEEDataset('CMeEE/CMeEE_dev.json')
    dataloader = DataLoader(ds_train, batch_size=10, shuffle=True, collate_fn=collate_fn)
    for batch in dataloader:
        batch