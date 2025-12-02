# -*-coding:utf8-*-

from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data  # a list of samples
        self.transforms = transforms

    def set_data(self, data):
        self.data.clear()
        self.data = data

    def __getitem__(self, index):
        di = self.data[index]
        if len(di) == 3:
            d_id, sp, lab = di
            if self.transforms is not None:
                sp = self.transforms(sp)
            return d_id, sp, lab
        elif len(di) == 4:
            d_id, sp, lab, logit = di
            if self.transforms is not None:
                sp = self.transforms(sp)
            return d_id, sp, lab, logit
        else:
            sp, lab = di
            if self.transforms is not None:
                sp = self.transforms(sp)
            return sp, lab

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for di in self.data:
            if len(di) == 3:
                d_id, sp, lab = di
                if self.transforms is not None:
                    sp = self.transforms(sp)
                yield d_id, sp, lab
            elif len(di) == 4:
                d_id, sp, lab, logit = di
                if self.transforms is not None:
                    sp = self.transforms(sp)
                yield d_id, sp, lab, logit
            else:
                sp, lab = di
                if self.transforms is not None:
                    sp = self.transforms(sp)
                yield sp, lab
