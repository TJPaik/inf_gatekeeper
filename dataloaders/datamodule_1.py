import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from typing import Optional
from torch import nn

from transforms import BaselineWander, DCTDelete, DownSampleReconstruct, PowerlineNoise, RandomResizedCrop, \
    GaussianFilter1D, GaussianNoise, RandomTF, RandomCrop

from torch.utils.data import Dataset
import numpy as np

augment_long_tf = nn.Sequential(
    RandomTF(DownSampleReconstruct(1250, 5000)),
    RandomTF(DownSampleReconstruct(2500, 5000)),
    BaselineWander(random=True),
    RandomTF(DCTDelete([slice(0, 2), slice(600, None, None)]), random_channel=True),
    RandomTF(GaussianFilter1D(4, ), random_channel=True),
    RandomTF(GaussianNoise(sigma=0.005), random_channel=True),
    RandomTF(PowerlineNoise(500, scale=0.05), random_channel=True),
    RandomCrop(4750)
)
augment_short_tf = nn.Sequential(
    RandomTF(DownSampleReconstruct(350, 1238)),
    RandomTF(DownSampleReconstruct(625, 1238)),
    BaselineWander(random=True),
    RandomTF(DCTDelete([slice(0, 1), slice(300, None, None)]), random_channel=True),
    RandomTF(GaussianFilter1D(4, ), random_channel=True),
    RandomTF(GaussianNoise(sigma=0.005), random_channel=True),
    RandomTF(PowerlineNoise(500, scale=0.05), random_channel=True),
    RandomCrop(1176)
)

test_long_tf = nn.Sequential(
    BaselineWander(random=False),
    RandomCrop(4750)
)
test_short_tf = nn.Sequential(
    BaselineWander(random=False),
    RandomCrop(1176)
)

common_path = '/mount/share_data/inf_gatekeeper/Data/Combined_ECGs'
train_test_split = np.load(
    f'{common_path}/220407_complete_data_train_test_split.npz')
ecg_data = np.load(f'{common_path}/220407_complete_data_ecgs_with_samples.npz')
ecg_data = {k: v for k, v in ecg_data.items()}
ecg_data['DB_label'] = [
    1 if el in ['AICON-SNUBH', 'SNUBH_sample'] else 0
    for el in ecg_data['DB']
]


class DsLoader(Dataset):
    def __init__(self, step, split_number):
        assert step in ['train', 'val', 'test']
        train_test_split_info = train_test_split[f'split_{split_number}']
        self.data_idx = np.where(train_test_split_info == step)[0]

        self.tf_long = augment_long_tf if step == 'train' else test_long_tf
        self.tf_short = augment_short_tf if step == 'train' else test_short_tf

        self.dict_names = ['wave_2_whole_500',
                           'wave_1_500', 'wave_2_500', 'wave_3_500',
                           'wave_aVR_500', 'wave_aVL_500', 'wave_aVF_500',
                           'wave_V1_500', 'wave_V2_500', 'wave_V3_500', 'wave_V4_500', 'wave_V5_500', 'wave_V6_500',
                           'DB_label']

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        real_idx = self.data_idx[idx]
        result = list(ecg_data[el][real_idx] for el in self.dict_names)
        return [
            self.tf_long(torch.from_numpy(result[0]).view(1, -1).float()),
            self.tf_short(torch.from_numpy(np.asarray(result[1:13])).float()),
            int(result[-1])
        ]


class DataModule(pl.LightningDataModule):
    def __init__(self, split_number, batch_size, num_works):
        super(DataModule, self).__init__()
        self.train = None
        self.val = None
        self.test = None
        self.num_works = num_works
        self.split_number = split_number
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        self.train = DsLoader('train', self.split_number)
        self.val = DsLoader('val', self.split_number)
        self.test = DsLoader('test', self.split_number)

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, True, num_workers=self.num_works)
        

    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size, False, num_workers=self.num_works)

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size, False, num_workers=self.num_works)