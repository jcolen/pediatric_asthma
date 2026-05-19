import torch
import pandas as pd
import numpy as np

from torch.utils.data import Subset, TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from .base import AirQualityDataModule

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TorchAQModule(AirQualityDataModule):
    def __init__(self, 
                 loader_kwargs : dict = {'batch_size': 256, 'pin_memory': True, 'num_workers': 2},
                 **kwargs):
        super().__init__(**kwargs)
        self.loader_kwargs = loader_kwargs

    def setup(self, stage : str):
        """ Done for training deep learning models
        """
        super().setup(stage)

        # Built torch dataset and perform train/test split
        if stage == 'fit':
            self.train_dataset = TensorDataset(
                torch.FloatTensor(self.x[self.train_ilocs]),
                torch.FloatTensor(self.y[self.train_ilocs]),
                torch.FloatTensor(self.offset[self.train_ilocs])
            )

            self.test_dataset = TensorDataset(
                torch.FloatTensor(self.x[self.test_ilocs]),
                torch.FloatTensor(self.y[self.test_ilocs]),
                torch.FloatTensor(self.offset[self.test_ilocs]),
            )
            logger.info(f'Train dataset size: {len(self.train_dataset)}\tTest dataset size: {len(self.test_dataset)}')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.loader_kwargs, shuffle=True)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, **self.loader_kwargs, shuffle=False)