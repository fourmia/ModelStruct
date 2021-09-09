from pathlib import PosixPath
from typing import List, Tuple
import h5py
import numpy as np
import pandas as pd
import torch
# from torch.utils.data import Dataset


class Dataset():
    """PyTorch data set to work with pre-packed hdf5 data base files.

    Parameters
    ----------
    h5_file : PosixPath
        Path to hdf5 file, containing the bundled data
    """

    def __init__(
        self,
        h5_file: PosixPath,
    ):
        self.h5_file = h5_file

        # Placeholder for catchment attributes stats
        self.df = None

        (self.x, self.y) = self._preload_data()

        self.num_samples = self.y.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y = self.y[idx]

        # 进行值到dbz及dbz到掩码的转变
        x = 255 * x /90
        x[x<0] = 0
        x[x>255] = 255
        # x = np.piecewise(x, [x<0, x>=255], [0, 255])

        y = 10*np.log(58.53) + 10*1.56*np.log(y+0.001)
        y = 255 * y /90
        y[y<0] = 0
        y[y>255] = 255
        # y = np.piecewise(y, [y<0, y>=255], [0, 255])



        # convert to torch tensors
        x = torch.from_numpy((x.astype(np.float32))/255)

        y = torch.from_numpy((y.astype(np.float32))/255)
        print(x, y)
        return {'image':x, 'mask':y}

    def _preload_data(self):

        with h5py.File(self.h5_file, "r") as f:
            if 'train' in self.h5_file:
                x = f["input_data"][568:570]
                y = f["target_data"][568:570]
                #x = f["input_data"][:]
                #y = f["target_data"][:]
            else:
                x = f["input_data"][:2]
                y = f["target_data"][:2]

        return x, y
