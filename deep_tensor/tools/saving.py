import os
import sys
from typing import Dict

import h5py
import numpy as np
import torch 
from torch import Tensor


def is_numpy_type(data) -> bool:
    is_array = isinstance(data, np.ndarray)
    is_num = np.issubdtype(type(data), np.number)
    return is_array or is_num


def dict_to_h5(f: h5py.File | h5py.Group, d: Dict, path: str = ""):

    for key, data in d.items():
        new_path = f"{path}{os.sep}{key}"
        if isinstance(data, dict):
            dict_to_h5(f, data, new_path)
        else:
            if isinstance(data, Tensor):
                data = data.numpy()
            f.create_dataset(new_path, data=data)

    return

def h5_to_dict(f: h5py.File | h5py.Group, d: Dict | None = None) -> Dict:

    if d is None:
        d = {}

    for key, data in f.items():
        if isinstance(data, h5py.Group):
            d[key] = {}
            h5_to_dict(f[key], d[key])
        else:
            data = data[()]
            if is_numpy_type(data):
                d[key] = torch.tensor(data)
            elif isinstance(data, bytes):
                d[key] = data.decode(sys.stdout.encoding)
            elif isinstance(data, str) or isinstance(data, np.bool):
                d[key] = data
            else:
                msg = f"Unknown data type encountered: {type(data)}."
                raise Exception(msg)
    
    return d
