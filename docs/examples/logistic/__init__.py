from collections import namedtuple
from pathlib import Path 

import torch 


DATA_PATH = Path(__file__).resolve().parent.joinpath(
    "data", 
    "german.data-numeric"
)


Data = namedtuple("Data", ["X", "y"])


def read_credit_data(fname: Path) -> Data:
    """Reads in the German credit dataset, then shifts and scales the 
    predictors such that each has a mean of zero and standard deviation 
    of 1, and scales the response variable such that it takes values in 
    {0, 1}.
    """

    with open(fname, "r") as f:
        data = [[float(l) for l in line.strip().split()] 
                for line in f.readlines()]

    data = torch.tensor(data)
    X, y = data[:, :-1], data[:, -1]

    mean_X = torch.mean(X, dim=0)
    std_X = torch.std(X, dim=0)
    
    X = (X - mean_X) / std_X
    y -= 1.0

    return Data(X, y)


credit_data = read_credit_data(DATA_PATH)