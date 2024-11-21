import numpy as np
from tensor_train import Tensor


np.random.seed(3)

# A = np.random.rand(4, 4, 4, 4)
# B = Tensor(A)

A = np.random.rand(4, 4, 4, 4, 4)
B = Tensor(A)
B.compress_svd(A, eps=0.2)
print(B.rs)

print(B[-1, -1, -1, -1, -1])
print(A[-1][-1][-1][-1][-1])