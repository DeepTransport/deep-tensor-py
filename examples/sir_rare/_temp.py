import numpy as np

# Adjacency matrix
# 1 V - Vorarlberg
# 2 T - Tirol (Tyrol)
# 3 Sa - Salzburg
# 4 K - Kärnten (Carinthia)
# 5 St - Steiermark (Styria)
# 6 O - Oberösterreich (Upper Austria)
# 7 N - Niederösterreich (Lower Austria)
# 8 W - Wien (Vienna)
# 9 B - Burgenland
d = 9
A = np.zeros((9, 9))
A[0, 1] = A[1, 0] = 1
A[1, 2] = A[2, 1] = 1
A[1, 3] = A[3, 1] = 1
A[2, 3] = A[3, 2] = 1
A[2, 4] = A[4, 2] = 1
A[2, 5] = A[5, 2] = 1
A[3, 4] = A[4, 3] = 1
A[4, 5] = A[5, 4] = 1
A[4, 6] = A[6, 4] = 1
A[4, 8] = A[8, 4] = 1
A[5, 6] = A[6, 5] = 1
A[6, 7] = A[7, 6] = 1
A[6, 8] = A[8, 6] = 1