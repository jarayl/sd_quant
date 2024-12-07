
import numpy as np
import torch

import torch.nn.functional as F

def fwht(x):
    """Fast Walsh-Hadamard Transform along the last dimension without normalization."""
    original_shape = x.shape
    N = x.shape[-1]
    # N_padded = 2 ** int(np.ceil(np.log2(N))) if (N & (N - 1)) != 0 else N
    # pad_size = N_padded - N if N_padded != N else 0

    # if pad_size > 0:
    #     x = F.pad(x, (0, pad_size))

    x = x.reshape(-1, N)
    batch_dim, d = x.shape
    h = 2
    while h <= d:
        hf = h // 2
        x = x.view(batch_dim, d // h, h)

        half_1, half_2 = x[:, :, :hf], x[:, :, hf:]

        x = torch.cat((half_1 + half_2, half_1 - half_2), dim=-1)

        h *= 2

    return (x / np.sqrt(d)).view(*original_shape)


v1 = torch.Tensor([[1,2,3], [1,3,4]])
v2 = torch.Tensor([[5,5,5],[6,6,6]])
# print(F.linear(v1,v2))
# print(v1.shape)
# print(v2.shape)

# Pad if necessary
N = v1.shape[-1]
sign_vector = torch.randint(0, 2, (N,)) * 2 - 1  # Random +/-1

N_padded = 2 ** int(np.ceil(np.log2(N))) if (N & (N - 1)) != 0 else N
N_padded = N_padded
pad_size = N_padded - N

if pad_size > 0:
    v1 = F.pad(v1, (0, pad_size))  # Pad last dimension (input dimension)

N_2 = v2.shape[-1]

# Pad if necessary
N_padded_2 = 2 ** int(np.ceil(np.log2(N_2))) if (N & (N - 1)) != 0 else N
N_padded_2 = N_padded
pad_size_2 = N_padded_2 - N_2

if pad_size_2 > 0:
    v2 = F.pad(v2, (0, pad_size_2))  # Pad last dimension (input dimension)


if pad_size > 0 and pad_size_2 > 0 and pad_size == pad_size_2:
    sign_vector = F.pad(sign_vector, (0, pad_size_2), value=0)

# print(v1.shape)
# print(v2.shape)
# print(v1)
# print (v2)

v1_prime = fwht(v1 * sign_vector)
# # Unpad if needed
# if pad_size > 0:
#     v1_prime = v1_prime[:, :N]

v2_prime = fwht(v2* sign_vector)

# # Unpad if needed
# if pad_size_2 > 0:
#     v2_prime = v2_prime[:, :N_2]

print(F.linear(v1_prime,v2_prime))
print(F.linear(v1,v2))

