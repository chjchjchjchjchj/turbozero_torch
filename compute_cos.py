import numpy as np
from tqdm import trange
from pathlib import Path
import torch
# import zarr

# vectors = np.load("24D_196560_1.npy")
# save_dir = Path("/data/haojun/cos_matrix")
# save_dir.mkdir(exist_ok=True, parents=True)
# N = vectors.shape[0]
# valid_values = np.array([0, -1, -1/4, 1/4, -1/2, 1/2, 1], dtype=np.float32)
# batch_size=1000
# device = torch.device("cuda")
# vectors = torch.from_numpy(vectors).to(device).to(torch.float16)
# for i in trange(0, N, batch_size):
#     end = min(i + batch_size, N)
#     vectors_a = vectors[i:end]
#     cos_arr = vectors_a @ vectors.T
#     m = cos_arr.shape[0]
#     cos_arr = cos_arr.cpu().numpy()
#     for j in range(m):
#         save_path = save_dir / f"{i + j}.zarr"
#         # print(save_path)
#         zarr.save(save_path, cos_arr[j])




# import numpy as np
# import zarr

# aa = zarr.load_group("/data/haojun/cos_matrix")
# print(aa)


# import zarr
# import numpy as np
# import os
# from tqdm import tqdm
# # 指定目录
# dir_path = "/data/haojun/cos_matrix"

# # 获取所有 .zarr 文件并按数字排序
# zarr_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.zarr')], key=lambda x: int(x.split('.')[0]))

# # 读取所有 zarr 文件并合并
# zarr_files = zarr_files[:100]
# arrays = []
# for f in tqdm(zarr_files, desc="Loading Zarr files", unit="file"):
#     arrays.append(zarr.open(os.path.join(dir_path, f), mode='r')[:])

# # 转换为 2D np.array
# result = np.stack(arrays, axis=0)
# import ipdb; ipdb.set_trace()
# print(result.shape)



# vectors = np.load("24D_196560_1.npy")
# save_dir = Path("/data/haojun/cos_matrix")
# save_dir.mkdir(exist_ok=True, parents=True)
# N = vectors.shape[0]
# valid_values = np.array([0, -1, -1/4, 1/4, 1], dtype=np.float32)
# batch_size=24570
# device = torch.device("cuda")
# vectors = torch.from_numpy(vectors).to(device).to(torch.float16)
# cos_matrix = []
# for i in trange(0, N, batch_size):
#     end = min(i + batch_size, N)
#     vectors_a = vectors[i:end]
#     cos_arr = vectors_a @ vectors.T
#     m = cos_arr.shape[0]
#     cos_arr = cos_arr.cpu().numpy()
#     cos_matrix.append(cos_arr)

# cos_matrix = np.concatenate(cos_matrix, axis=0)

# adj_matrix = np.min(np.abs(cos_matrix[:, :, None] - valid_values), axis=2) < 1e-6

# from scipy import sparse
# sparse_adj = sparse.csr_matrix(adj_matrix)
# sparse.save_npz(save_dir / "adj_matrix_sparse.npz", sparse_adj)

vectors = np.load("24D_196560_1.npy")
save_dir = Path("/data/haojun/cos_matrix")
save_dir.mkdir(exist_ok=True, parents=True)
N = vectors.shape[0]
valid_values = np.array([0, -1, -1/4, 1/4, 1], dtype=np.float32)
batch_size=24570
device = torch.device("cuda")
vectors = torch.from_numpy(vectors).to(device).to(torch.float16)
for i in trange(0, N, batch_size):
    end = min(i + batch_size, N)
    vectors_a = vectors[i:end]
    cos_arr = vectors_a @ vectors.T
    m = cos_arr.shape[0]
    cos_arr = cos_arr.cpu().numpy()
    adj_arr = np.min(np.abs(cos_arr[:, :, None] - valid_values), axis=1) < 1e-6
    print(adj_arr.nbytes)
    save_dir = save_dir / f"adj_matrix_{i}.npy"
    np.save(save_dir, adj_arr)


