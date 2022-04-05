import torch
import torch.nn as nn
import torch.nn.functional as F

BOX_OFFSETS = torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]])


def hash(coords, log2_T):
    """
    coords: this function can process up to 7 dim coordinates
    """
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    return torch.tensor((1 << log2_T) - 1).to(xor_result.device) & xor_result


class MultiresolutionHashEncoder(nn.Module):
    def __init__(self, bounding_box, *, L=16, log2_T=19, F=2, N_min=16, N_max=512) -> None:
        """
        bounding_box: Tuple[tensor,tensor] - Minimum and maximum voxel for bounding box of scene
        L: int - The number of resolutions at which we encode
        log2_T: int - Log2 of the hash table size
        F: int - Number of output features per entry
        N_min: int - The coarest resolution at which we encode
        N_max: int - The finest resolution at which we encode
        """
        super(MultiresolutionHashEncoder, self).__init__()
        self.bounding_box = bounding_box
        self.L = L
        self.log2_T = log2_T
        self.F = F
        self.N_min = torch.tensor(N_min)
        self.N_max = torch.tensor(N_max)

        # compute b - Eq (3) [Mueller] (basically torch.logspace)
        self.b = torch.exp((torch.log(self.N_max) - torch.log(self.N_min)) / (self.L - 1))
        self.resolutions = [torch.floor(self.N_min * self.b**l) for l in range(L)]

        # hash table
        self.embeddings = nn.ModuleList([nn.Embedding(2**log2_T, F) for _ in range(self.L)])
        for l in range(L):
            nn.init.uniform_(self.embeddings[l].weight, a=-0.0001, b=0.0001)

        # specify out dim
        self.out_dim = self.L * self.F

    def _trilerp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        """
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x F
        """
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = (
            voxel_embedds[:, 0] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 4] * weights[:, 0][:, None]
        )
        c01 = (
            voxel_embedds[:, 1] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 5] * weights[:, 0][:, None]
        )
        c10 = (
            voxel_embedds[:, 2] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 6] * weights[:, 0][:, None]
        )
        c11 = (
            voxel_embedds[:, 3] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 7] * weights[:, 0][:, None]
        )

        # step 2
        c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
        c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]

        # step 3
        c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]

        return c

    def _get_voxel_vertices(self, xyz, resolution):
        box_min, box_max = self.bounding_box
        if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
            print("ALERT: some points are outside bounding box. Clipping them!")
            xyz = torch.clamp(xyz, min=box_min, max=box_max)

        grid_size = (box_max - box_min) / resolution
        bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
        voxel_min_vertex = bottom_left_idx * grid_size + box_min
        voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0, 1.0, 1.0]) * grid_size

        voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
        hashed_voxel_indices = self._hash(voxel_indices)

        return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

    def _hash(self, coords):
        return hash(coords, self.log2_T)

    def forward(self, x):
        """
        x: B x 3
        """
        x_emb = []

        for l in range(self.L):
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = self._get_voxel_vertices(
                x, resolution=self.resolutions[l]
            )
            voxel_embeddings = self.embeddings[l](hashed_voxel_indices)
            x_emb.append(self._trilerp(x, voxel_min_vertex, voxel_max_vertex, voxel_embeddings))

        return torch.cat(x_emb, dim=-1)


if __name__ == "__main__":
    bounding_box = (torch.tensor([0, 0, 0]), torch.tensor([1, 1, 1]))
    enc = MultiresolutionHashEncoder(bounding_box)
    x = torch.Tensor([[0.1233, 0.234323, 0.23232]])
    print(enc.forward(x))
