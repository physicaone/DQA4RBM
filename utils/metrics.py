import torch

def hamming_distance_with_symmetry(x, y):
    dist_direct = (x != y).sum(dim=1)
    dist_flipped = (x != -y).sum(dim=1)
    return torch.min(dist_direct, dist_flipped).float() / x.size(1)
