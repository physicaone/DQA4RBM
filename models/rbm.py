import torch

class RBM(torch.nn.Module):
    def __init__(self, n_vis, n_hid, mask):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(n_vis, n_hid) * 0.01)
        self.mask = mask

    def sample_h(self, v):
        p = torch.tanh(v @ (self.W * self.mask))
        return torch.sign(p + torch.randn_like(p) * 0.01)

    def sample_v(self, h):
        p = torch.tanh(h @ (self.W * self.mask).t())
        return torch.sign(p + torch.randn_like(p) * 0.01)
