import torch
import numpy as np
import functools
from torch import nn
from torch_geometric.nn import EdgeConv
from torch_scatter import scatter_sum
from samplers import samples_gen
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

class PointNet(nn.Module):
    def __init__(self, feat_len):
        super(PointNet, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.mlp1 = nn.Linear(1024, feat_len)
        self.bn6 = nn.BatchNorm1d(feat_len)

    """
        Input: B x N x 3 (B x P x N x 3)
        Output: B x F (B x P x F)
    """

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))

        x = x.max(dim=-1)[0]

        x = torch.relu(self.bn6(self.mlp1(x)))
        return x

def marginal_prob_std(t, conf):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.
    Returns:
      The standard deviation.
    """
    t = torch.tensor(t, device=conf.device)
    return torch.sqrt((conf.sigma ** (2 * t) - 1.) / 2. / np.log(conf.sigma))

def diffusion_coeff(t, conf):
    """Compute the diffusion coefficient of our SDE.
    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.
    Returns:
      The vector of diffusion coefficients.
    """
    return torch.tensor(conf.sigma ** t, device=conf.device)


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
