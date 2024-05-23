import torch

# from torch_geometric.nn import knn
# from torch_geometric.nn.pool import knn
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter


def knn_interpolate(x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor,
                    batch_x: OptTensor = None, batch_y: OptTensor = None,
                    k: int = 3, num_workers: int = 1):
    r"""The k-NN interpolation from the `"PointNet++: Deep Hierarchical
    Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper.
    For each point :math:`y` with position :math:`\mathbf{p}(y)`, its
    interpolated features :math:`\mathbf{f}(y)` are given by

    .. math::
        \mathbf{f}(y) = \frac{\sum_{i=1}^k w(x_i) \mathbf{f}(x_i)}{\sum_{i=1}^k
        w(x_i)} \textrm{, where } w(x_i) = \frac{1}{d(\mathbf{p}(y),
        \mathbf{p}(x_i))^2}

    and :math:`\{ x_1, \ldots, x_k \}` denoting the :math:`k` nearest points
    to :math:`y`.

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        pos_x (torch.Tensor): Node position matrix
            :math:`\in \mathbb{R}^{N \times d}`.
        pos_y (torch.Tensor): Upsampled node position matrix
            :math:`\in \mathbb{R}^{M \times d}`.
        batch_x (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b_x} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node from :math:`\mathbf{X}` to a specific example.
            (default: :obj:`None`)
        batch_y (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b_y} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node from :math:`\mathbf{Y}` to a specific example.
            (default: :obj:`None`)
        k (int, optional): Number of neighbors. (default: :obj:`3`)
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
    """

    with torch.no_grad():
        assign_index = knn(pos_x, pos_y, k, batch_x=batch_x, batch_y=batch_y,
                           num_workers=num_workers)
        y_idx, x_idx = assign_index[0], assign_index[1]
        diff = pos_x[x_idx] - pos_y[y_idx]
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

    y = scatter(x[x_idx] * weights, y_idx, 0, pos_y.size(0), reduce='sum')
    y = y / scatter(weights, y_idx, 0, pos_y.size(0), reduce='sum')

    return y

#########################################

import warnings
from typing import Optional
from torch import Tensor

import torch_geometric.typing
# from torch_geometric.typing import OptTensor, torch_cluster

# from .asap import ASAPooling
# from .avg_pool import avg_pool, avg_pool_neighbor_x, avg_pool_x
# from .edge_pool import EdgePooling
# from .glob import global_add_pool, global_max_pool, global_mean_pool
# from .knn import KNNIndex, L2KNNIndex, MIPSKNNIndex
# from .graclus import graclus
# from .max_pool import max_pool, max_pool_neighbor_x, max_pool_x
# from .mem_pool import MemPooling
# from .pan_pool import PANPooling
# from .sag_pool import SAGPooling
# from .topk_pool import TopKPooling
# from .voxel_grid import voxel_grid
# from .approx_knn import approx_knn, approx_knn_graph

def knn(
    x: Tensor,
    y: Tensor,
    k: int,
    batch_x: OptTensor = None,
    batch_y: OptTensor = None,
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Finds for each element in :obj:`y` the :obj:`k` nearest points in
    :obj:`x`.

    .. code-block:: python

        import torch
        from torch_geometric.nn import knn

        x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        batch_y = torch.tensor([0, 0])
        assign_index = knn(x, y, 2, batch_x, batch_y)

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{M \times F}`.
        k (int): The number of neighbors.
        batch_x (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        batch_y (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. (default: :obj:`None`)
        cosine (bool, optional): If :obj:`True`, will use the cosine
            distance instead of euclidean distance to find nearest neighbors.
            (default: :obj:`False`)
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor`
    """
#     if not torch_geometric.typing.WITH_TORCH_CLUSTER_BATCH_SIZE:
#         # return torch_cluster.knn(x, y, k, batch_x, batch_y, cosine,
#         #                          num_workers)
#         return cluster_knn(x, y, k, batch_x, batch_y, cosine,
#                                  num_workers)
    return cluster_knn(x, y, k, batch_x, batch_y)
    
    # return torch_cluster.knn(x, y, k, batch_x, batch_y, cosine, num_workers,
    #                          batch_size)
    return cluster_knn(x, y, k, batch_x, batch_y, cosine, num_workers,
                             batch_size)

#############################################################################

import torch
import scipy.spatial

# if torch.cuda.is_available():
    # import torch_cluster.knn_cuda


def cluster_knn(x, y, k, batch_x=None, batch_y=None):
    r"""Finds for each element in :obj:`y` the :obj:`k` nearest points in
    :obj:`x`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{M \times F}`.
        k (int): The number of neighbors.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. testsetup::

        import torch
        from torch_cluster import knn

    .. testcode::

        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch_x = torch.tensor([0, 0, 0, 0])
        >>> y = torch.Tensor([[-1, 0], [1, 0]])
        >>> batch_x = torch.tensor([0, 0])
        >>> assign_index = knn(x, y, 2, batch_x, batch_y)
    """

    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    # if x.is_cuda:
    #     return torch_cluster.knn_cuda.knn(x, y, k, batch_x, batch_y)

    # Rescale x and y.
    min_xy = min(x.min().item(), y.min().item())
    x, y = x - min_xy, y - min_xy

    max_xy = max(x.max().item(), y.max().item())
    x, y, = x / max_xy, y / max_xy

    # Concat batch/features to ensure no cross-links between examples exist.
    x = torch.cat([x, 2 * x.size(1) * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
    y = torch.cat([y, 2 * y.size(1) * batch_y.view(-1, 1).to(y.dtype)], dim=-1)

    tree = scipy.spatial.cKDTree(x.detach().numpy())
    dist, col = tree.query(
        y.detach().cpu(), k=k, distance_upper_bound=x.size(1))
    dist = torch.from_numpy(dist).to(x.dtype)
    col = torch.from_numpy(col).to(torch.long)
    row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, k)
    
    mask = ~torch.isinf(dist).view(-1) #1 - torch.isinf(dist).view(-1)
    # row, col = row.view(-1)[mask], col.view(-1)[mask]
    
    row, col = row.view(-1), col.view(-1)

    return torch.stack([row, col], dim=0)
