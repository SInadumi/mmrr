import torch

eps = 1e-6


def calc_4d_euclidean_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes batched the euclidean distance between each pair of the two collections of row vectors.

    cf.) https://pytorch.org/docs/stable/generated/torch.cdist.html
    """
    return torch.cdist(x, y, p=2)


def calc_4d_cosine_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes batched the cosine distance between each pair of the two collections of row vectors.

    cf.) https://github.com/pytorch/pytorch/issues/48306
    """
    # (b, rel, seq, hid)
    x = torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)
    # (b, rel, seq, hid) -> (b, rel, hid, seq)
    y = torch.nn.functional.normalize(y, p=2, dim=-1, eps=eps).permute(0, 1, 3, 2)
    return torch.matmul(x, y)


def calc_4d_dot_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, y.permute(0, 1, 3, 2))
