import torch

eps = 1e-6


def calc_4d_cost_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes batched the euclidean distance between each pair of the two collections of row vectors.

    c.f.) https://pytorch.org/docs/stable/generated/torch.cdist.html
    """
    return torch.cdist(x, y, p=2)  # euclidean


def calc_4d_cost_cosine_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes batched the cosine distance between each pair of the two collections of row vectors.

    c.f.) https://github.com/pytorch/pytorch/issues/48306
    """
    # (b, rel, seq, hid)
    x = torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)
    # (b, rel, seq, hid) -> (b, rel, hid, seq)
    y = torch.nn.functional.normalize(y, p=2, dim=-1, eps=eps).permute(0, 1, 3, 2)
    return 1 - torch.matmul(x, y)


def cross_entropy_loss(
    output: torch.Tensor,  # (b, rel, seq, seq)
    target: torch.Tensor,  # (b, rel, seq, seq)
    mask: torch.Tensor,  # (b, rel, seq, seq)
) -> torch.Tensor:  # ()
    log_softmax = torch.log_softmax(output, dim=3)  # (b, rel, seq, seq)
    # reduce using masked mean (target ⊆ mask)
    # TODO: 最後の次元についてまず mean を取る（cross entropy の定義）
    return torch.sum(-log_softmax * target * mask).div(torch.sum(target * mask) + eps)


def binary_cross_entropy_with_logits(
    output: torch.Tensor,  # (b, seq, seq)
    target: torch.Tensor,  # (b, seq, seq)
    mask: torch.Tensor,  # (b, seq, seq)
) -> torch.Tensor:  # ()
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        output, target.float(), reduction="none"
    )  # (b, seq, seq)
    # reduce using masked mean
    return torch.sum(losses * mask).div(torch.sum(mask) + eps)


class ContrastiveLoss:
    def __init__(self, dist_func_name="cosine"):
        assert dist_func_name in ["cosine", "euclidean"]
        self.margin_pos: float = -1.0
        self.margin_neg: float = 0.0
        self.flip_matrix: float = -1.0
        if dist_func_name == "cosine":
            # https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#contrastiveloss
            self.dist_func = calc_4d_cost_cosine_matrix
        elif dist_func_name == "euclidean":
            self.margin_pos = 0.0
            self.margin_neg = 1.0
            self.flip_matrix = 1.0
            self.dist_func = calc_4d_cost_matrix

    def compute_loss(
        self,
        h_src: torch.Tensor,  # (b, seq, rel, hid)
        t_src: torch.Tensor,  # (b, seq, rel, hid)
        mask: torch.Tensor,  # (b, rel, seq, seq)
    ) -> torch.Tensor:
        dist_matrix = self.dist_func(
            h_src.permute(0, 2, 1, 3), t_src.permute(0, 2, 1, 3)
        )  # (b, seq, rel, hid) ->(b, rel, seq, seq)
        dist_matrix = self.flip_matrix * dist_matrix

        dist_pos = torch.abs(dist_matrix - self.margin_pos).masked_fill(mask == 0, 0.0)
        dist_neg = torch.abs(self.margin_neg - dist_matrix).masked_fill(mask == 1, 0.0)
        pos_loss = dist_pos[dist_pos != 0.0].mean()
        neg_loss = dist_neg[dist_neg != 0.0].mean()
        return pos_loss + neg_loss
