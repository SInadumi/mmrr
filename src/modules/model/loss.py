import torch

eps = 1e-6


def calc_4d_cost_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes batched the euclidean distance between each pair of the two collections of row vectors.

    cf.) https://pytorch.org/docs/stable/generated/torch.cdist.html
    """
    return torch.cdist(x, y, p=2)  # euclidean


def calc_4d_cost_cosine_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes batched the cosine distance between each pair of the two collections of row vectors.

    cf.) https://github.com/pytorch/pytorch/issues/48306
    """
    # (b, rel, seq, hid)
    x = torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)
    # (b, rel, seq, hid) -> (b, rel, hid, seq)
    y = torch.nn.functional.normalize(y, p=2, dim=-1, eps=eps).permute(0, 1, 3, 2)
    return 1 - torch.matmul(x, y)


def calc_4d_dot_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, y.permute(0, 1, 3, 2))


# soft-cross entropy loss
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


class SupConLoss:
    def __init__(self):
        self.temperature: float = 0.1
        self.dist_func = calc_4d_cost_cosine_matrix

    def compute_loss(
        self,
        h_src: torch.Tensor,  # (b, seq, rel, hid)
        t_src: torch.Tensor,  # (b, seq, rel, hid)
        mask: torch.Tensor,  # (b, rel, seq, seq)
    ):
        # https://github.com/VICO-UoE/CIN-SSL/blob/main/models/losses.py
        # https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/supcon_loss.py
        dist_matrix = self.dist_func(
            h_src.permute(0, 2, 1, 3), t_src.permute(0, 2, 1, 3)
        )  # (b, seq, rel, hid) -> (b, rel, seq, seq)

        # -> (b, rel, seq * seq)
        # cf.) https://github.com/HobbitLong/SupContrast/blob/master/losses.py
        dist_matrix = dist_matrix.view(dist_matrix.shape[0], dist_matrix.shape[1], -1)
        mask = mask.view(mask.shape[0], mask.shape[1], -1)

        pos_mask = mask.eq(1).bool()
        neg_mask = mask.eq(0).bool()

        dist_matrix = dist_matrix / self.temperature
        mat_max, _ = dist_matrix.max(dim=1, keepdim=True)
        dist_matrix = dist_matrix - mat_max.detach()

        denominator = self._logsumexp(
            dist_matrix, keep_mask=(pos_mask + neg_mask).bool(), add_one=False, dim=1
        )
        log_prob = dist_matrix - denominator
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (
            pos_mask.sum(dim=1) + self._small_val(dist_matrix.dtype)
        )
        return (-mean_log_prob_pos[mean_log_prob_pos != 0]).mean()  # avg non-zero

    @staticmethod
    def _small_val(dtype: torch.dtype) -> float:
        return torch.finfo(dtype).tiny

    @staticmethod
    def _neg_inf(dtype: torch.dtype) -> float:
        return torch.finfo(dtype).min

    def _logsumexp(
        self,
        x: torch.Tensor,
        keep_mask: torch.Tensor = None,
        add_one: bool = True,
        dim: int = 1,
    ) -> torch.Tensor:
        # https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/utils/loss_and_miner_utils.py
        if keep_mask is not None:
            x = x.masked_fill(~keep_mask, self._neg_inf(x.dtype))
        if add_one:
            zeros = torch.zeros(
                x.size(dim - 1), dtype=x.dtype, device=x.device
            ).unsqueeze(dim)
            x = torch.cat([x, zeros], dim=dim)

        output = torch.logsumexp(x, dim=dim, keepdim=True)
        if keep_mask is not None:
            output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
        return output
