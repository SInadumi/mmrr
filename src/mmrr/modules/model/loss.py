from typing import Optional

import torch

eps = 1e-6


# softmax cross entropy loss
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
    def __init__(self, margin_pos: float = 1.0, margin_neg: float = 0.0):
        # cf.) https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#contrastiveloss
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg

    def compute_loss(
        self,
        dist_matrix: torch.Tensor,  # (b, rel, seq, seq)
        target: torch.Tensor,  # (b, rel, seq, seq)
    ) -> torch.Tensor:  # ()
        dist_pos = torch.abs(self.margin_pos - dist_matrix).masked_fill(
            target == 0, 0.0
        )
        dist_neg = torch.abs(dist_matrix - self.margin_neg).masked_fill(
            target == 1, 0.0
        )
        pos_loss = dist_pos[dist_pos != 0.0].mean()
        neg_loss = dist_neg[dist_neg != 0.0].mean()
        return pos_loss + neg_loss


class SupConLoss:
    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature

    def compute_loss(
        self,
        dist_matrix: torch.Tensor,  # (b, rel, seq, seq)
        target: torch.Tensor,  # (b, rel, seq, seq)
    ) -> torch.Tensor:  # ()
        # cf.) https://github.com/HobbitLong/SupContrast/blob/master/losses.py
        dist_matrix = dist_matrix.view(dist_matrix.shape[0], dist_matrix.shape[1], -1)
        # -> (b, rel, seq * seq)
        target = target.view(target.shape[0], target.shape[1], -1)

        pos_mask = target.eq(1).bool()
        neg_mask = target.eq(0).bool()

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
        keep_mask: Optional[torch.Tensor] = None,
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
