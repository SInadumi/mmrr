import torch

eps = 1e-6


def cross_entropy_loss(
    output: torch.Tensor,  # (b, rel, seq1, seq2)
    target: torch.Tensor,  # (b, rel, seq1, seq2)
    mask: torch.Tensor,  # (b, rel, seq1, seq2)
) -> torch.Tensor:  # ()
    log_softmax = torch.log_softmax(output, dim=3)  # (b, rel, seq1, seq2)
    # reduce using masked mean (target ⊆ mask)
    # TODO: 最後の次元についてまず mean を取る（cross entropy の定義）
    return torch.sum(-log_softmax * target * mask).div(torch.sum(target * mask) + eps)


def binary_cross_entropy_with_logits(
    output: torch.Tensor,  # (b, seq1, seq2)
    target: torch.Tensor,  # (b, seq1, seq2)
    mask: torch.Tensor,  # (b, seq1, seq2)
) -> torch.Tensor:  # ()
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        output, target.float(), reduction="none"
    )  # (b, seq1, seq2)
    # reduce using masked mean
    return torch.sum(losses * mask).div(torch.sum(mask) + eps)
