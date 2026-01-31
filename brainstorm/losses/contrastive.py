"""Thanks to Stéphane d'Ascoli for sharing this code."""

import torch
from torch import nn
from torch.nn import functional as F

class ClipLoss(nn.Module):
    """CLIP constrastive loss.

    Contrastive Language-Image Pretraining (CLIP) loss from [1]_. Default values reflect the
    configuration of the CLIP loss used in [2]_.

    Parameters
    ----------
    norm_kind :
        How to normalize the estimates and/or candidates before computing their dot products.
            'x': normalize estimates only.
            'y': normalize candidates only (approach originally used in brainmagick).
            'xy': normalize both estimates and candidates.
            None: do not normalize.
    temperature :
        If True, use learnable temperature parameter.
    symmetric :
        If True, compute loss in both retrieval directions, i.e. retrieve candidates given
        estimates and retrieve estimates given candidates (requires estimates and candidates to be
        of the same shape). If False, only do the former.

    References
    ----------
    .. [1] Radford, Alec, et al. "Learning transferable visual models from natural language
        supervision." International conference on machine learning. PMLR, 2021.
    .. [2] Défossez, Alexandre, et al. "Decoding speech perception from non-invasive brain
        recordings." Nature Machine Intelligence (2023): 1-11.
    """

    def __init__(
        self,
        norm_kind: str | None = "y",
        temperature: bool = True,
        symmetric: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        self.norm_kind = norm_kind
        # FSDP requires parameters to be 1D tensors, not scalars
        self.temperature = (
            nn.Parameter(torch.tensor([1 / 0.07]).log())
            if temperature
            else nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        )
        self.symmetric = symmetric
        self.reduction = reduction

    @staticmethod
    def _compute_similarity(
        x: torch.Tensor, y: torch.Tensor, norm: str | None = None, eps=1e-15
    ) -> torch.Tensor:
        if norm is None:
            eq, inv_norms = "b", torch.ones(x.shape[0])
        elif norm == "x":
            eq, inv_norms = "b", 1 / (eps + x.norm(dim=(1), p=2))
        elif norm == "y":
            eq, inv_norms = "o", 1 / (eps + y.norm(dim=(1), p=2))
        elif norm == "xy":
            eq = "bo"
            inv_norms = 1 / (
                eps + torch.outer(x.norm(dim=(1), p=2), y.norm(dim=(1), p=2))
            )
        else:
            raise ValueError(f"norm must be None, x, y or xy, got {norm}.")

        # Normalize inside einsum to avoid creating a copy of candidates which can be pretty big
        return torch.einsum(f"bc,oc,{eq}->bo", x, y, inv_norms)

    def get_scores(self, estimate: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        """Given estimates of shape [B, F] and candidates of shape [B', F], return a [B, B'] matrix
        of similarity scores.
        """
        scores = self._compute_similarity(estimate, candidate, norm=self.norm_kind)
        scores = self.temperature.exp() * scores
        return scores

    def get_probabilities(
        self, estimate: torch.Tensor, candidate: torch.Tensor
    ) -> torch.Tensor:
        """Given estimates of shape [B, F] and candidates of shape [B', F], return a [B, B'] matrix
        of matching probability.
        """
        scores = self.get_scores(estimate, candidate)
        return F.softmax(scores, dim=1)

    def forward(self, estimate: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        """Warning: estimate and candidate are not necessarily symmetrical.

        If estimate of shape [B, C] and candidate of shape [B', C] with B'>=B, the first B samples
        of candidate are targets, while the remaining B'-B samples of candidate are only used as
        negatives.
        """
        assert estimate.size(0) <= candidate.size(
            0
        ), "need at least as many targets as estimates"
        scores = self.get_scores(estimate, candidate)
        target = torch.arange(len(scores), device=estimate.device)

        loss_e = F.cross_entropy(scores, target, reduction=self.reduction)
        if self.symmetric:
            assert (
                scores.shape[0] == scores.shape[1]
            ), "need square score matrix for symmetric loss"
            loss_c = F.cross_entropy(
                scores.transpose(1, 0), target, reduction=self.reduction
            )
            loss = (loss_e + loss_c) / 2
        else:
            loss = loss_e
        return loss


class SigLipLoss(ClipLoss):
    """SigLIP contrastive loss.

    Sigmoid loss for Language-Image Pretraining (SigLIP) from [1]_.

    Parameters
    ----------
    norm_kind :
        How to normalize the estimates and/or candidates before computing their dot products.
            'x': normalize estimates only.
            'y': normalize candidates only (approach originally used in brainmagick).
            'xy': normalize both estimates and candidates.
            None: do not normalize.
    temperature :
        If True, use learnable temperature parameter. Initialized to ln(10).
    bias :
        If True, use learnable bias parameter, initalized to -10 as most samples are negative.
    identical_candidates_threshold :
        If given, estimates are matched not only to their candidate, but all candidates
        that have a large cosine similarity to their candidate (larger or equal this threshold).
        Assumes such other candidates with high cosine similarity are duplicates.
        Intended to use only if candidate generator is frozen.

    References
    ----------
    .. [1] Zhai, Xiaohua, et al. "Sigmoid loss for language image pre-training." arXiv preprint
        arXiv:2303.15343 (2023).

    Note
    ----
    Official jax implementation: https://github.com/google-research/big_vision/blob/474dd2ebde37268db4ea44decef14c7c1f6a0258/big_vision/trainers/proj/image_text/siglip.py
    """

    def __init__(
        self,
        norm_kind: str | None = "y",
        temperature: bool = True,
        bias: bool = True,
        identical_candidates_threshold: float | None = 0.999,
        reduction: str = "sum",
    ):
        super().__init__(
            norm_kind=norm_kind, temperature=False, symmetric=True, reduction=reduction
        )
        # FSDP requires parameters to be 1D tensors, not scalars
        self.temperature = (
            nn.Parameter(torch.tensor([10.0]).log())
            if temperature
            else nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        )
        self.bias = (
            nn.Parameter(torch.tensor([-10.0]))
            if bias
            else nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        )
        self.identical_candidates_threshold = identical_candidates_threshold

    def get_scores(self, estimate: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        """Given estimates of shape [B, F] and candidates of shape [B', F], return a [B, B'] matrix
        of similarity scores.
        """
        return super().get_scores(estimate, candidate) + self.bias

    def forward(self, estimate: torch.Tensor, candidate: torch.Tensor, reweigh_positives: bool) -> torch.Tensor:
        assert estimate.size(0) <= candidate.size(
            0
        ), "need at least as many targets as estimates"
        scores = self.get_scores(estimate, candidate)
        if self.identical_candidates_threshold is not None:
            candidate_sim = self._compute_similarity(
                candidate, candidate, "xy", eps=1e-15
            )
            targets = 1.0 * (candidate_sim >= self.identical_candidates_threshold)
            targets = targets[: len(estimate)]
            if reweigh_positives:
                weights = 1.0 * (candidate_sim >= self.identical_candidates_threshold)
                weights = 1 - weights  # remove all duplicates
                weights += torch.eye(
                    *weights.shape, device=weights.device
                )  # keep only one
            else:
                weights = None
        else:
            weights = None
            targets = torch.eye(*scores.shape, device=scores.device)
        loss = F.binary_cross_entropy_with_logits(
            scores, targets, weights, reduction=self.reduction
        ) / len(scores)
        return loss
