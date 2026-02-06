# RVQ implementation adapted from https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license: MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import typing as tp, warnings

from einops import rearrange, repeat
import torch, torch.nn as nn
import torch.nn.functional as F


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype
    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
        dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EuclideanCodebook(nn.Module):
    """
    Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = (
            uniform_init if not kmeans_init else torch.zeros
        )
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        """
        Initialize the codebook with k-means.
        """
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        # L2-normalize the initialized centroids
        embed = F.normalize(embed, p=2, dim=1)
        
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))

    def deprecated_replace_(self, samples, mask):
        """
        Replace the dead codes with random samples from the current batch.
        """
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def replace_(self, samples, mask):
        """
        Replace dead codes (low EMA usage) with safe vectors sampled from the batch.
        NEW: Apply clipping and normalization to avoid introducing extreme outliers.
        """
        # Sample replacement vectors
        rv = sample_vectors(samples, self.codebook_size)
        # Clamp to a safe range
        rv = torch.clamp(rv, -1.0, 1.0)
        # L2-normalize
        rv = F.normalize(rv, p=2, dim=1)
        # Apply replacements where mask is True
        modified_codebook = torch.where(mask[..., None], rv, self.embed)
        # Update the codebook
        self.embed.data.copy_(modified_codebook)


    def expire_codes_(self, batch_samples):
        """
        Replace codes with cluster size less than threshold (dead).
        """
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flatten all but the last dimensions of the tensor x.
        """
        return rearrange(x, "... d -> (...) d")

    def deprecated_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize input x by computing the squared distance between x and
        the codebook. The index of the closest codebook vector is returned.
        """
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        return dist.max(dim=-1).indices

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize input x by computing cos-sim between x and the codebook.
        """
        # L2-normalize inputs and codebook
        x = F.normalize(x, p=2, dim=1)
        codebook = F.normalize(self.embed, p=2, dim=1)  # [codebook_size, dim]
    
        # Compute cosine similarity [N, codebook_size]
        sim = x @ codebook.t()
        return sim.max(dim=-1).indices

    def postprocess_emb(self, embed_ind, shape):
        """
        Reshape indices to match the given shape.
        """
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind: torch.Tensor) -> torch.Tensor:
        """
        Dequantize embed_ind by looking up the corresponding
        codebook vector for each of the indices in embed_ind.
        """
        return F.embedding(embed_ind, self.embed)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize input x:
        - Flatten all but the last dimensions of x.
        - Compute the squared distance between x and the codebook.
        - Return the index of the closest codebook vector.
        - Reshape indices to match the original shape of x.
        """
        shape = x.shape
        x = self.preprocess(x)
        embed_ind = self.quantize(x)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        """
        Dequantize embed_ind:
        - Look up the codebook vector for each of the indices in embed_ind.
        """
        return self.dequantize(embed_ind)

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        - Initialize the codebook if not done already and quantize x.
        - Get the quantized output and the indices of the codebook vectors.
        - If in training, update the codebook using the exponential moving average.
        """
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)  # flatten all but the last dimensions

        self.init_embed_(x)  # initialize codebook if not already

        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantized = self.dequantize(embed_ind)

        if self.training:  # update codebook using exponential moving average
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            embed_normalized = F.normalize(embed_normalized, p=2, dim=1)  # L2-norm
            self.embed.data.copy_(embed_normalized)

        return quantized, embed_ind


class VectorQuantization(nn.Module):
    """
    Implementation of Vector Quantization [euclidean]
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        )

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        - Swap the last two dimensions of x.
        - Project x to the codebook dimension.
        - Quantize x using the codebook.
        - Return the indices of the quantized vectors.
        """
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_ind = self._codebook.encode(x)
        return embed_ind

    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        """
        - Decode the indices to get the quantized vectors.
        - Project the quantized vectors back to the orig dimension.
        """
        quantized = self._codebook.decode(embed_ind)
        quantized = self.project_out(quantized)
        return rearrange(quantized, "b n d -> b d n")

    def forward(
        self, x: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        - Swap the last two dimensions of x.
        - Project x to the codebook dimension.
        - Quantize x and get the indices of the quantized vectors.
        - Project the quantized vectors back to the orig dimension.
        - If in training, compute the commitment loss, and return.
        """
        device = x.device

        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        quantized, embed_ind = self._codebook(x)

        if self.training:
            # used for the backward pass
            quantized = x + (quantized - x).detach()

        # Compute the commitment loss
        loss = torch.tensor([0.0], device=device, requires_grad=self.training)
        if self.training:
            warnings.warn(
                "When using RVQ in training model, first check "
                "https://github.com/facebookresearch/encodec/issues/25 . "
                "The bug wasn't fixed here for reproducibility."
            )
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantized.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantized = self.project_out(quantized)
        quantized = rearrange(quantized, "b n d -> b d n")

        return quantized, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """
    Implementation of Residual Vector Quantization (RVQ).
    Follows from https://arxiv.org/pdf/2107.03312.pdf.
    """

    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x, n_q: tp.Optional[int] = None):
        """
        Forward pass through all vector quantizers, each
        quantizing the residual from the previous quantizer.
        The output is the sum of all quantized vectors.
        Args:
            x (torch.Tensor): Input tensor.
            n_q (int): Number of residual vector quantizers used."
        """
        quantized_out = 0.0
        residual = x
        n_q = n_q or len(self.layers)

        all_losses, all_indices = [], []
        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        n_q = n_q or len(self.layers)

        all_indices = []
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)

        return torch.stack(all_indices)

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quant_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quant_out = quant_out + quantized
        return quant_out
