import typing as tp, torch, torch.nn as nn
from dataclasses import dataclass, field

from .core_vq import ResidualVectorQuantization


@dataclass
class QuantizedResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    penalty: tp.Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer.
    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_dim=self.dimension // 8,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
        )

    def forward(self, x: torch.Tensor) -> QuantizedResult:
        """
        Residual vector quantization on the given input tensor.
        """
        quantized, codes, commit_loss = self.vq(x, n_q=self.n_q)
        return QuantizedResult(quantized, codes, penalty=torch.mean(commit_loss))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input x with the specified frame rate.
        The RVQ encode method returns indices for each quantizer.
        """
        return self.vq.encode(x, n_q=self.n_q)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode the given codes to the quantized representation.
        """
        return self.vq.decode(codes)
