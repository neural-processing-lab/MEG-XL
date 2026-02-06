import typing as tp
import torch, torch.nn as nn

from . import modules as m
from . import quantization as qt
from .utils import _linear_overlap_add

EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]


class BioCodecModel(nn.Module):
    """BioCodec model operating on raw waveforms.
    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        sample_rate (int): Signal sample rate.
        channels (int): Number of signal channels.
        normalize (bool): Whether to apply signal normalization.
        segment (float or None): segment duration in sec (overlap-add).
        overlap (float): overlap between segment (fraction of the segment duration).
        name (str): name of the model, used as metadata.
    """

    def __init__(
        self,
        encoder: m.SEANetEncoder,
        decoder: m.SEANetDecoder,
        quantizer: qt.ResidualVectorQuantizer,
        sample_rate: int,
        channels: int,
        normalize: bool = False,
        segment: tp.Optional[float] = None,
        overlap: float = 0.01,
        name: str = "unset",
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.segment = segment
        self.overlap = overlap
        self.name = name

    @property
    def segment_length(self) -> tp.Optional[int]:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)

    @property
    def segment_stride(self) -> tp.Optional[int]:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap) * segment_length))

    def encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        """
        Given a tensor `x`, returns a list of frames containing
        the discrete codes for `x`, along with rescaling factors
        for each segment, when `self.normalize` is True.

        Each frames is a tuple `(codebook, scale)`, with `codebook`
        of shape `[B, K, T]`, with `K` the number of codebooks.
        """
        assert x.dim() == 3
        _, channels, length = x.shape
        assert channels > 0 and channels <= 2

        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride
            assert stride is not None

        encoded_frames: tp.List[EncodedFrame] = []
        for offset in range(0, length, stride):
            frame = x[:, :, offset : offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames

    def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
        length = x.shape[-1]
        duration = length / self.sample_rate
        assert self.segment is None or duration <= 1e-5 + self.segment

        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = torch.clamp(volume, min=1e-8)  # numerical stability
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        emb = self.encoder(x)
        if self.training:
            return emb, scale
        else:
            codes = self.quantizer.encode(emb)
            codes = codes.transpose(0, 1)
            # codes is [B, K, T], with T frames, K codebooks.
            return codes, scale

    def decode(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        """
        Decode the given (quantized) frames into a waveform.
        Output might be bigger than the input => just trim.
        """
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])

        frames = [self._decode_frame(frame) for frame in encoded_frames]
        return _linear_overlap_add(frames, self.segment_stride or 1)

    def _decode_frame(self, encoded_frame: EncodedFrame) -> torch.Tensor:
        codes, scale = encoded_frame
        if self.training:
            emb = codes
        else:
            codes = codes.transpose(0, 1)  # [B, K, T] --> [K, B, T]
            emb = self.quantizer.decode(codes)

        out = self.decoder(emb)
        if scale is not None:
            out = out * scale.view(-1, 1, 1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input_wav -> encoder [B, 1, T]
        frames = self.encode(x)

        if self.training:
            # embedding -> quantizer FORWARD -> decode
            loss_w = torch.zeros(1, device=x.device, requires_grad=True)
            # self.quantizer.train(self.training)
            all_codes = []
            for codes, scale in frames:
                qv = self.quantizer(codes)
                loss_w = loss_w + qv.penalty  # loss_w is the RVQ commit loss
                all_codes.append((qv.quantized, scale))

            return self.decode(all_codes)[:, :, : x.shape[-1]], loss_w, frames
        else:
            # embedding -> quantizer ENCODE -> decode
            return self.decode(frames)[:, :, : x.shape[-1]]

    @staticmethod
    def _get_optimized_model(
        sample_rate: int = 250,
        channels: int = 1,
        causal: bool = True,
        model_norm: str = "weight_norm",
        signal_normalize: bool = True,
        segment: tp.Optional[float] = None,
        name: str = "eeg_optimized",
        n_q: int = 6,
        q_bins: int = 256,
    ):
        """
        EEG-optimized model with better frequency preservation.
        """
        encoder = m.SEANetEncoder(
            channels=channels,
            norm=model_norm,
            causal=causal,
            ratios=[3, 2, 2],
            n_residual_layers=2,
            true_skip=True,
            compress=1,
        )
        decoder = m.SEANetDecoder(
            channels=channels,
            norm=model_norm,
            causal=causal,
            ratios=[3, 2, 2],
            n_residual_layers=2,
            true_skip=True,
            compress=1,
        )
        quantizer = qt.ResidualVectorQuantizer(
            dimension=encoder.dimension,
            n_q=n_q,
            bins=q_bins,
        )
        return BioCodecModel(
            encoder,
            decoder,
            quantizer,
            sample_rate,
            channels,
            normalize=signal_normalize,
            segment=segment,
            name=name,
        )

    @staticmethod
    def _get_emg_model(
        sample_rate: int = 1000,
        channels: int = 1,
        causal: bool = True,
        model_norm: str = "weight_norm",
        signal_normalize: bool = True,
        segment: tp.Optional[float] = None,
        name: str = "emg_optimized",
        n_q: int = 6,
        q_bins: int = 256,
    ):
        encoder = m.SEANetEncoder(
            channels=channels,
            norm=model_norm,
            causal=causal,
            ratios=[3, 3, 2],
            n_residual_layers=2,
            true_skip=True,
            compress=1,
        )
        decoder = m.SEANetDecoder(
            channels=channels,
            norm=model_norm,
            causal=causal,
            ratios=[3, 3, 2],
            n_residual_layers=2,
            true_skip=True,
            compress=1,
        )
        quantizer = qt.ResidualVectorQuantizer(
            dimension=encoder.dimension,
            n_q=n_q,
            bins=q_bins,
        )
        return BioCodecModel(
            encoder,
            decoder,
            quantizer,
            sample_rate,
            channels,
            normalize=signal_normalize,
            segment=segment,
            name=name,
        )


if __name__ == "__main__":
    # Load BioCodec model checkpoint
    model = BioCodecModel._get_optimized_model()
    checkpoint = torch.load("./brainstorm/tokenizers/biocodec_ckpt.pt", map_location="cuda")

    # Rename keys to remove _orig_mod prefix
    new_state_dict = {}
    for key, value in checkpoint["model_state_dict"].items():
        if key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod.") :]
        else:
            new_key = key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)

    model.eval()

    meg_sensor = torch.randn(16, 1, 1000 * 5)
    print("Sample input shape:", meg_sensor.shape)
    
    codes = model.encode(meg_sensor)
    codes = torch.stack([c[0] for c in codes], dim=0)
    print("Quantized embedding shape:", codes[0].shape)
    
    output = model(meg_sensor)
    print("Reconstructed shape:", output.shape)

    breakpoint()
