import math, torch, hydra
import torch.nn as nn
from biocodec.model import BioCodecModel
from biocodec.utils import count_parameters


def sinusoidal_position_encoding(T: int, d_model: int) -> torch.Tensor:
    """
    Create fixed sinusoidal positional encodings.
    Returns a tensor of shape (T, d_model).
    """
    pe = torch.zeros(T, d_model)
    position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class BioCodecFT(nn.Module):
    def __init__(
        self,
        config,
        C: int,
        T: int,
        num_classes: int,
        n_books: int = 6,
        n_used: int = 6,
        n_bins: int = 256,
        d_model: int = 16,
        n_heads: int = 8,
        n_layers: int = 1,
        dropout: float = 0.2,
        is_emg: bool = False,
    ):
        """
        Transformer-based classifier for RVQ-encoded EEG/EMG.
        Args:
            C           : number of channels
            T           : number of time steps
            num_classes : number of target classes
            n_books     : number of codebooks
            n_bins      : codebook size
            d_model     : embedding dimension
            n_heads     : self-attention heads
            n_layers    : Transformer encoder layers
            dropout     : dropout probability
            is_emg      : whether the signal is EMG
        """
        super().__init__()

        # Model parameters
        self.config = config
        self.C, self.T = C, T
        self.n_books, self.n_used = n_books, n_used
        self.d_model = d_model #* self.n_used
        self.linear_dim = self.d_model * 4
        self.is_emg = is_emg

        # Look-up embeddings for RVQ codes
        self.code_embs = nn.ModuleList(
            [nn.Embedding(n_bins, d_model, max_norm=1.0) for _ in range(n_books)]
        )
        self.load_codec_and_init()

        # Sinusoidal positional encoding
        pe_t = sinusoidal_position_encoding(self.T, self.d_model)
        self.register_buffer("t_enc", pe_t)
        pe_c = sinusoidal_position_encoding(self.C, self.d_model)
        self.register_buffer("c_enc", pe_c)

        # Transformer encoders
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.linear_dim,
            activation="gelu",
            dropout=dropout,
            batch_first=True,
        )
        self.time_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.chan_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.d_model * self.C, self.linear_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(self.linear_dim, num_classes),
        )

    def _init_from_decoder(self, codec_model, method="concat"):
        """
        Initialize code embeddings from quantizer codebook vectors
        """
        assert method == "concat"
        codebook = [cb.codebook for cb in codec_model.quantizer.vq.layers]

        for i in range(self.n_books):
            cb = codebook[min(i, len(codebook) - 1)]
            cb -= cb.mean(dim=0, keepdim=True)
            cb /= cb.std(dim=0, keepdim=True) + 1e-8
            cb /= cb.norm(dim=1, keepdim=True) + 1e-8

            self.code_embs[i].weight.data.copy_(cb)

    def load_codec_and_init(self):
        """
        Load a pretrained codec model and initialize embeddings from it
        """
        codec_model = BioCodecModel._get_optimized_model(
            sample_rate=self.config.model.sample_rate,
            causal=self.config.pretrained.causal,
            model_norm=self.config.pretrained.norm,
            signal_normalize=self.config.pretrained.normalize,
            segment=eval(self.config.pretrained.segment),
            name=self.config.pretrained.name,
            n_q=self.config.pretrained.n_q,
            q_bins=self.config.pretrained.q_bins,
        ) if not self.is_emg else BioCodecModel._get_emg_model(
            sample_rate=self.config.model.sample_rate,
            causal=self.config.pretrained.causal,
            model_norm=self.config.pretrained.norm,
            signal_normalize=self.config.pretrained.normalize,
            segment=eval(self.config.pretrained.segment),
            name=self.config.pretrained.name,
            n_q=self.config.pretrained.n_q,
            q_bins=self.config.pretrained.q_bins,
        )
        # load state dict from codec path
        checkpoint = torch.load(self.config.common.codec_path, map_location="cuda")
        checkpoint["model_state_dict"] = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
        }
        codec_model.load_state_dict(checkpoint["model_state_dict"])
        codec_model = codec_model.cuda()
        self._init_from_decoder(codec_model)

    def forward(self, codes: torch.LongTensor) -> torch.Tensor:
        """
        Args: RVQ codes (B, C, T, n_books) in [0, n_bins)
        Returns: logits (B, num_classes)
        """
        B = codes.size(0)
        T = codes.size(2)
        codes = codes[:, :, T - self.T :]

        # 1) Embed the 256 codes:
        codes = codes[..., : self.n_used]
        x = [
            self.code_embs[i](codes[:, :, :, i])  # (B, C, T, d_model)
            for i in range(self.n_used)
        ]
        x = torch.sum(x, dim=-1)  # (B, C, T, n_books*d_model=K)

        # 2) Transformer on dim T:
        x = x.reshape(B * self.C, self.T, -1)  # (B*C, T, K)
        # Positional encoding for T:
        time_emb = self.t_enc.expand(B * self.C, -1, -1)
        x = self.time_transformer(x + time_emb)  # (B*C, T, K)

        # reshape back
        x = x.reshape(B, self.C, self.T, -1)  # (B, C, T, K)
        x = x.permute(0, 2, 1, 3)  # (B, T, C, K)

        # 3) Transformer on dim C:
        x = x.reshape(B * self.T, self.C, -1)  # (B*T, C, K)
        # Positional encoding for C:
        chan_emb = self.c_enc.expand(B * self.T, -1, -1)
        x = self.chan_transformer(x + chan_emb)  # (B*T, C, K)

        # 4) Classification head over C:
        x = x.reshape(B, self.T, self.C, -1).mean(dim=1).view(B, -1)
        return self.classifier(x)  # (B, num_classes)


@hydra.main(config_path="configs", config_name="ft_config")
def main(cfg):
    # Example config
    B, C, T, nc = 16, 2, 105 * 6, 5
    # Example RVQ codes
    n_books, n_used = 6, 6
    # Instantiate the model
    model = BioCodecFT(
        config=cfg,
        C=C,
        T=T,
        num_classes=nc,
        n_books=n_books,
        n_used=n_used,
        n_bins=256,
        d_model=16,
        n_heads=8,
        n_layers=1,
        dropout=0.1,
        is_emg=False,
    )
    print("Model parameters:", count_parameters(model))

    # Example RVQ codes
    example_codes = torch.randint(0, 256, (B, C, T, n_books))
    print(example_codes.shape)  # Output: torch.Size([B, C, T, n_books])
    # Test the model function
    logits = model(example_codes)
    print(logits.shape)  # Output: torch.Size([B, nc])


if __name__ == "__main__":
    main()
