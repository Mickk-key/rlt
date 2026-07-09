"""RL token encoder-decoder (Section IV-A, Eq. 1-2)."""

from __future__ import annotations

import torch
import torch.nn as nn


class RLTokenEncoderDecoder(nn.Module):
    """Compress VLA final-layer token embeddings into a single RL token z_rl.

    Architecture follows the paper: lightweight encoder/decoder transformers with
    a learned special token e_rl appended to the VLA embedding sequence.
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        token_dim: int = 2048,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 4096,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.token_dim = token_dim

        self.rl_token_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        if token_dim != embed_dim:
            self.token_proj = nn.Linear(embed_dim, token_dim)
        else:
            self.token_proj = nn.Identity()

    def encode(self, vla_embeddings: torch.Tensor) -> torch.Tensor:
        """Extract RL token from VLA embeddings.

        Args:
            vla_embeddings: (B, M, D) final-layer VLA token embeddings z_{1:M}
        Returns:
            z_rl: (B, token_dim)
        """
        batch = vla_embeddings.shape[0]
        rl_tok = self.rl_token_embed.expand(batch, -1, -1)
        seq = torch.cat([vla_embeddings, rl_tok], dim=1)
        encoded = self.encoder(seq)
        z_rl = encoded[:, -1, :]
        return self.token_proj(z_rl)

    def decode_step(
        self,
        z_rl: torch.Tensor,
        prefix: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive decode one embedding step (teacher forcing during training)."""
        # prefix: (B, i, D) already-decoded tokens including z_rl at position 0
        memory = z_rl.unsqueeze(1)
        out = self.decoder(prefix, memory)
        return self.output_proj(out[:, -1:, :])

    def reconstruction_loss(self, vla_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute L_ro from Eq. (2) with stop-gradient on VLA embeddings."""
        target = vla_embeddings.detach()
        z_rl = self.encode(vla_embeddings)

        batch, seq_len, dim = target.shape
        total_loss = torch.zeros((), device=target.device, dtype=target.dtype)
        # Decoder input: [z_rl, z_1, ..., z_{i-1}]
        dec_in = z_rl.unsqueeze(1)
        for i in range(seq_len):
            pred = self.decode_step(z_rl, dec_in)
            total_loss = total_loss + (pred.squeeze(1) - target[:, i, :]).pow(2).mean()
            dec_in = torch.cat([dec_in, target[:, i : i + 1, :]], dim=1)

        return total_loss / seq_len, z_rl

    def forward(self, vla_embeddings: torch.Tensor) -> torch.Tensor:
        return self.encode(vla_embeddings)


def load_rl_token_state_dict(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    weights_only: bool = True,
) -> dict[str, torch.Tensor]:
    """Load RL token weights from flat or artifact-style ``rl_token.pt``."""
    obj = torch.load(path, map_location=map_location, weights_only=weights_only)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported rl_token checkpoint format: {path}")
