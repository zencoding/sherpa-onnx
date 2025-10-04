"""
Lightweight Voting-LFQ (Look-up-Free Quantizer) Implementation

Inspired by StableToken paper (arxiv:2509.22220) but simplified for iOS deployment.
Uses multi-branch voting mechanism for noise-robust semantic tokenization.

Key Features:
- 3 independent voting branches for robustness
- Bit-wise majority voting (not token-level)
- Straight-through estimator for gradient flow
- Learnable codebook with commitment loss
- Lightweight: ~5MB after INT8 quantization

Author: iOS ASR Project
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LightweightVotingLFQ(nn.Module):
    """
    Lightweight Look-up-Free Quantizer with multi-branch voting.

    Converts continuous encoder features to discrete semantic tokens using:
    1. Shared base projection (dimensionality reduction)
    2. Three independent voting branches
    3. Bit-wise majority voting across branches
    4. Binary to integer token conversion

    Args:
        input_dim: Dimension of input features (from Zipformer encoder)
        projection_dim: Dimension of shared base projection
        num_branches: Number of voting branches (default: 3)
        num_bits: Number of bits for quantization (default: 12, i.e., 2^12=4096 tokens)
        commitment_weight: Weight for commitment loss
        dropout: Dropout probability
        init_std: Standard deviation for weight initialization
    """

    def __init__(
        self,
        input_dim: int = 512,
        projection_dim: int = 128,
        num_branches: int = 3,
        num_bits: int = 12,
        commitment_weight: float = 0.25,
        dropout: float = 0.1,
        init_std: float = 0.02,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.num_branches = num_branches
        self.num_bits = num_bits
        self.codebook_size = 2 ** num_bits
        self.commitment_weight = commitment_weight

        # Shared base projection (reduces dimensionality)
        self.base_projection = nn.Linear(input_dim, projection_dim)

        # Three independent voting branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(projection_dim, projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim, num_bits)  # Project to bits
            )
            for _ in range(num_branches)
        ])

        # Learnable codebook centers (for commitment loss)
        self.codebook = nn.Embedding(self.codebook_size, projection_dim)

        # Initialize weights
        self._init_weights(init_std)

    def _init_weights(self, std: float):
        """Initialize model weights with small random values."""
        nn.init.normal_(self.base_projection.weight, mean=0.0, std=std)
        nn.init.zeros_(self.base_projection.bias)

        for branch in self.branches:
            for module in branch:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                    nn.init.zeros_(module.bias)

        nn.init.normal_(self.codebook.weight, mean=0.0, std=std)

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: convert continuous features to discrete tokens.

        Args:
            x: Input features [batch_size, time_steps, input_dim]
            return_all: If True, return additional debugging info

        Returns:
            tokens: Discrete tokens [batch_size, time_steps] (LongTensor)
            commitment_loss: Scalar tensor (commitment loss)
            consensus_score: Scalar tensor (fraction of bits with unanimous agreement)
        """
        batch_size, time_steps, _ = x.shape

        # Shared base projection
        base_features = self.base_projection(x)  # [B, T, projection_dim]

        # Each branch produces bit logits
        branch_logits = []
        for branch in self.branches:
            logits = branch(base_features)  # [B, T, num_bits]
            branch_logits.append(logits)

        # Stack branches: [num_branches, B, T, num_bits]
        branch_logits = torch.stack(branch_logits, dim=0)

        # Binarize each branch using sigmoid + threshold
        # Straight-through estimator: forward uses binary, backward uses continuous
        branch_bits_continuous = torch.sigmoid(branch_logits)
        branch_bits_binary = (branch_bits_continuous > 0.5).float()

        # Straight-through trick: replace forward with binary, keep backward continuous
        branch_bits = branch_bits_binary - branch_bits_continuous.detach() + branch_bits_continuous

        # Bit-wise majority voting across branches
        # vote_sum: [B, T, num_bits], each entry in {0, 1, 2, 3}
        vote_sum = branch_bits.sum(dim=0)  # Sum across branches

        # Majority decision: >=2 votes → 1, else 0 (for 3 branches)
        threshold = (self.num_branches + 1) // 2  # = 2 for 3 branches
        final_bits = (vote_sum >= threshold).float()  # [B, T, num_bits]

        # Convert binary vector to integer token
        # Binary weights: [2^(num_bits-1), 2^(num_bits-2), ..., 2^1, 2^0]
        bit_weights = 2 ** torch.arange(
            self.num_bits - 1, -1, -1,
            device=x.device,
            dtype=torch.float32
        )  # [num_bits]

        # Compute tokens: sum of weighted bits
        tokens = (final_bits * bit_weights).sum(dim=-1).long()  # [B, T]

        # Quantize base features to nearest codebook vector
        quantized = self.codebook(tokens)  # [B, T, projection_dim]

        # Commitment loss: encourage encoder to produce quantizable features
        commitment_loss = F.mse_loss(base_features, quantized.detach())

        # Consensus score: fraction of bits where all branches agree
        # All agree if vote_sum == 0 (all 0) OR vote_sum == num_branches (all 1)
        all_agree = (vote_sum == 0) | (vote_sum == self.num_branches)
        consensus_score = all_agree.float().mean()

        # Weighted commitment loss
        weighted_commitment_loss = self.commitment_weight * commitment_loss

        if return_all:
            # Return additional info for debugging
            return {
                'tokens': tokens,
                'commitment_loss': weighted_commitment_loss,
                'consensus_score': consensus_score,
                'branch_bits': branch_bits,  # [num_branches, B, T, num_bits]
                'final_bits': final_bits,    # [B, T, num_bits]
                'vote_sum': vote_sum,        # [B, T, num_bits]
                'base_features': base_features,  # [B, T, projection_dim]
                'quantized': quantized,      # [B, T, projection_dim]
            }

        return tokens, weighted_commitment_loss, consensus_score

    def get_codebook_usage(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute codebook usage statistics.

        Args:
            tokens: Discrete tokens [batch_size, time_steps]

        Returns:
            usage: Fraction of codebook entries used [codebook_size]
        """
        unique_tokens = torch.unique(tokens)
        usage = torch.zeros(self.codebook_size, device=tokens.device)
        usage[unique_tokens] = 1.0
        return usage.mean()

    def extra_repr(self) -> str:
        """String representation of module parameters."""
        return (
            f"input_dim={self.input_dim}, "
            f"projection_dim={self.projection_dim}, "
            f"num_branches={self.num_branches}, "
            f"num_bits={self.num_bits}, "
            f"codebook_size={self.codebook_size}, "
            f"commitment_weight={self.commitment_weight}"
        )


class TokenEmbedding(nn.Module):
    """
    Learnable token embedding layer.

    Maps discrete tokens to continuous embeddings.

    Args:
        vocab_size: Size of token vocabulary (default: 4096)
        embedding_dim: Dimension of embeddings (default: 256)
        dropout: Dropout probability
        init_std: Standard deviation for weight initialization
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        embedding_dim: int = 256,
        dropout: float = 0.1,
        init_std: float = 0.02,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: map tokens to embeddings.

        Args:
            tokens: Discrete tokens [batch_size, time_steps] (LongTensor)

        Returns:
            embeddings: Continuous embeddings [batch_size, time_steps, embedding_dim]
        """
        embeddings = self.embedding(tokens)  # [B, T, embedding_dim]
        embeddings = self.dropout(embeddings)
        return embeddings

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim}"


class AdapterProjection(nn.Module):
    """
    Adapter projection layer.

    Projects token embeddings to Zipformer decoder input dimension.

    Args:
        input_dim: Input dimension (from embeddings, default: 256)
        output_dim: Output dimension (Zipformer decoder input, default: 512)
        use_layer_norm: Whether to use layer normalization
        activation: Activation function ('relu', 'gelu', 'silu')
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 512,
        use_layer_norm: bool = True,
        activation: str = 'relu',
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Linear projection
        self.projection = nn.Linear(input_dim, output_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: project embeddings to decoder input dimension.

        Args:
            embeddings: Token embeddings [batch_size, time_steps, input_dim]

        Returns:
            output: Projected features [batch_size, time_steps, output_dim]
        """
        output = self.projection(embeddings)  # [B, T, output_dim]
        output = self.layer_norm(output)
        output = self.activation(output)
        output = self.dropout(output)
        return output

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"


class MelDecoder(nn.Module):
    """
    Mel spectrogram decoder for TTS validation.

    Decodes token embeddings back to mel spectrograms to validate
    that tokens preserve acoustic information for TTS.

    NOTE: This module is ONLY used during training for validation.
          It is NOT exported to iOS.

    Args:
        input_dim: Input dimension (from embeddings, default: 256)
        mel_dim: Number of mel bins (default: 80)
        hidden_dim: Hidden layer dimension (default: 512)
        num_layers: Number of decoder layers (default: 2)
        use_layer_norm: Whether to use layer normalization
        activation: Activation function
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 256,
        mel_dim: int = 80,
        hidden_dim: int = 512,
        num_layers: int = 2,
        use_layer_norm: bool = True,
        activation: str = 'relu',
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Build decoder layers
        layers = []

        # First layer: input_dim → hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(self._get_activation(activation))
        layers.append(nn.Dropout(dropout))

        # Hidden layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))

        # Output layer: hidden_dim → mel_dim
        layers.append(nn.Linear(hidden_dim, mel_dim))

        self.decoder = nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'silu':
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: decode embeddings to mel spectrograms.

        Args:
            embeddings: Token embeddings [batch_size, time_steps, input_dim]

        Returns:
            mel: Mel spectrogram [batch_size, time_steps, mel_dim]
        """
        mel = self.decoder(embeddings)  # [B, T, mel_dim]
        return mel

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, mel_dim={self.mel_dim}, "
            f"hidden_dim={self.hidden_dim}, num_layers={self.num_layers}"
        )


# Factory function for creating model from config
def create_voting_lfq_from_config(config: dict) -> LightweightVotingLFQ:
    """
    Create Voting-LFQ module from configuration dict.

    Args:
        config: Configuration dictionary (from model_config.yaml)

    Returns:
        voting_lfq: Initialized Voting-LFQ module
    """
    voting_lfq_config = config.get('voting_lfq', {})
    return LightweightVotingLFQ(
        input_dim=voting_lfq_config.get('input_dim', 512),
        projection_dim=voting_lfq_config.get('projection_dim', 128),
        num_branches=voting_lfq_config.get('num_branches', 3),
        num_bits=voting_lfq_config.get('num_bits', 12),
        commitment_weight=voting_lfq_config.get('commitment_weight', 0.25),
        dropout=voting_lfq_config.get('dropout', 0.1),
        init_std=voting_lfq_config.get('init_std', 0.02),
    )


def create_token_embedding_from_config(config: dict) -> TokenEmbedding:
    """Create TokenEmbedding module from configuration dict."""
    embedding_config = config.get('embedding', {})
    return TokenEmbedding(
        vocab_size=embedding_config.get('vocab_size', 4096),
        embedding_dim=embedding_config.get('embedding_dim', 256),
        dropout=embedding_config.get('dropout', 0.1),
        init_std=embedding_config.get('init_std', 0.02),
    )


def create_adapter_from_config(config: dict) -> AdapterProjection:
    """Create AdapterProjection module from configuration dict."""
    adapter_config = config.get('adapter', {})
    return AdapterProjection(
        input_dim=adapter_config.get('input_dim', 256),
        output_dim=adapter_config.get('output_dim', 512),
        use_layer_norm=adapter_config.get('use_layer_norm', True),
        activation=adapter_config.get('activation', 'relu'),
        dropout=adapter_config.get('dropout', 0.1),
    )


def create_mel_decoder_from_config(config: dict) -> MelDecoder:
    """Create MelDecoder module from configuration dict."""
    mel_decoder_config = config.get('mel_decoder', {})
    return MelDecoder(
        input_dim=mel_decoder_config.get('input_dim', 256),
        mel_dim=mel_decoder_config.get('mel_dim', 80),
        hidden_dim=mel_decoder_config.get('hidden_dim', 512),
        num_layers=mel_decoder_config.get('num_layers', 2),
        use_layer_norm=mel_decoder_config.get('use_layer_norm', True),
        activation=mel_decoder_config.get('activation', 'relu'),
        dropout=mel_decoder_config.get('dropout', 0.1),
    )
