import torch

from src.mla_mnist.config import ModelConfig
from src.mla_mnist.models.attention import MultiHeadLatentAttention, precompute_rope


def make_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=16,
        num_heads=2,
        mlp_dim=32,
        num_layers=1,
        q_lora_rank=8,
        kv_lora_rank=6,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        attention_dropout=0.0,
        causal_attention=False,
    )


def test_rope_shapes():
    cos, sin = precompute_rope(dim=8, max_seq_len=32)
    assert cos.shape == (32, 8)
    assert sin.shape == (32, 8)


def test_standard_and_absorbed_paths_match():
    torch.manual_seed(0)
    config = make_config()
    attention = MultiHeadLatentAttention(config).eval()
    x = torch.randn(2, 5, config.hidden_size)

    standard, standard_cache = attention(x, use_cache=True)
    absorbed, absorbed_cache = attention.forward_absorbed(x, use_cache=True)

    assert standard.shape == absorbed.shape == (2, 5, config.hidden_size)
    assert standard_cache is not None
    assert absorbed_cache is not None
    torch.testing.assert_close(standard, absorbed, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(standard_cache[0], absorbed_cache[0])
    torch.testing.assert_close(standard_cache[1], absorbed_cache[1])


def test_causal_cache_path_matches_full_forward():
    torch.manual_seed(1)
    config = make_config()
    config.causal_attention = True
    attention = MultiHeadLatentAttention(config).eval()

    x = torch.randn(1, 6, config.hidden_size)

    full, _ = attention(x)
    first, cache = attention(x[:, :4], use_cache=True)
    second, cache2 = attention(x[:, 4:], past_kv=cache, use_cache=True)

    combined = torch.cat([first[:, :], second], dim=1)

    # The first four tokens in a cached decode are identical to the full pass.
    torch.testing.assert_close(combined, full, rtol=1e-4, atol=1e-5)
    assert cache2 is not None


def test_bidirectional_mode_has_no_causal_mask_effect():
    torch.manual_seed(2)
    config = make_config()
    config.causal_attention = False
    attention = MultiHeadLatentAttention(config).eval()

    x = torch.randn(1, 4, config.hidden_size)
    output, _ = attention(x)

    assert output.shape == x.shape
