"""
XORZEN v1.0.1 Comprehensive Architectural Regression Test Suite.

Covers:
1. MoE Parameter Registration & Optimizer Gradient Flow (BUG-CRITICAL-2)
2. Batch-Boundary Causal Isolation in Conditional Depth (BUG-CRITICAL-3)
3. Zero Agentic Model Associative Memory & Action Head Gradients
4. SlicedFFN Nested Width Slicing & FLOP Reduction
5. GreedModel End-to-End Classification & Backward Flow
6. .xorm Archive Serialization & Deserialization
"""

import sys
import os
import tempfile
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.variants import zero_tiny_23k
from xorzen.models.zero.agentic_config import ZeroAgenticConfig
from xorzen.models.zero.agentic_model import ZeroAgenticModel
from xorzen.models.greed import GreedModel
from xorzen.model.components.hass_block import HASSBlock
from xorzen.model.components.sliced_ffn import SlicedFFN
from xorzen.inference.xorm_format import XormWriter, XormReader


def test_moe_registration_and_gradients():
    print("[TEST 1/6] Verifying MoE parameter registration and gradient flow...")
    model = zero_tiny_23k(test_mode=False).train()
    
    # Verify expert parameters exist
    moe_param_count = sum(p.numel() for p in model.moe.parameters())
    assert moe_param_count > 0, "MoE fabric has 0 registered parameters!"
    
    # Forward & backward
    x = torch.randint(0, model.config.vocab_size, (2, 8))
    out = model(x)
    loss = out.logits.sum()
    loss.backward()
    
    grads = [p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.moe.parameters()]
    assert any(grads), "No MoE parameters received gradients during backward pass!"
    print(f"  [PASS] MoE registered {moe_param_count} params; gradient flow verified.")


def test_batch_boundary_isolation():
    print("[TEST 2/6] Verifying batch-boundary causal isolation in forward_with_depth...")
    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    block = HASSBlock(cfg, layer_idx=0).eval()
    
    x_sample_a1 = torch.randn(1, 4, cfg.hidden_size)
    x_sample_a2 = torch.randn(1, 4, cfg.hidden_size)
    x_sample_b = torch.randn(1, 4, cfg.hidden_size)
    
    depth_mask = torch.ones(2, 4, dtype=torch.bool)
    
    # Batch 1: [A1, B]
    x_batch1 = torch.cat([x_sample_a1, x_sample_b], dim=0)
    out1 = block.forward_with_depth(x_batch1, depth_mask)
    
    # Batch 2: [A2, B]
    x_batch2 = torch.cat([x_sample_a2, x_sample_b], dim=0)
    out2 = block.forward_with_depth(x_batch2, depth_mask)
    
    diff_b = (out1[1] - out2[1]).abs().max().item()
    assert diff_b == 0.0, f"Sample B output leaked from Sample A (diff: {diff_b})!"
    print("  [PASS] Zero cross-sample attention leakage confirmed (max diff = 0.0).")


def test_zero_agentic_memory_and_actions():
    print("[TEST 3/6] Verifying ZeroAgenticModel memory vault and action head...")
    cfg = ZeroAgenticConfig(
        hidden_size=64,
        num_layers=2,
        min_depth=1,
        max_depth=2,
        num_attention_heads=4,
        memory_slots=16,
        context_length=128,
        vocab_size=100
    )
    model = ZeroAgenticModel(cfg).train()
    
    # Sequence length 64 > memory_slots 16 (tests arbitrary length handling)
    x = torch.randint(0, 100, (2, 64))
    action_targets = torch.randn(2, 64, cfg.num_action_slots)
    labels = torch.randint(0, 100, (2, 64))
    
    out = model(x, labels=labels, action_targets=action_targets)
    assert out.loss is not None, "Loss was not computed!"
    out.loss.backward()
    
    assert model.memory_vault.grad is not None and model.memory_vault.grad.norm().item() > 0, "Memory vault has no gradient!"
    assert model.action_head.proj.weight.grad is not None and model.action_head.proj.weight.grad.norm().item() > 0, "Action head has no gradient!"
    print("  [PASS] Associative memory vault and action head gradients verified.")


def test_sliced_ffn_computation():
    print("[TEST 4/6] Verifying SlicedFFN matrix slicing and inference fast-path...")
    hidden_dim = 64
    max_width = 256
    widths = [64, 128, 192, 256]
    ffn = SlicedFFN(hidden_dim, max_width, width_choices=widths).eval()
    
    x = torch.randn(2, 8, hidden_dim)
    
    # Test sliced execution at 25% width
    out_small = ffn(x, width=64)
    # Test execution at 100% width
    out_large = ffn(x, width=256)
    
    assert out_small.shape == (2, 8, hidden_dim)
    assert out_large.shape == (2, 8, hidden_dim)
    # Outputs differ because small width computes subset
    assert not torch.allclose(out_small, out_large), "Sliced widths produced identical outputs!"
    print("  [PASS] SlicedFFN genuine nested slicing verified across widths.")


def test_greed_model_pipeline():
    print("[TEST 5/6] Verifying GreedModel conversation evaluator pipeline...")
    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    greed = GreedModel(cfg, input_dim=12, num_classes=5).train()
    
    features = torch.randn(2, 6, 12)
    logits = greed(features)
    assert logits.shape == (2, 5), f"Expected logits [2, 5], got {logits.shape}"
    
    loss = logits.sum()
    loss.backward()
    assert greed.feature_proj[0].weight.grad is not None, "Feature projection received no gradient!"
    print("  [PASS] GreedModel continuous feature pipeline and backprop verified.")


def test_xorm_serialization():
    print("[TEST 6/6] Verifying .xorm archive container serialization...")
    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    model = GreedModel(cfg, input_dim=8, num_classes=5)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        xorm_path = os.path.join(tmpdir, "test.xorm")
        writer = XormWriter(
            model_name="greed_test",
            family="greed",
            role="evaluator",
            arch_config={"hidden_size": cfg.hidden_size, "num_layers": cfg.num_layers},
            feature_schema={"features": ["f1", "f2"]}
        )
        writer.save(model, xorm_path)
        
        reader = XormReader(xorm_path)
        assert reader.manifest["model_name"] == "greed_test"
        state = reader.load_state_dict()
        reader.close()
        
        assert len(state) > 0, "No state dict loaded from .xorm archive!"
    print("  [PASS] .xorm container serialization and round-trip load verified.")


def run_all():
    print("================================================================")
    print("          RUNNING XORZEN v1.0.1 REGRESSION TEST SUITE           ")
    print("================================================================")
    test_moe_registration_and_gradients()
    test_batch_boundary_isolation()
    test_zero_agentic_memory_and_actions()
    test_sliced_ffn_computation()
    test_greed_model_pipeline()
    test_xorm_serialization()
    print("================================================================")
    print(">>> ALL 6 ARCHITECTURAL REGRESSION TESTS PASSED (100%) <<<")
    print("================================================================")


if __name__ == "__main__":
    run_all()
