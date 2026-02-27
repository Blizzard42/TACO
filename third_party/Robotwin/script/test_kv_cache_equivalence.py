#!/usr/bin/env python3
"""
Check at the floating-point level that the batched KV-cache expansion path in
PI05Pytorch.sample_actions produces the same action trajectories as the
equivalent sequential single-sample denoising loop.

Both approaches share the same prefix KV cache and receive *identical* noise
tensors (fixed seed).  Any numerical differences arise only from batched vs
serial CUDA kernel scheduling in bfloat16 and should be well below 1e-2.

Usage (from third_party/Robotwin/):
    python script/test_kv_cache_equivalence.py \\
        --policy_path /path/to/pretrained_model \\
        [--task_name  blocks_stack_easy]         \\
        [--num_samples 4]                        \\
        [--num_steps   10]                       \\
        [--atol        1e-2]
"""

import argparse
import sys
import os

sys.path.append("./")
sys.path.append("./policy")

import numpy as np
import torch

from policy.pi05 import pi05_model_torch
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

# -- hardcoded seed for fully deterministic noise -----------------------------
NOISE_SEED = 42
IMG_SEED   = 7


# --------------------------- input construction ------------------------------

def make_preprocessed_batch(model_wrapper, instruction: str = "pick up the red block"):
    """
    Build a synthetic observation (fixed-seed random images, zero joint state)
    and run the lerobot preprocessor to tokenise the instruction and normalise
    the state, reproducing exactly what Lerobot_torch_PI05.get_action does.
    """
    H, W = 224, 224
    rng = np.random.RandomState(IMG_SEED)
    img_np = (rng.rand(H, W, 3) * 255).astype(np.float32)  # HWC, [0,255]
    state  = np.zeros(14, dtype=np.float32)

    model_wrapper.set_language(instruction)
    # update_observation_window expects HWC numpy arrays
    model_wrapper.update_observation_window([img_np, img_np, img_np], state)

    device = model_wrapper.policy.config.device
    raw = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in model_wrapper.observation_window.items()
    }
    # preprocessor tokenises 'task' -> OBS_LANGUAGE_TOKENS / OBS_LANGUAGE_ATTENTION_MASK
    # and normalises observation.state; num_result=1 so no repeat is needed
    return model_wrapper.preprocessor(raw)


def extract_model_inputs(policy, batch):
    """Return the (images, img_masks, tokens, masks) tuple consumed by PI05Pytorch."""
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]
    return images, img_masks, tokens, masks


# --------------------------- prefix KV cache ---------------------------------

@torch.no_grad()
def build_prefix_kv_cache(m, images, img_masks, tokens, masks):
    """
    Embed image + language prefix and store the resulting KV cache - exactly
    mirroring the first block of PI05Pytorch.sample_actions.

    Returns (past_key_values, prefix_pad_masks), both with bsize=1.
    """
    prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
        images, img_masks, tokens, masks
    )
    att_2d    = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    pos_ids   = torch.cumsum(prefix_pad_masks, dim=1) - 1
    att_2d_4d = m._prepare_attention_masks_4d(att_2d)
    m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

    _, past_kv = m.paligemma_with_expert.forward(
        attention_mask=att_2d_4d,
        position_ids=pos_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return past_kv, prefix_pad_masks


# --------------------------- denoising paths ---------------------------------

@torch.no_grad()
def run_sequential(m, past_kv, prefix_pad_masks, noises, num_steps):
    """
    Baseline: N independent bsize=1 denoising passes, one per noise vector.
    This is the old for-loop approach that sample_actions used to implement.
    The same past_kv is reused across passes (use_cache=False in denoise_step
    ensures it is never modified in-place).
    """
    device = noises.device
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)

    results = []
    for i in range(len(noises)):
        x_t = noises[i : i + 1].clone()           # [1, T, A]
        t   = torch.tensor(1.0, dtype=torch.float32, device=device)
        while t >= -dt / 2:
            v_t = m.denoise_step(prefix_pad_masks, past_kv, x_t, t.expand(1))
            x_t = x_t + dt * v_t
            t  += dt
        results.append(x_t)

    return torch.cat(results, dim=0)               # [N, T, A]


@torch.no_grad()
def run_batched(m, past_kv, prefix_pad_masks, noises, num_steps):
    """
    New path: expand the bsize=1 KV cache to bsize=N (zero-copy), then run a
    single batched denoising pass over all N noise vectors simultaneously.
    """
    N      = len(noises)
    device = noises.device
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)

    exp_kv     = m._expand_kv_cache(past_kv, N)
    exp_prefix = prefix_pad_masks.expand(N, -1)

    x_t = noises.clone()                           # [N, T, A]
    t   = torch.tensor(1.0, dtype=torch.float32, device=device)
    while t >= -dt / 2:
        v_t = m.denoise_step(exp_prefix, exp_kv, x_t, t.expand(N))
        x_t = x_t + dt * v_t
        t  += dt

    return x_t                                     # [N, T, A]


# --------------------------- VRAM monitoring ---------------------------------

def _mb(bytes_: int) -> str:
    return f"{bytes_ / 1024**2:.1f} MB"


class VRAMSnapshot:
    """Capture peak and current VRAM before/after a code block."""

    def __init__(self, label: str, device):
        self.label  = label
        self.device = device
        self.before_current  = 0
        self.before_reserved = 0
        self.peak_allocated  = 0
        self.after_current   = 0
        self.after_reserved  = 0

    def __enter__(self):
        if self.device.type != "cuda":
            return self
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.before_current  = torch.cuda.memory_allocated(self.device)
        self.before_reserved = torch.cuda.memory_reserved(self.device)
        return self

    def __exit__(self, *_):
        if self.device.type != "cuda":
            return
        torch.cuda.synchronize(self.device)
        self.peak_allocated = torch.cuda.max_memory_allocated(self.device)
        self.after_current  = torch.cuda.memory_allocated(self.device)
        self.after_reserved = torch.cuda.memory_reserved(self.device)

    def delta_peak(self) -> int:
        """Extra VRAM allocated above the pre-run baseline at the high-water mark."""
        return self.peak_allocated - self.before_current

    def print(self):
        if self.device.type != "cuda":
            print(f"  [{self.label}] VRAM monitoring requires a CUDA device.")
            return
        print(f"  [{self.label}]")
        print(f"    before  allocated : {_mb(self.before_current)}")
        print(f"    peak    allocated : {_mb(self.peak_allocated)}  "
              f"(+{_mb(self.delta_peak())} above baseline)")
        print(f"    after   allocated : {_mb(self.after_current)}")
        print(f"    after   reserved  : {_mb(self.after_reserved)}")


def print_vram_comparison(snap_seq: VRAMSnapshot, snap_bat: VRAMSnapshot):
    if snap_seq.device.type != "cuda":
        return
    saved = snap_seq.delta_peak() - snap_bat.delta_peak()
    pct   = 100.0 * saved / snap_seq.delta_peak() if snap_seq.delta_peak() else 0.0
    print(f"\n  Peak delta  sequential : {_mb(snap_seq.delta_peak())}")
    print(f"  Peak delta  batched    : {_mb(snap_bat.delta_peak())}")
    print(f"  Saved by batched path  : {_mb(saved)}  ({pct:.1f}%)")


# --------------------------- comparison helpers -------------------------------

def _tol_tag(ok: bool) -> str:
    return "\033[92mok\033[0m" if ok else "\033[91mfail\033[0m"


def compare_sample(seq: torch.Tensor, bat: torch.Tensor, idx: int, tols=(1e-2, 1e-3, 1e-4)):
    """Pretty-print per-sample statistics; return max absolute difference."""
    s = seq.float()
    b = bat.float()
    diff = (s - b).abs()

    max_d  = diff.max().item()
    mean_d = diff.mean().item()
    exact  = torch.equal(seq, bat)

    print(f"\n  Sample {idx}  shape={tuple(seq.shape)}")
    print(f"    torch.equal             : {exact}")
    print(f"    max  |seq - bat|        : {max_d:.3e}")
    print(f"    mean |seq - bat|        : {mean_d:.3e}")
    for atol in tols:
        ok = torch.allclose(s, b, atol=atol, rtol=0.0)
        print(f"    allclose(atol={atol:.0e})    {_tol_tag(ok)}")

    return max_d


# ------------------------------- main ----------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify sequential-vs-batched KV-cache sampling equivalence"
    )
    parser.add_argument("--policy_path", required=True,
                        help="Path to PI05 pretrained checkpoint directory")
    parser.add_argument("--task_name",   default="blocks_stack_easy",
                        help="RoboTwin task name (used only for model init)")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of action samples to generate and compare")
    parser.add_argument("--num_steps",   type=int, default=10,
                        help="Denoising steps per trajectory (fewer = faster)")
    parser.add_argument("--atol",        type=float, default=1e-2,
                        help="Max-abs-diff threshold for overall PASS/FAIL")
    args = parser.parse_args()

    assert args.num_samples > 1, "--num_samples must be > 1 to exercise the batched path"

    # -- Load model --------------------------------------------------------
    print(f"\nLoading model  {args.policy_path}")
    wrapper = pi05_model_torch.Lerobot_torch_PI05(args.task_name, args.policy_path)
    m      = wrapper.policy.model          # PI05Pytorch (raw module)
    device = next(m.parameters()).device
    print(f"  device: {device}")

    # -- Build synthetic preprocessed inputs -------------------------------
    print("\nBuilding synthetic preprocessed batch  (img_seed={IMG_SEED}) ...")
    batch = make_preprocessed_batch(wrapper)
    images, img_masks, tokens, masks = extract_model_inputs(wrapper.policy, batch)
    print(f"  tokens shape : {tokens.shape}")

    # -- Build prefix KV cache (bsize=1) -----------------------------------
    print("\nBuilding prefix KV cache (bsize=1) ...")
    past_kv, prefix_pad_masks = build_prefix_kv_cache(m, images, img_masks, tokens, masks)
    print(f"  prefix_pad_masks shape: {prefix_pad_masks.shape}")

    # Check KV cache format
    if hasattr(past_kv, "key_cache"):
        layers  = len(past_kv.key_cache)
        kv_shape = tuple(past_kv.key_cache[0].shape) if layers else "empty"
        print(f"  KV format: DynamicCache  |  {layers} layers  |  per-layer key shape: {kv_shape}")
    else:
        layers   = len(past_kv)
        kv_shape = tuple(past_kv[0][0].shape) if layers else "empty"
        print(f"  KV format: tuple-of-tuples  |  {layers} layers  |  per-layer key shape: {kv_shape}")

    # -- Fixed noise tensors (same for both paths) -------------------------
    T = m.config.chunk_size
    A = m.config.max_action_dim
    torch.manual_seed(NOISE_SEED)
    noises = torch.randn(args.num_samples, T, A, dtype=torch.float32, device=device)
    print(f"\nNoise: shape={tuple(noises.shape)}  seed={NOISE_SEED}  dtype={noises.dtype}")
    print(f"Denoising steps: {args.num_steps}  (model default: {m.config.num_inference_steps})")

    # Print first-token of first noise for sanity
    print(f"  noises[0, 0, :4] = {noises[0, 0, :4].tolist()}")

    # -- Sequential baseline -----------------------------------------------
    print(f"\n[1] Sequential  - {args.num_samples} x bsize=1 independent passes ...")
    snap_seq = VRAMSnapshot("sequential", device)
    with snap_seq:
        seq_out = run_sequential(m, past_kv, prefix_pad_masks, noises, args.num_steps)
    print(f"    output shape: {tuple(seq_out.shape)}")
    print(f"    seq_out[0, 0, :4] = {seq_out[0, 0, :4].tolist()}")
    snap_seq.print()

    # -- Batched (new) -----------------------------------------------------
    print(f"\n[2] Batched     - bsize={args.num_samples} expanded-KV single pass ...")
    snap_bat = VRAMSnapshot("batched", device)
    with snap_bat:
        bat_out = run_batched(m, past_kv, prefix_pad_masks, noises, args.num_steps)
    print(f"    output shape: {tuple(bat_out.shape)}")
    print(f"    bat_out[0, 0, :4] = {bat_out[0, 0, :4].tolist()}")
    snap_bat.print()

    # -- Shape sanity check ------------------------------------------------
    assert seq_out.shape == bat_out.shape, \
        f"Shape mismatch: seq={seq_out.shape}  bat={bat_out.shape}"

    # -- VRAM summary ------------------------------------------------------
    print("\n" + "=" * 62)
    print("VRAM COMPARISON")
    print("=" * 62)
    print_vram_comparison(snap_seq, snap_bat)

    # -- Per-sample diff report --------------------------------------------
    print("\n" + "=" * 62)
    print("NUMERICAL COMPARISON")
    print("=" * 62)

    worst = 0.0
    for i in range(args.num_samples):
        d = compare_sample(seq_out[i], bat_out[i], i)
        worst = max(worst, d)

    # Global stats across all samples
    global_diff = (seq_out.float() - bat_out.float()).abs()
    print(f"\n  Global max  |seq - bat| : {global_diff.max().item():.3e}")
    print(f"  Global mean |seq - bat| : {global_diff.mean().item():.3e}")

    # -- Verdict -----------------------------------------------------------
    print("\n" + "=" * 62)
    passed = worst <= args.atol
    tag    = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    print(f"Worst per-sample max-abs-diff : {worst:.3e}")
    print(f"Tolerance (--atol)            : {args.atol:.0e}")
    print(f"Verdict                       : {tag}")
    print("=" * 62)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
