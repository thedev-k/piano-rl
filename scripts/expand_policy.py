"""Script to expand MultiKeyPitchConvPolicy from small (64 channels) to medium (96 channels).

Transfers learned weights from checkpoints/multikey_full_curriculum/final.zip into the
expanded medium architecture without loss of existing mathematical behavior.
"""

from pathlib import Path
import json
import torch as th
from stable_baselines3 import PPO

from pianorl.agent import MultiKeyPitchConvPolicy
from pianorl.env import MultiKeyPianoEnv
from pianorl.score import load_score


def main():
    source_path = Path("checkpoints/multikey_full_curriculum/final.zip")
    output_dir = Path("checkpoints/medium_baseline_expanded")
    output_path = output_dir / "final.zip"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found at: {source_path}")

    # Create dummy environment for loading/instantiating policies
    manifest_path = Path("data/manifest_multikey.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    dummy_score = load_score(Path("data") / manifest[0]["filename"])
    dummy_env = MultiKeyPianoEnv(scores=[dummy_score], seed=42)

    print(f"Loading small model from '{source_path}'...")
    small_model = PPO.load(
        str(source_path),
        custom_objects={
            "env": dummy_env,
            "MultiKeyPitchConvPolicy": MultiKeyPitchConvPolicy,
        },
        device="cpu",
    )

    print("Initializing fresh medium model (size='medium', 96 channels)...")
    medium_model = PPO(
        MultiKeyPitchConvPolicy,
        dummy_env,
        policy_kwargs={"policy_size": "medium"},
        device="cpu",
    )

    small_sd = small_model.policy.state_dict()
    medium_sd = medium_model.policy.state_dict()

    # Zero out all parameters in medium model first so new channels contribute 0
    for k in medium_sd:
        medium_sd[k].zero_()

    # Transfer small model weights and biases into corresponding top-left slices
    print("\nTransferring weights from small model to medium model:")
    for k, s_tensor in small_sd.items():
        if k not in medium_sd:
            print(f"  [Warning] Key '{k}' not in medium state_dict, skipping.")
            continue

        m_tensor = medium_sd[k]
        if k == "value_net.0.weight":
            # Value net input summary is [avg_pool(C), max_pool(C), pos_in_beat(4), tempo(1)]
            # In small: 64 + 64 + 4 + 1 = 133
            # In medium: 96 + 96 + 4 + 1 = 197
            m_tensor[:64, 0:64] = s_tensor[:64, 0:64]          # avg pool features
            m_tensor[:64, 96:160] = s_tensor[:64, 64:128]      # max pool features
            m_tensor[:64, 192:196] = s_tensor[:64, 128:132]    # pos in beat
            m_tensor[:64, 196:197] = s_tensor[:64, 132:133]    # tempo
            print(f"  {k:<22}: {str(list(s_tensor.shape)):<16} -> {str(list(m_tensor.shape)):<16} (aligned summary)")
        else:
            slices = tuple(slice(0, dim) for dim in s_tensor.shape)
            m_tensor[slices] = s_tensor
            print(f"  {k:<22}: {str(list(s_tensor.shape)):<16} -> {str(list(m_tensor.shape)):<16} (top-left slice)")

    # Load transferred weights into medium policy
    medium_model.policy.load_state_dict(medium_sd)

    # Verification on dummy inputs
    dummy_obs = th.randn(5, 2821)
    with th.no_grad():
        s_logits, s_vals = small_model.policy._forward_network(dummy_obs)
        m_logits, m_vals = medium_model.policy._forward_network(dummy_obs)

    logit_diff = (s_logits - m_logits).abs().max().item()
    val_diff = (s_vals - m_vals).abs().max().item()
    print(f"\nVerification check on random observations:")
    print(f"  Max action logit difference: {logit_diff:.8f}")
    print(f"  Max value estimate difference: {val_diff:.8f}")

    assert logit_diff < 1e-6, f"Action logits differ by {logit_diff}! Transfer is not exact."
    assert val_diff < 1e-6, f"Value estimates differ by {val_diff}! Transfer is not exact."

    print(f"\nSaving expanded medium model to '{output_path}'...")
    medium_model.save(str(output_path))
    print("Expansion and weight transfer complete successfully!\n")


if __name__ == "__main__":
    main()
