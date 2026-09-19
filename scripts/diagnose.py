import argparse
import json
from pathlib import Path
from typing import Dict

from pianorl.eval.diagnosis import DiagnosisStats, run_diagnosis
from pianorl.score import load_score


def print_diagnosis_table(results: Dict[str, DiagnosisStats], model_path: Path, split: str) -> None:
    print("=" * 96)
    print(f"DIAGNOSTIC ERROR BREAKDOWN: Model = {model_path.name} | Split = {split.upper()}")
    print("=" * 96)
    print(
        f"{'Level':<10} | {'Wrong Total':<11} | {'Repeat':<14} | {'Wrong Key':<15} | "
        f"{'No Note':<15} | {'Presses/Note':<12} | {'Recall':<7} | {'Precision'}"
    )
    print("-" * 96)

    for name, st in results.items():
        if name == "Overall":
            print("-" * 96)
        repeat_str = f"{st.wrong_repeat} ({st.repeat_pct:.1f}%)"
        wrong_key_str = f"{st.wrong_key_near} ({st.wrong_key_near_pct:.1f}%)"
        no_note_str = f"{st.wrong_no_note} ({st.no_note_nearby_pct:.1f}%)"

        print(
            f"{name:<10} | {st.total_wrong:<11} | {repeat_str:<14} | {wrong_key_str:<15} | "
            f"{no_note_str:<15} | {st.presses_per_note:<12.2f} | {st.recall:<7.3f} | {st.precision:.3f}"
        )
    print("=" * 96)


def main():
    parser = argparse.ArgumentParser(description="Diagnose error patterns and wrong presses for a PPO model.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained PPO model zip file (e.g. checkpoints/all_levels_run/final.zip)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="heldout",
        choices=["heldout", "train"],
        help="Dataset split to evaluate (default: heldout)",
    )
    args = parser.parse_args()

    model_file = Path(args.model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at '{model_file}'")

    manifest_file = Path("data/manifest.json")
    if not manifest_file.exists():
        raise FileNotFoundError("data/manifest.json not found. Run scripts/generate_dataset.py first.")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    split_items = [item for item in manifest if item["split"] == args.split]
    if not split_items:
        raise ValueError(f"No pieces found in manifest for split '{args.split}'")

    print(f"Loading {len(split_items)} {args.split} pieces for diagnosis...")
    data_dir = Path("data")
    scores_list = [load_score(data_dir / item["filename"]) for item in split_items]

    print(f"Running diagnosis with {model_file.name} on {args.split} split...")
    results = run_diagnosis(model_file, scores_list, split_items)

    print_diagnosis_table(results, model_file, args.split)


if __name__ == "__main__":
    main()
