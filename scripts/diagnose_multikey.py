"""Diagnose error patterns, wrong presses, and chord completion for a multi-key PPO model."""

import argparse
import json
from pathlib import Path
from typing import Dict

from pianorl.eval.diagnosis_multikey import (
    MultiKeyDiagnosisStats,
    run_multikey_diagnosis,
    run_multikey_midi_segment_diagnosis,
    print_repeat_errors_list,
)
from pianorl.score import load_score
from pianorl.data import normalize_level_tag


def print_multikey_diagnosis_table(
    results: Dict[str, MultiKeyDiagnosisStats], model_path: Path, split: str
) -> None:
    print("=" * 115)
    print(f"MULTI-KEY DIAGNOSTIC ERROR REPORT: Model = {model_path.name} | Split = {split.upper()}")
    print("=" * 115)
    print(
        f"{'Level':<10} | {'Wrong Tot':<9} | {'Repeat':<13} | {'Neighbor Key':<15} | "
        f"{'Not in Window':<14} | {'No Note':<12} | {'Press/Note':<10} | {'Full Chord Hit'}"
    )
    print("-" * 115)

    for name, st in results.items():
        if name == "Overall":
            print("-" * 115)
        repeat_str = f"{st.wrong_repeat} ({st.repeat_pct:.1f}%)"
        neighbor_str = f"{st.wrong_neighbor_key} ({st.neighbor_key_pct:.1f}%)"
        no_win_str = f"{st.wrong_not_in_window} ({st.not_in_window_pct:.1f}%)"
        no_note_str = f"{st.wrong_no_note_nearby} ({st.no_note_nearby_pct:.1f}%)"
        chord_str = f"{st.chord_steps_all_hit}/{st.chord_steps_total} ({st.full_chord_hit_rate * 100:.1f}%)"

        print(
            f"{name:<10} | {st.total_wrong:<9} | {repeat_str:<13} | {neighbor_str:<15} | "
            f"{no_win_str:<14} | {no_note_str:<12} | {st.presses_per_note:<10.2f} | {chord_str}"
        )
    print("=" * 115)
    print("  * Repeat: Pressed key was scheduled/hit within +-2 steps (double strike).")
    print("  * Neighbor Key: Pressed key was 1 or 2 semitones away from an active note within +-2 steps.")
    print("  * Not in Window: Pressed key does not appear anywhere in the 16-step lookahead window.")
    print("  * No Note Nearby: No note starts within +-2 steps (unprovoked speculative strike).")
    print("  * Full Chord Hit: Percentage of chord moments where ALL constituent notes were hit.\n")


def print_multikey_midi_segment_table(
    results: Dict[str, MultiKeyDiagnosisStats],
    model_path: Path,
    midi_path: Path,
    duration_seconds: float,
    total_notes: int,
) -> None:
    print("=" * 125)
    print(
        f"MULTI-KEY MIDI DIAGNOSTIC REPORT: Model = {model_path.name} | "
        f"File = {midi_path.name} ({duration_seconds:.1f}s, {total_notes} notes)"
    )
    print("=" * 125)
    print(
        f"{'Time Segment':<18} | {'Notes':<6} | {'Wrong Tot':<9} | {'Repeat':<13} | "
        f"{'Neighbor Key':<15} | {'Not in Window':<14} | {'No Note':<12} | "
        f"{'Press/Note':<10} | {'Full Chord Hit'}"
    )
    print("-" * 125)

    for name, st in results.items():
        if name == "Overall":
            print("-" * 125)
        repeat_str = f"{st.wrong_repeat} ({st.repeat_pct:.1f}%)"
        neighbor_str = f"{st.wrong_neighbor_key} ({st.neighbor_key_pct:.1f}%)"
        no_win_str = f"{st.wrong_not_in_window} ({st.not_in_window_pct:.1f}%)"
        no_note_str = f"{st.wrong_no_note_nearby} ({st.no_note_nearby_pct:.1f}%)"
        chord_str = f"{st.chord_steps_all_hit}/{st.chord_steps_total} ({st.full_chord_hit_rate * 100:.1f}%)"

        print(
            f"{name:<18} | {st.total_notes:<6} | {st.total_wrong:<9} | {repeat_str:<13} | "
            f"{neighbor_str:<15} | {no_win_str:<14} | {no_note_str:<12} | "
            f"{st.presses_per_note:<10.2f} | {chord_str}"
        )
    print("=" * 125)
    print("  * Repeat: Pressed key was scheduled/hit within +-2 steps (double strike).")
    print("  * Neighbor Key: Pressed key was 1 or 2 semitones away from an active note within +-2 steps.")
    print("  * Not in Window: Pressed key does not appear anywhere in the 16-step lookahead window.")
    print("  * No Note Nearby: No note starts within +-2 steps (unprovoked speculative strike).")
    print("  * Full Chord Hit: Percentage of chord moments where ALL constituent notes were hit.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose error patterns and chord accuracy for a multi-key PPO model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained multi-key PPO model .zip file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="heldout",
        choices=["heldout", "train"],
        help="Dataset split to evaluate (default: heldout).",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=None,
        help="Specific levels to diagnose (e.g. --levels 1M or 1 2 3). Defaults to all levels.",
    )
    parser.add_argument(
        "--midi-path",
        type=str,
        default=None,
        help="Path to a single MIDI file to diagnose across time segments (e.g. data/uploads/rush_e_real.mid).",
    )
    parser.add_argument(
        "--split-seconds",
        type=str,
        default="30.0",
        help="Comma-separated list of cutoff seconds for time segments (default: '30.0' for [0s-30s, 30s-end]).",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=None,
        help="Optional uniform segment width in seconds (e.g. 30.0 for every 30 seconds). Overrides --split-seconds.",
    )
    parser.add_argument(
        "--list-repeat-errors",
        action="store_true",
        default=False,
        help="Print details of every Repeat error (timestep, pitch, and true score notes active within +-0.5 beats). Requires --midi-path.",
    )
    args = parser.parse_args()

    model_file = Path(args.model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at '{model_file}'")

    if args.list_repeat_errors and not args.midi_path:
        parser.error("--list-repeat-errors requires --midi-path to be specified.")

    if args.midi_path:
        midi_file = Path(args.midi_path)
        if not midi_file.exists():
            raise FileNotFoundError(f"MIDI file not found at '{midi_file}'")

        split_sec_list = [float(x.strip()) for x in args.split_seconds.split(",") if x.strip()]

        print(f"Loading MIDI file '{midi_file.name}' for multi-key time-segment diagnosis...")
        score = load_score(midi_file)

        tempo = float(score.tempo_bpm) if (score.tempo_bpm and score.tempo_bpm > 0) else 120.0
        sec_per_beat = 60.0 / tempo
        max_end_beat = max((n.start_beat + n.duration_beats for n in score.notes), default=0.0)
        dur_sec = max_end_beat * sec_per_beat

        print(f"Running multi-key diagnosis with {model_file.name} across time segments...")
        if args.list_repeat_errors:
            results, _, repeat_errors = run_multikey_midi_segment_diagnosis(
                player_or_path=model_file,
                score_or_path=score,
                split_seconds=split_sec_list,
                segment_seconds=args.segment_seconds,
                return_repeat_errors=True,
            )
        else:
            results, _ = run_multikey_midi_segment_diagnosis(
                player_or_path=model_file,
                score_or_path=score,
                split_seconds=split_sec_list,
                segment_seconds=args.segment_seconds,
                return_repeat_errors=False,
            )

        print_multikey_midi_segment_table(
            results=results,
            model_path=model_file,
            midi_path=midi_file,
            duration_seconds=dur_sec,
            total_notes=results["Overall"].total_notes,
        )

        if args.list_repeat_errors:
            print_repeat_errors_list(repeat_errors)

        return

    manifest_file = Path("data/manifest_multikey.json")
    if not manifest_file.exists():
        raise FileNotFoundError(
            "data/manifest_multikey.json not found. Run scripts/generate_multikey_dataset.py first."
        )

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    split_items = [item for item in manifest if item["split"] == args.split]
    if args.levels:
        target_lvls = {normalize_level_tag(l) for l in args.levels}
        split_items = [
            it for it in split_items if normalize_level_tag(it["level"]) in target_lvls
        ]

    if not split_items:
        raise ValueError(
            f"No pieces found for split='{args.split}' and levels={args.levels}"
        )

    print(f"Loading {len(split_items)} {args.split} pieces for multi-key diagnosis...")
    data_dir = Path("data")
    scores_list = [load_score(data_dir / item["filename"]) for item in split_items]

    print(f"Running multi-key diagnosis with {model_file.name}...")
    results = run_multikey_diagnosis(model_file, scores_list, split_items)

    print_multikey_diagnosis_table(results, model_file, args.split)


if __name__ == "__main__":
    main()
