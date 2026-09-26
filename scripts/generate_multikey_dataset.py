"""Generate procedural multi-key polyphonic piano pieces for curriculum Levels 1M to 8M.

Generates 200 pieces per level (170 train, 30 held-out) across all 8 levels (1,600 pieces total).
Pieces are saved as MIDI files into data/multikey_train and data/multikey_heldout,
and indexed in data/manifest_multikey.json.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from pianorl.score import Score, save_score_to_midi
from pianorl.data import generate_multi_key_score, normalize_level_tag

PIECES_PER_LEVEL = 200
TRAIN_COUNT = 170
HELDOUT_COUNT = 30


def get_score_fingerprint(score: Score) -> Tuple[Tuple[int, float, float], ...]:
    """Tuple representation of notes used to detect duplicate musical content."""
    return tuple(
        (n.pitch, round(n.start_beat, 4), round(n.duration_beats, 4))
        for n in score.notes
    )


def generate_multikey_dataset(
    output_base: Path = Path("data"),
    base_seed: int = 1000,
    levels: Optional[List[Union[int, str]]] = None,
    pieces_per_level: int = PIECES_PER_LEVEL,
    train_count: int = TRAIN_COUNT,
    heldout_count: int = HELDOUT_COUNT,
) -> List[Dict]:
    """Generate and save multi-key polyphony dataset for specified levels (1M to 10M)."""
    train_dir = output_base / "multikey_train"
    heldout_dir = output_base / "multikey_heldout"
    train_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)

    target_levels: List[int] = []
    if levels is None:
        target_levels = list(range(1, 11))
    else:
        for lvl in levels:
            clean = str(lvl).upper().rstrip("M")
            target_levels.append(int(clean))

    # Read existing manifest if present to preserve other levels
    manifest_path = output_base / "manifest_multikey.json"
    existing_items: List[Dict] = []
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing_items = json.load(f)
        except Exception:
            existing_items = []

    target_tags = {f"{lvl}M" for lvl in target_levels}
    # Keep items for levels that are NOT being regenerated
    preserved_items = [it for it in existing_items if it.get("level") not in target_tags]

    new_items: List[Dict] = []
    summary_stats = []

    print(
        f"Generating {pieces_per_level} polyphonic pieces per level across {sorted(target_tags)}..."
    )

    for level in target_levels:
        lvl_str = f"{level}M"
        unique_scores: List[Score] = []
        seen_fingerprints: Set[Tuple] = set()

        seed_counter = base_seed + level * 100000

        while len(unique_scores) < pieces_per_level:
            score = generate_multi_key_score(level=level, seed=seed_counter)
            seed_counter += 1

            fingerprint = get_score_fingerprint(score)
            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
                unique_scores.append(score)

        train_scores = unique_scores[:train_count]
        heldout_scores = unique_scores[train_count : train_count + heldout_count]

        # Save train pieces
        for idx, sc in enumerate(train_scores, start=1):
            filename = f"level{lvl_str}_{idx:04d}.mid"
            file_path = train_dir / filename
            save_score_to_midi(sc, file_path)
            new_items.append(
                {
                    "filename": str(Path("multikey_train") / filename),
                    "split": "train",
                    "level": lvl_str,
                    "tempo_bpm": sc.tempo_bpm,
                    "notes_count": len(sc),
                    "total_beats": sc.total_beats,
                }
            )

        # Save heldout pieces
        for idx, sc in enumerate(heldout_scores, start=1):
            filename = f"level{lvl_str}_{idx:04d}.mid"
            file_path = heldout_dir / filename
            save_score_to_midi(sc, file_path)
            new_items.append(
                {
                    "filename": str(Path("multikey_heldout") / filename),
                    "split": "heldout",
                    "level": lvl_str,
                    "tempo_bpm": sc.tempo_bpm,
                    "notes_count": len(sc),
                    "total_beats": sc.total_beats,
                }
            )

        summary_stats.append(
            (lvl_str, len(train_scores), len(heldout_scores), len(unique_scores))
        )

    # Combined manifest sorted by level and filename
    combined_manifest = preserved_items + new_items
    combined_manifest.sort(
        key=lambda it: (
            int(it["level"].rstrip("M")) if it["level"].rstrip("M").isdigit() else 99,
            it["split"],
            it["filename"],
        )
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(combined_manifest, f, indent=2)

    # Print summary table
    print("\n" + "=" * 55)
    print(f"{'Level':<10} | {'Train':<10} | {'Held-out':<10} | {'Total'}")
    print("-" * 55)
    total_train = 0
    total_heldout = 0
    for lvl, tr, ho, tot in summary_stats:
        print(f"Level {lvl:<4} | {tr:<10} | {ho:<10} | {tot}")
        total_train += tr
        total_heldout += ho
    print("-" * 55)
    print(
        f"{'Generated':<10} | {total_train:<10} | {total_heldout:<10} | {total_train + total_heldout}"
    )
    print(f"{'Manifest':<10} | Total pieces across all levels: {len(combined_manifest)}")
    print("=" * 55)
    print(f"Manifest saved to: {manifest_path}")

    return combined_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate procedural multi-key dataset pieces (Levels 1M to 9M)."
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=None,
        help="Specific levels to generate (e.g. --levels 9M). Defaults to all levels (1M to 9M).",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="data",
        help="Base directory for output datasets (default: data).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=1000,
        help="Random seed base for deterministic piece generation (default: 1000).",
    )
    parser.add_argument(
        "--pieces-per-level",
        type=int,
        default=PIECES_PER_LEVEL,
        help="Total pieces to generate per level (default: 200).",
    )
    args = parser.parse_args()

    generate_multikey_dataset(
        output_base=Path(args.output_base),
        base_seed=args.base_seed,
        levels=args.levels,
        pieces_per_level=args.pieces_per_level,
    )


if __name__ == "__main__":
    main()
