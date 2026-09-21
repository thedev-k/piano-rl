"""Generate procedural multi-key polyphonic piano pieces for curriculum Levels 1M to 8M.

Generates 200 pieces per level (170 train, 30 held-out) across all 8 levels (1,600 pieces total).
Pieces are saved as MIDI files into data/multikey_train and data/multikey_heldout,
and indexed in data/manifest_multikey.json.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from pianorl.score import Score, save_score_to_midi
from pianorl.data import generate_multi_key_score

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
) -> List[Dict]:
    """Generate and save the full multi-key polyphony dataset."""
    train_dir = output_base / "multikey_train"
    heldout_dir = output_base / "multikey_heldout"
    train_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict] = []
    summary_stats = []

    print(
        f"Generating {PIECES_PER_LEVEL} polyphonic pieces per level across Levels 1M to 8M..."
    )

    for level in range(1, 9):
        lvl_str = f"{level}M"
        unique_scores: List[Score] = []
        seen_fingerprints: Set[Tuple] = set()

        seed_counter = base_seed + level * 100000

        while len(unique_scores) < PIECES_PER_LEVEL:
            score = generate_multi_key_score(level=level, seed=seed_counter)
            seed_counter += 1

            fingerprint = get_score_fingerprint(score)
            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
                unique_scores.append(score)

        train_scores = unique_scores[:TRAIN_COUNT]
        heldout_scores = unique_scores[TRAIN_COUNT : TRAIN_COUNT + HELDOUT_COUNT]

        # Save train pieces
        for idx, sc in enumerate(train_scores, start=1):
            filename = f"level{lvl_str}_{idx:04d}.mid"
            file_path = train_dir / filename
            save_score_to_midi(sc, file_path)
            manifest.append(
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
            manifest.append(
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

    # Save manifest_multikey.json
    manifest_path = output_base / "manifest_multikey.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

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
        f"{'Total':<10} | {total_train:<10} | {total_heldout:<10} | {total_train + total_heldout}"
    )
    print("=" * 55)
    print(f"Manifest saved to: {manifest_path}")

    return manifest


def main():
    generate_multikey_dataset()


if __name__ == "__main__":
    main()
