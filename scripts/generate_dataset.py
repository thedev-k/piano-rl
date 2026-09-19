import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from pianorl.score import Score, generate_score, save_score_to_midi

PIECES_PER_LEVEL = 200
HELDOUT_RATIO = 0.15  # 15% held-out, 85% train


def get_score_fingerprint(score: Score) -> Tuple[Tuple[int, float, float], ...]:
    """Tuple representation of notes used to detect duplicate musical content."""
    return tuple((n.pitch, n.start_beat, n.duration_beats) for n in score.notes)


def generate_dataset(
    output_base: Path = Path("data"),
    base_seed: int = 42,
) -> None:
    train_dir = output_base / "train"
    heldout_dir = output_base / "heldout"
    train_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict] = []
    summary_stats = []

    print(f"Generating {PIECES_PER_LEVEL} unique pieces per level across 4 levels...")

    for level in [1, 2, 3, 4]:
        unique_scores: List[Score] = []
        seen_fingerprints: Set[Tuple] = set()

        seed_counter = base_seed + level * 100000

        while len(unique_scores) < PIECES_PER_LEVEL:
            score = generate_score(level=level, seed=seed_counter)
            seed_counter += 1

            fingerprint = get_score_fingerprint(score)
            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
                unique_scores.append(score)

        # Split: 85% train, 15% held-out
        n_heldout = int(round(PIECES_PER_LEVEL * HELDOUT_RATIO))
        n_train = PIECES_PER_LEVEL - n_heldout

        train_scores = unique_scores[:n_train]
        heldout_scores = unique_scores[n_train:]

        # Save train pieces
        for idx, sc in enumerate(train_scores, start=1):
            filename = f"level{level}_{idx:04d}.mid"
            file_path = train_dir / filename
            save_score_to_midi(sc, file_path)
            manifest.append(
                {
                    "filename": str(Path("train") / filename),
                    "split": "train",
                    "level": level,
                    "tempo_bpm": sc.tempo_bpm,
                    "notes_count": len(sc),
                    "total_beats": sc.total_beats,
                }
            )

        # Save held-out pieces
        for idx, sc in enumerate(heldout_scores, start=1):
            filename = f"level{level}_{idx:04d}.mid"
            file_path = heldout_dir / filename
            save_score_to_midi(sc, file_path)
            manifest.append(
                {
                    "filename": str(Path("heldout") / filename),
                    "split": "heldout",
                    "level": level,
                    "tempo_bpm": sc.tempo_bpm,
                    "notes_count": len(sc),
                    "total_beats": sc.total_beats,
                }
            )

        summary_stats.append((level, len(train_scores), len(heldout_scores), len(unique_scores)))

    # Save manifest.json
    manifest_path = output_base / "manifest.json"
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
    print(f"{'Total':<10} | {total_train:<10} | {total_heldout:<10} | {total_train + total_heldout}")
    print("=" * 55)
    print(f"Manifest saved to: {manifest_path}")


def main():
    generate_dataset()


if __name__ == "__main__":
    main()
