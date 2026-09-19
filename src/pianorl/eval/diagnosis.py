from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union
import numpy as np

from pianorl.env import PianoFreeKeysEnv
from pianorl.eval.players import PPOPlayer
from pianorl.score import Score, load_score


def categorize_wrong_press(
    pressed_pitch: int,
    current_step: int,
    targets: List[dict],
) -> str:
    """Categorize a wrong key strike into one of three mutually exclusive types:

    1. 'repeat': The pressed key matches a note whose start step is within 2 steps
       before or after this step (a double-strike around a real note).
    2. 'wrong_key_near': Some note starts within 2 steps, but with a different pitch.
    3. 'no_note_nearby': No note starts within 2 steps.

    Args:
        pressed_pitch: MIDI pitch of the struck key (21 to 108).
        current_step: Discrete 16th-note step when the strike occurred.
        targets: List of all note target dicts in the piece.

    Returns:
        str: 'repeat', 'wrong_key_near', or 'no_note_nearby'.
    """
    # Category A: repeat of a note starting within +-2 steps
    for t in targets:
        if t["pitch"] == pressed_pitch and abs(t["start_step"] - current_step) <= 2:
            return "repeat"

    # Category B: wrong key near a note (some note starts within +-2 steps)
    for t in targets:
        if abs(t["start_step"] - current_step) <= 2:
            return "wrong_key_near"

    # Category C: completely unprovoked strike with no note nearby
    return "no_note_nearby"


@dataclass
class DiagnosisStats:
    """Diagnostic counts and metrics for a collection of pieces."""
    pieces_count: int = 0
    total_notes: int = 0
    total_presses: int = 0
    hits_exact: int = 0
    hits_off_by_one: int = 0
    missed_notes: int = 0

    wrong_repeat: int = 0
    wrong_key_near: int = 0
    wrong_no_note: int = 0

    @property
    def total_wrong(self) -> int:
        return self.wrong_repeat + self.wrong_key_near + self.wrong_no_note

    @property
    def repeat_pct(self) -> float:
        return (100.0 * self.wrong_repeat / self.total_wrong) if self.total_wrong > 0 else 0.0

    @property
    def wrong_key_near_pct(self) -> float:
        return (100.0 * self.wrong_key_near / self.total_wrong) if self.total_wrong > 0 else 0.0

    @property
    def no_note_nearby_pct(self) -> float:
        return (100.0 * self.wrong_no_note / self.total_wrong) if self.total_wrong > 0 else 0.0

    @property
    def presses_per_note(self) -> float:
        return (self.total_presses / self.total_notes) if self.total_notes > 0 else 0.0

    @property
    def precision(self) -> float:
        tp = self.hits_exact + self.hits_off_by_one
        total_attempts = tp + self.total_wrong
        return (tp / total_attempts) if total_attempts > 0 else 0.0

    @property
    def recall(self) -> float:
        tp = self.hits_exact + self.hits_off_by_one
        total_target = tp + self.missed_notes
        return (tp / total_target) if total_target > 0 else 0.0


def run_diagnosis(
    model_path: Union[str, Path],
    scores_list: List[Score],
    split_items: List[dict],
) -> Dict[str, DiagnosisStats]:
    """Run diagnostic play on all pieces and collect detailed error breakdowns."""
    player = PPOPlayer(model_path)
    env = PianoFreeKeysEnv(scores=scores_list, seed=42)

    level_stats: Dict[int, DiagnosisStats] = {1: DiagnosisStats(), 2: DiagnosisStats(), 3: DiagnosisStats(), 4: DiagnosisStats()}
    overall_stats = DiagnosisStats()

    for i, item in enumerate(split_items):
        level = item["level"]
        lvl = level_stats[level]

        obs, info = env.reset(options={"piece_index": i})
        targets = env.targets

        lvl.pieces_count += 1
        overall_stats.pieces_count += 1
        lvl.total_notes += len(targets)
        overall_stats.total_notes += len(targets)

        prev_wrong = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            step_before = env.current_step
            action = player.act(obs)

            if action > 0:
                lvl.total_presses += 1
                overall_stats.total_presses += 1

            obs, reward, terminated, truncated, info = env.step(action)

            # Check if this step incurred a wrong press
            curr_wrong = info["wrong_presses"]
            if curr_wrong > prev_wrong:
                pressed_pitch = 20 + action
                category = categorize_wrong_press(pressed_pitch, step_before, targets)
                if category == "repeat":
                    lvl.wrong_repeat += 1
                    overall_stats.wrong_repeat += 1
                elif category == "wrong_key_near":
                    lvl.wrong_key_near += 1
                    overall_stats.wrong_key_near += 1
                else:
                    lvl.wrong_no_note += 1
                    overall_stats.wrong_no_note += 1
                prev_wrong = curr_wrong

        # Aggregate hits and misses at end of piece
        lvl.hits_exact += info["hits_exact"]
        overall_stats.hits_exact += info["hits_exact"]
        lvl.hits_off_by_one += info["hits_off_by_one"]
        overall_stats.hits_off_by_one += info["hits_off_by_one"]
        lvl.missed_notes += info["missed_notes"]
        overall_stats.missed_notes += info["missed_notes"]

    results = {}
    for lvl_num in sorted(level_stats.keys()):
        if level_stats[lvl_num].pieces_count > 0:
            results[f"Level {lvl_num}"] = level_stats[lvl_num]
    results["Overall"] = overall_stats

    return results
