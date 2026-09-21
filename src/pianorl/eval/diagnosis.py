from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from pianorl.env import PianoFreeKeysEnv, split_observation
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


def get_window_slot_category(obs: np.ndarray, action: int) -> Tuple[Optional[int], str]:
    """Inspect the observation window to check if the pressed pitch appears as a note start.

    Args:
        obs: Flat observation vector seen by the player at that step.
        action: Key strike action taken (1 to 88).

    Returns:
        Tuple of (slot_index_or_None, category_name):
            - slot_index: 0 to 15 if found, or None if not in window.
            - category_name: 'slots_0_1', 'slots_2_3', 'slots_4_7', 'slots_8_15', or 'not_in_window'.
    """
    row = action - 1
    window, _, _ = split_observation(obs)
    onset_slots = np.where(window[0, row, :] > 0.5)[0]
    if len(onset_slots) == 0:
        return None, "not_in_window"

    first_slot = int(onset_slots[0])
    if first_slot in (0, 1):
        return first_slot, "slots_0_1"
    elif first_slot in (2, 3):
        return first_slot, "slots_2_3"
    elif 4 <= first_slot <= 7:
        return first_slot, "slots_4_7"
    else:
        return first_slot, "slots_8_15"


def get_semitone_distance(
    pressed_pitch: int,
    current_step: int,
    targets: List[dict],
) -> Optional[int]:
    """Calculate distance in semitones between pressed key and nearest note start within 2 steps.

    Args:
        pressed_pitch: MIDI pitch of the struck key (21 to 108).
        current_step: Discrete 16th-note step when strike occurred.
        targets: List of all target note dicts in the piece.

    Returns:
        int: Minimum semitone distance to any note start within 2 steps (0, 1, 2, ...),
             or None if no note starts within 2 steps.
    """
    nearby_targets = [t for t in targets if abs(t["start_step"] - current_step) <= 2]
    if not nearby_targets:
        return None
    return min(abs(pressed_pitch - t["pitch"]) for t in nearby_targets)


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

    # Lookahead score-window onset groups for wrong presses
    window_slots_0_1: int = 0
    window_slots_2_3: int = 0
    window_slots_4_7: int = 0
    window_slots_8_15: int = 0
    window_not_in_window: int = 0

    # Semitone distance to nearest note start within 2 steps (if any)
    dist_0: int = 0               # Distance 0 (repeat of target pitch)
    dist_1_2: int = 0             # Distance 1 or 2 semitones
    dist_3_plus: int = 0          # Distance 3 or more semitones
    dist_no_nearby_note: int = 0  # No note starts within 2 steps

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

    # Window onset percentages
    @property
    def slots_0_1_pct(self) -> float:
        return (100.0 * self.window_slots_0_1 / self.total_wrong) if self.total_wrong > 0 else 0.0

    @property
    def slots_2_3_pct(self) -> float:
        return (100.0 * self.window_slots_2_3 / self.total_wrong) if self.total_wrong > 0 else 0.0

    @property
    def slots_4_7_pct(self) -> float:
        return (100.0 * self.window_slots_4_7 / self.total_wrong) if self.total_wrong > 0 else 0.0

    @property
    def slots_8_15_pct(self) -> float:
        return (100.0 * self.window_slots_8_15 / self.total_wrong) if self.total_wrong > 0 else 0.0

    @property
    def not_in_window_pct(self) -> float:
        return (100.0 * self.window_not_in_window / self.total_wrong) if self.total_wrong > 0 else 0.0

    # Semitone distance percentages
    @property
    def dist_1_2_pct(self) -> float:
        return (100.0 * self.dist_1_2 / self.total_wrong) if self.total_wrong > 0 else 0.0

    @property
    def dist_3_plus_pct(self) -> float:
        return (100.0 * self.dist_3_plus / self.total_wrong) if self.total_wrong > 0 else 0.0

    @property
    def dist_0_pct(self) -> float:
        return (100.0 * self.dist_0 / self.total_wrong) if self.total_wrong > 0 else 0.0

    @property
    def dist_no_nearby_note_pct(self) -> float:
        return (100.0 * self.dist_no_nearby_note / self.total_wrong) if self.total_wrong > 0 else 0.0

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

    level_stats: Dict[int, DiagnosisStats] = {
        lvl: DiagnosisStats() for lvl in sorted(set(item["level"] for item in split_items))
    }
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
            obs_before = obs
            action = player.act(obs)

            if action > 0:
                lvl.total_presses += 1
                overall_stats.total_presses += 1

            obs, reward, terminated, truncated, info = env.step(action)

            # Check if this step incurred a wrong press
            curr_wrong = info["wrong_presses"]
            if curr_wrong > prev_wrong:
                pressed_pitch = 20 + action

                # Basic 3-category diagnosis
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

                # Extended check (a): Time slot in visible score window (using observation seen at this step)
                _, win_cat = get_window_slot_category(obs_before, action)
                if win_cat == "slots_0_1":
                    lvl.window_slots_0_1 += 1
                    overall_stats.window_slots_0_1 += 1
                elif win_cat == "slots_2_3":
                    lvl.window_slots_2_3 += 1
                    overall_stats.window_slots_2_3 += 1
                elif win_cat == "slots_4_7":
                    lvl.window_slots_4_7 += 1
                    overall_stats.window_slots_4_7 += 1
                elif win_cat == "slots_8_15":
                    lvl.window_slots_8_15 += 1
                    overall_stats.window_slots_8_15 += 1
                else:
                    lvl.window_not_in_window += 1
                    overall_stats.window_not_in_window += 1

                # Extended check (b): Semitone distance to nearest note start within 2 steps (if any)
                semitone_dist = get_semitone_distance(pressed_pitch, step_before, targets)
                if semitone_dist is None:
                    lvl.dist_no_nearby_note += 1
                    overall_stats.dist_no_nearby_note += 1
                elif semitone_dist == 0:
                    lvl.dist_0 += 1
                    overall_stats.dist_0 += 1
                elif semitone_dist in (1, 2):
                    lvl.dist_1_2 += 1
                    overall_stats.dist_1_2 += 1
                else:  # semitone_dist >= 3
                    lvl.dist_3_plus += 1
                    overall_stats.dist_3_plus += 1

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
