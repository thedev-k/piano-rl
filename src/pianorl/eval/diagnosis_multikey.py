"""Multi-key diagnostic error analysis for polyphonic piano playing.

Analyzes wrong presses into 4 mutually exclusive categories:
1. repeat: Struck key matches a target note within +-2 steps.
2. neighbor_key: Struck key is within 1 or 2 semitones of a note within +-2 steps.
3. not_in_window: Struck key does not appear anywhere in the 16-slot lookahead score window.
4. no_note_nearby: No note starts within +-2 steps (unprovoked strike).

Also measures presses per note and full-chord completion rate (share of chords where all notes were hit).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from pianorl.env.multi_keys_env import MultiKeyPianoEnv
from pianorl.env.free_keys_env import split_observation, RewardConfig
from pianorl.eval.players import PPOPlayer
from pianorl.score import Score, NUM_PIANO_KEYS


def categorize_multikey_wrong_press(
    pressed_pitch: int,
    current_step: int,
    targets: List[dict],
    obs: np.ndarray,
) -> str:
    """Categorize an unmatched struck pitch into one of 4 error types."""
    # 1. Repeat: Same pitch starts within +-2 steps
    for t in targets:
        if t["pitch"] == pressed_pitch and abs(t["start_step"] - current_step) <= 2:
            return "repeat"

    # 2. Neighbor key: Within 1 or 2 semitones of a note starting within +-2 steps
    for t in targets:
        if abs(t["start_step"] - current_step) <= 2:
            if 1 <= abs(t["pitch"] - pressed_pitch) <= 2:
                return "neighbor_key"

    # 3. Not in window: Key does not appear in the 16-slot lookahead score window
    row = pressed_pitch - 21
    if 0 <= row < NUM_PIANO_KEYS:
        window, _, _ = split_observation(obs)
        if not np.any(window[0, row, :] > 0.5):
            return "not_in_window"

    # 4. No note nearby
    return "no_note_nearby"


@dataclass
class MultiKeyDiagnosisStats:
    """Diagnostic counts and metrics for a set of multi-key pieces."""
    pieces_count: int = 0
    total_notes: int = 0
    total_presses: int = 0
    hits_exact: int = 0
    hits_off_by_one: int = 0
    missed_notes: int = 0

    wrong_repeat: int = 0
    wrong_neighbor_key: int = 0
    wrong_not_in_window: int = 0
    wrong_no_note_nearby: int = 0

    chord_steps_total: int = 0
    chord_steps_all_hit: int = 0

    @property
    def total_wrong(self) -> int:
        return (
            self.wrong_repeat
            + self.wrong_neighbor_key
            + self.wrong_not_in_window
            + self.wrong_no_note_nearby
        )

    @property
    def repeat_pct(self) -> float:
        return (100.0 * self.wrong_repeat / self.total_wrong) if self.total_wrong > 0 else 0.0

    @property
    def neighbor_key_pct(self) -> float:
        return (100.0 * self.wrong_neighbor_key / self.total_wrong) if self.total_wrong > 0 else 0.0

    @property
    def not_in_window_pct(self) -> float:
        return (100.0 * self.wrong_not_in_window / self.total_wrong) if self.total_wrong > 0 else 0.0

    @property
    def no_note_nearby_pct(self) -> float:
        return (100.0 * self.wrong_no_note_nearby / self.total_wrong) if self.total_wrong > 0 else 0.0

    @property
    def presses_per_note(self) -> float:
        return (self.total_presses / self.total_notes) if self.total_notes > 0 else 0.0

    @property
    def full_chord_hit_rate(self) -> float:
        return (self.chord_steps_all_hit / self.chord_steps_total) if self.chord_steps_total > 0 else 1.0

    @property
    def precision(self) -> float:
        tp = self.hits_exact + self.hits_off_by_one
        denom = tp + self.total_wrong
        return (tp / denom) if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        tp = self.hits_exact + self.hits_off_by_one
        denom = tp + self.missed_notes
        return (tp / denom) if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

    @property
    def exact_rate(self) -> float:
        return (self.hits_exact / self.total_notes) if self.total_notes > 0 else 0.0

    def add(self, other: "MultiKeyDiagnosisStats") -> "MultiKeyDiagnosisStats":
        return MultiKeyDiagnosisStats(
            pieces_count=self.pieces_count + other.pieces_count,
            total_notes=self.total_notes + other.total_notes,
            total_presses=self.total_presses + other.total_presses,
            hits_exact=self.hits_exact + other.hits_exact,
            hits_off_by_one=self.hits_off_by_one + other.hits_off_by_one,
            missed_notes=self.missed_notes + other.missed_notes,
            wrong_repeat=self.wrong_repeat + other.wrong_repeat,
            wrong_neighbor_key=self.wrong_neighbor_key + other.wrong_neighbor_key,
            wrong_not_in_window=self.wrong_not_in_window + other.wrong_not_in_window,
            wrong_no_note_nearby=self.wrong_no_note_nearby + other.wrong_no_note_nearby,
            chord_steps_total=self.chord_steps_total + other.chord_steps_total,
            chord_steps_all_hit=self.chord_steps_all_hit + other.chord_steps_all_hit,
        )


def run_multikey_diagnosis(
    player_or_path: Union[str, Path, object],
    scores_list: List[Score],
    split_items: Optional[List[dict]] = None,
) -> Dict[str, MultiKeyDiagnosisStats]:
    """Run diagnostic play on multi-key pieces and collect error and chord statistics."""
    if isinstance(player_or_path, (str, Path)):
        player = PPOPlayer(player_or_path)
    else:
        player = player_or_path

    if split_items is None:
        split_items = [{"level": "Custom"} for _ in scores_list]

    standard_rewards = RewardConfig(
        hit_exact=1.0,
        hit_off_by_one=0.5,
        wrong_press=-0.5,
        miss=-1.0,
    )
    env = MultiKeyPianoEnv(scores=scores_list, seed=42, reward_config=standard_rewards)

    level_stats: Dict[str, MultiKeyDiagnosisStats] = {}
    overall_stats = MultiKeyDiagnosisStats()

    for i, item in enumerate(split_items):
        lvl_name = f"Level {item['level']}"
        if lvl_name not in level_stats:
            level_stats[lvl_name] = MultiKeyDiagnosisStats()
        lvl = level_stats[lvl_name]

        obs, info = env.reset(options={"piece_index": i})
        targets = env.targets

        lvl.pieces_count += 1
        overall_stats.pieces_count += 1
        lvl.total_notes += len(targets)
        overall_stats.total_notes += len(targets)

        # Track chord steps in ground truth
        step_target_counts: Dict[int, int] = {}
        for t in targets:
            s = t["start_step"]
            step_target_counts[s] = step_target_counts.get(s, 0) + 1

        chord_steps = [s for s, cnt in step_target_counts.items() if cnt >= 2]
        lvl.chord_steps_total += len(chord_steps)
        overall_stats.chord_steps_total += len(chord_steps)

        terminated = False
        truncated = False
        prev_wrong = 0

        while not (terminated or truncated):
            step_before = env.current_step
            obs_before = obs
            action = player.act(obs)

            act_arr = np.asarray(action)
            struck_pitches = [21 + k for k in range(NUM_PIANO_KEYS) if act_arr[k] > 0]

            lvl.total_presses += len(struck_pitches)
            overall_stats.total_presses += len(struck_pitches)

            # Check for wrong strikes by inspecting matches against targets
            obs, reward, terminated, truncated, info = env.step(act_arr)

            curr_wrong = info["wrong_presses"]
            if curr_wrong > prev_wrong:
                num_new_wrong = curr_wrong - prev_wrong
                # Identify which struck pitches were unmatched
                for p in struck_pitches:
                    # Check if p was matched to any target
                    matched = any(
                        t["matched"] and t["pitch"] == p and abs(t["start_step"] - step_before) <= 1
                        for t in env.targets
                    )
                    if not matched:
                        cat = categorize_multikey_wrong_press(p, step_before, targets, obs_before)
                        if cat == "repeat":
                            lvl.wrong_repeat += 1
                            overall_stats.wrong_repeat += 1
                        elif cat == "neighbor_key":
                            lvl.wrong_neighbor_key += 1
                            overall_stats.wrong_neighbor_key += 1
                        elif cat == "not_in_window":
                            lvl.wrong_not_in_window += 1
                            overall_stats.wrong_not_in_window += 1
                        else:
                            lvl.wrong_no_note_nearby += 1
                            overall_stats.wrong_no_note_nearby += 1

                prev_wrong = curr_wrong

        # End of piece: update hits, misses, and check which chords had ALL notes hit
        lvl.hits_exact += info["hits_exact"]
        lvl.hits_off_by_one += info["hits_off_by_one"]
        lvl.missed_notes += info["missed_notes"]

        overall_stats.hits_exact += info["hits_exact"]
        overall_stats.hits_off_by_one += info["hits_off_by_one"]
        overall_stats.missed_notes += info["missed_notes"]

        for s in chord_steps:
            chord_targets = [t for t in targets if t["start_step"] == s]
            if all(t["matched"] for t in chord_targets):
                lvl.chord_steps_all_hit += 1
                overall_stats.chord_steps_all_hit += 1

    results = {**level_stats, "Overall": overall_stats}
    return results
