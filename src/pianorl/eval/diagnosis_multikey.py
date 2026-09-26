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
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from pianorl.env.multi_keys_env import MultiKeyPianoEnv
from pianorl.env.free_keys_env import split_observation, RewardConfig
from pianorl.eval.players import PPOPlayer
from pianorl.score import Score, NUM_PIANO_KEYS, load_score, pitch_to_note_name


@dataclass
class RepeatErrorDetail:
    """Detailed record of a wrong press classified as Repeat."""
    step: int
    time_sec: float
    beat: float
    pitch: int
    pitch_name: str
    same_pitch_notes: List[dict]
    other_notes: List[dict]


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

    total_mismatch_notes: int = 0
    missed_mismatch_notes: int = 0
    wrong_press_mismatch: int = 0

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
            total_mismatch_notes=self.total_mismatch_notes + other.total_mismatch_notes,
            missed_mismatch_notes=self.missed_mismatch_notes + other.missed_mismatch_notes,
            wrong_press_mismatch=self.wrong_press_mismatch + other.wrong_press_mismatch,
        )


def target_has_duration_mismatch(t: dict, all_targets: List[dict], threshold_beats: float = 1.0, threshold_ratio: float = 2.0) -> bool:
    d1 = t.get("duration_beats", 0.0)
    for t2 in all_targets:
        if t2 is t:
            continue
        if abs(t2["start_step"] - t["start_step"]) <= 2 and abs(t2["pitch"] - t["pitch"]) <= 2:
            d2 = t2.get("duration_beats", 0.0)
            if min(d1, d2) > 0 and max(d1, d2) > threshold_ratio * min(d1, d2):
                return True
            if abs(d1 - d2) > threshold_beats:
                return True
    return False


def run_multikey_diagnosis(
    player_or_path: Union[str, Path, object],
    scores_list: List[Score],
    split_items: Optional[List[dict]] = None,
    analyze_mismatch: bool = False,
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

        if analyze_mismatch:
            for t in targets:
                if target_has_duration_mismatch(t, targets):
                    lvl.total_mismatch_notes += 1
                    overall_stats.total_mismatch_notes += 1

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

                        if analyze_mismatch:
                            # Check if nearest active targets have duration mismatch
                            for tgt in targets:
                                if abs(tgt["start_step"] - step_before) <= 2 and abs(tgt["pitch"] - p) <= 2:
                                    if target_has_duration_mismatch(tgt, targets):
                                        lvl.wrong_press_mismatch += 1
                                        overall_stats.wrong_press_mismatch += 1
                                        break

                prev_wrong = curr_wrong

        # End of piece: update hits, misses, and check which chords had ALL notes hit
        lvl.hits_exact += info["hits_exact"]
        lvl.hits_off_by_one += info["hits_off_by_one"]
        lvl.missed_notes += info["missed_notes"]

        overall_stats.hits_exact += info["hits_exact"]
        overall_stats.hits_off_by_one += info["hits_off_by_one"]
        overall_stats.missed_notes += info["missed_notes"]

        if analyze_mismatch:
            for t in targets:
                if not t.get("matched", False):
                    if target_has_duration_mismatch(t, targets):
                        lvl.missed_mismatch_notes += 1
                        overall_stats.missed_mismatch_notes += 1

        for s in chord_steps:
            chord_targets = [t for t in targets if t["start_step"] == s]
            if all(t["matched"] for t in chord_targets):
                lvl.chord_steps_all_hit += 1
                overall_stats.chord_steps_all_hit += 1

    results = {**level_stats, "Overall": overall_stats}
    return results


def run_multikey_midi_segment_diagnosis(
    player_or_path: Union[str, Path, object],
    score_or_path: Union[str, Path, Score],
    split_seconds: Optional[Union[float, List[float]]] = 30.0,
    segment_seconds: Optional[float] = None,
    return_repeat_errors: bool = False,
    repeat_window_beats: float = 0.5,
    analyze_mismatch: bool = False,
) -> Union[
    Tuple[Dict[str, MultiKeyDiagnosisStats], Score],
    Tuple[Dict[str, MultiKeyDiagnosisStats], Score, List[RepeatErrorDetail]],
]:
    """Diagnose error patterns and chord hit rates for a single MIDI/score piece split across time segments.

    Parameters
    ----------
    player_or_path : PPO model path or player instance with .act(obs).
    score_or_path : Path to MIDI / JSON score file, or an existing Score instance.
    split_seconds : Boundary cutoff second(s), e.g. 30.0 for [0s-30s, 30s-end].
    segment_seconds : Uniform segment duration in seconds, e.g. 30.0 for [0s-30s, 30s-60s, ...]. Overrides split_seconds.
    return_repeat_errors : If True, also return a list of RepeatErrorDetail records for every repeat strike.
    repeat_window_beats : Half-window size in beats for active score note context (default: 0.5 beats = +-2 steps).

    Returns
    -------
    results : Dict mapping segment label (e.g. '0.0s - 30.0s', 'Overall') to MultiKeyDiagnosisStats.
    score : The loaded Score instance.
    repeat_errors : (Optional, if return_repeat_errors is True) List of RepeatErrorDetail objects.
    """
    if isinstance(player_or_path, (str, Path)):
        player = PPOPlayer(player_or_path)
    else:
        player = player_or_path

    if isinstance(score_or_path, (str, Path)):
        score = load_score(score_or_path)
    else:
        score = score_or_path

    tempo = float(score.tempo_bpm) if (score.tempo_bpm and score.tempo_bpm > 0) else 120.0
    steps_per_beat = 4
    sec_per_beat = 60.0 / tempo
    sec_per_step = sec_per_beat / steps_per_beat

    # Calculate total duration in seconds
    max_end_beat = max((n.start_beat + n.duration_beats for n in score.notes), default=0.0)
    duration_seconds = max_end_beat * sec_per_beat

    # Construct segment intervals: [(start_sec, end_sec, label), ...]
    intervals: List[Tuple[float, float, str]] = []
    if segment_seconds is not None and segment_seconds > 0:
        cur = 0.0
        while cur < duration_seconds:
            nxt = min(cur + segment_seconds, duration_seconds)
            intervals.append((cur, nxt, f"{cur:.1f}s - {nxt:.1f}s"))
            if nxt >= duration_seconds:
                break
            cur = nxt
    else:
        if split_seconds is None:
            split_list = [30.0]
        elif isinstance(split_seconds, (int, float)):
            split_list = [float(split_seconds)]
        else:
            split_list = sorted([float(s) for s in split_seconds])

        valid_cutoffs = sorted(set(s for s in split_list if 0 < s < duration_seconds))
        prev = 0.0
        for c in valid_cutoffs:
            intervals.append((prev, c, f"{prev:.1f}s - {c:.1f}s"))
            prev = c
        intervals.append((prev, duration_seconds, f"{prev:.1f}s - {duration_seconds:.1f}s"))

    if not intervals:
        intervals.append((0.0, max(duration_seconds, 1.0), f"0.0s - {max(duration_seconds, 1.0):.1f}s"))

    num_intervals = len(intervals)
    segment_stats = [MultiKeyDiagnosisStats(pieces_count=1) for _ in range(num_intervals)]
    overall_stats = MultiKeyDiagnosisStats(pieces_count=1)

    def find_interval_idx(t_sec: float) -> int:
        for idx, (s, e, _) in enumerate(intervals):
            if idx == num_intervals - 1:
                if t_sec >= s:
                    return idx
            else:
                if s <= t_sec < e:
                    return idx
        return num_intervals - 1

    standard_rewards = RewardConfig(
        hit_exact=1.0,
        hit_off_by_one=0.5,
        wrong_press=-0.5,
        miss=-1.0,
    )
    env = MultiKeyPianoEnv(scores=[score], seed=42, reward_config=standard_rewards)
    obs, info = env.reset(options={"piece_index": 0})
    targets = env.targets

    # 1. Attribute targets to segments
    for t in targets:
        t_sec = t["start_step"] * sec_per_step
        seg_idx = find_interval_idx(t_sec)
        segment_stats[seg_idx].total_notes += 1
        overall_stats.total_notes += 1
        if analyze_mismatch and target_has_duration_mismatch(t, targets):
            segment_stats[seg_idx].total_mismatch_notes += 1
            overall_stats.total_mismatch_notes += 1

    # 2. Track chords
    step_target_counts: Dict[int, int] = {}
    for t in targets:
        s = t["start_step"]
        step_target_counts[s] = step_target_counts.get(s, 0) + 1

    chord_steps = [s for s, cnt in step_target_counts.items() if cnt >= 2]
    for s in chord_steps:
        s_sec = s * sec_per_step
        seg_idx = find_interval_idx(s_sec)
        segment_stats[seg_idx].chord_steps_total += 1
        overall_stats.chord_steps_total += 1

    # 3. Step through the score
    terminated = False
    truncated = False
    prev_wrong = 0
    repeat_events: List[Tuple[int, int]] = []

    while not (terminated or truncated):
        step_before = env.current_step
        step_sec = step_before * sec_per_step
        seg_idx = find_interval_idx(step_sec)

        obs_before = obs
        action = player.act(obs)

        act_arr = np.asarray(action)
        struck_pitches = [21 + k for k in range(NUM_PIANO_KEYS) if act_arr[k] > 0]

        segment_stats[seg_idx].total_presses += len(struck_pitches)
        overall_stats.total_presses += len(struck_pitches)

        obs, reward, terminated, truncated, info = env.step(act_arr)

        curr_wrong = info["wrong_presses"]
        if curr_wrong > prev_wrong:
            # Check which pitches were unmatched
            for p in struck_pitches:
                matched = any(
                    t["matched"] and t["pitch"] == p and abs(t["start_step"] - step_before) <= 1
                    for t in (
                        env._targets_by_step.get(step_before, [])
                        + env._targets_by_step.get(step_before - 1, [])
                        + env._targets_by_step.get(step_before + 1, [])
                    )
                )
                if not matched:
                    cat = categorize_multikey_wrong_press(p, step_before, targets, obs_before)
                    if cat == "repeat":
                        segment_stats[seg_idx].wrong_repeat += 1
                        overall_stats.wrong_repeat += 1
                        if return_repeat_errors:
                            repeat_events.append((step_before, p))
                    elif cat == "neighbor_key":
                        segment_stats[seg_idx].wrong_neighbor_key += 1
                        overall_stats.wrong_neighbor_key += 1
                    elif cat == "not_in_window":
                        segment_stats[seg_idx].wrong_not_in_window += 1
                        overall_stats.wrong_not_in_window += 1
                    else:
                        segment_stats[seg_idx].wrong_no_note_nearby += 1
                        overall_stats.wrong_no_note_nearby += 1

                    if analyze_mismatch:
                        for tgt in targets:
                            if abs(tgt["start_step"] - step_before) <= 2 and abs(tgt["pitch"] - p) <= 2:
                                if target_has_duration_mismatch(tgt, targets):
                                    segment_stats[seg_idx].wrong_press_mismatch += 1
                                    overall_stats.wrong_press_mismatch += 1
                                    break

            prev_wrong = curr_wrong

    # 4. Attribute hits and misses
    for t in targets:
        t_sec = t["start_step"] * sec_per_step
        seg_idx = find_interval_idx(t_sec)
        m_type = t.get("match_type")
        if m_type == "exact":
            segment_stats[seg_idx].hits_exact += 1
            overall_stats.hits_exact += 1
        elif m_type == "off":
            segment_stats[seg_idx].hits_off_by_one += 1
            overall_stats.hits_off_by_one += 1
        else:
            segment_stats[seg_idx].missed_notes += 1
            overall_stats.missed_notes += 1
            if analyze_mismatch and target_has_duration_mismatch(t, targets):
                segment_stats[seg_idx].missed_mismatch_notes += 1
                overall_stats.missed_mismatch_notes += 1

    # 5. Check chord completion
    for s in chord_steps:
        s_sec = s * sec_per_step
        seg_idx = find_interval_idx(s_sec)
        chord_targets = env._targets_by_step.get(s, [])
        if all(t["matched"] for t in chord_targets):
            segment_stats[seg_idx].chord_steps_all_hit += 1
            overall_stats.chord_steps_all_hit += 1

    results: Dict[str, MultiKeyDiagnosisStats] = {}
    for (_, _, label), stats in zip(intervals, segment_stats):
        results[label] = stats
    results["Overall"] = overall_stats

    if return_repeat_errors:
        repeat_errors: List[RepeatErrorDetail] = []
        window_steps = int(round(repeat_window_beats * steps_per_beat))

        for step_idx, struck_pitch in repeat_events:
            t_sec = step_idx * sec_per_step
            t_beat = step_idx / steps_per_beat
            p_name = pitch_to_note_name(struck_pitch)

            same_pitch_notes = []
            other_notes = []

            min_s = max(0, step_idx - getattr(env, "_max_dur_steps", 16) - window_steps)
            max_s = step_idx + window_steps
            for s in range(min_s, max_s + 1):
                for t in env._targets_by_step.get(s, []):
                    dur_steps = max(1, int(round(t["duration_beats"] * steps_per_beat)))
                    end_step = t["start_step"] + dur_steps
                    if t["start_step"] <= step_idx + window_steps and end_step >= step_idx - window_steps:
                        note_info = {
                            "pitch": t["pitch"],
                            "pitch_name": pitch_to_note_name(t["pitch"]),
                            "start_step": t["start_step"],
                            "start_beat": t["start_step"] / steps_per_beat,
                            "duration_beats": t["duration_beats"],
                            "duration_steps": dur_steps,
                            "end_step": end_step,
                            "end_beat": end_step / steps_per_beat,
                            "matched": t["matched"],
                            "match_type": t.get("match_type"),
                        }
                        if t["pitch"] == struck_pitch:
                            same_pitch_notes.append(note_info)
                        else:
                            other_notes.append(note_info)

            same_pitch_notes.sort(key=lambda x: x["start_step"])
            other_notes.sort(key=lambda x: (x["start_step"], x["pitch"]))

            repeat_errors.append(
                RepeatErrorDetail(
                    step=step_idx,
                    time_sec=t_sec,
                    beat=t_beat,
                    pitch=struck_pitch,
                    pitch_name=p_name,
                    same_pitch_notes=same_pitch_notes,
                    other_notes=other_notes,
                )
            )

        return results, score, repeat_errors

    return results, score


def print_repeat_errors_list(
    repeat_errors: List[RepeatErrorDetail],
    window_beats: float = 0.5,
    max_other_notes: int = 8,
) -> None:
    """Print a clear, human-readable list of Repeat errors with timing and active score notes."""
    print("=" * 125)
    print(
        f"REPEAT ERROR DETAILS (Total: {len(repeat_errors)} instances, "
        f"active window: ±{window_beats:.2f} beats / ±{int(round(window_beats * 4))} steps)"
    )
    print("=" * 125)

    if not repeat_errors:
        print("  No repeat errors detected! The model made 0 double-strikes.\n")
        return

    for idx, err in enumerate(repeat_errors, 1):
        print(
            f"[{idx:>3}] Step: {err.step:<5} | Time: {err.time_sec:6.2f}s (Beat {err.beat:6.2f}) | "
            f"Struck: Pitch {err.pitch:<3} ({err.pitch_name})"
        )

        if err.same_pitch_notes:
            print("      Target notes with SAME pitch in window:")
            for note in err.same_pitch_notes:
                match_str = f"Matched: {note['matched']}"
                if note.get("match_type"):
                    match_str += f" ({note['match_type']})"
                print(
                    f"        -> Pitch {note['pitch']:<3} ({note['pitch_name']:<3}) | "
                    f"Beat {note['start_beat']:6.2f} - {note['end_beat']:6.2f} "
                    f"(Step {note['start_step']:<5} - {note['end_step']:<5}, dur {note['duration_beats']:.2f}b) | "
                    f"{match_str}"
                )
        else:
            print("      Target notes with SAME pitch in window: None (struck unprovoked)")

        if err.other_notes:
            num_other = len(err.other_notes)
            shown_notes = err.other_notes[:max_other_notes]
            print(f"      Other active score notes in window ({num_other} total):")
            for note in shown_notes:
                match_str = f"Matched: {note['matched']}"
                if note.get("match_type"):
                    match_str += f" ({note['match_type']})"
                print(
                    f"           Pitch {note['pitch']:<3} ({note['pitch_name']:<3}) | "
                    f"Beat {note['start_beat']:6.2f} - {note['end_beat']:6.2f} "
                    f"(Step {note['start_step']:<5} - {note['end_step']:<5}, dur {note['duration_beats']:.2f}b) | "
                    f"{match_str}"
                )
            if num_other > max_other_notes:
                print(f"           ... and {num_other - max_other_notes} more concurrent notes in chord")
        else:
            print("      Other active score notes in window: None")

        print("-" * 125)
    print()
