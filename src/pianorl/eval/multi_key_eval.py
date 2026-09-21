"""Multi-key evaluation metrics and evaluation runner for polyphonic piano play.

Calculates Precision, Recall, F1 score, Exact Rate, and chord accuracy
when comparing binary arrays of model key presses against ground-truth targets.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union
import numpy as np

from pianorl.env.multi_keys_env import MultiKeyPianoEnv
from pianorl.env.free_keys_env import RewardConfig
from pianorl.score import Score, NUM_PIANO_KEYS


@dataclass
class MultiKeyBinaryMetrics:
    """Evaluation metrics from directly comparing binary press arrays to target arrays."""
    true_positives: int
    false_positives: int
    false_negatives: int
    total_targets: int
    total_presses: int
    precision: float
    recall: float
    f1: float
    exact_rate: float
    step_exact_rate: float
    chord_exact_rate: float
    total_chord_steps: int


@dataclass
class MultiKeyEpisodeCounters:
    """Accumulated counts from MultiKeyPianoEnv episodes."""
    hits_exact: int = 0
    hits_off_by_one: int = 0
    wrong_presses: int = 0
    missed_notes: int = 0
    total_notes: int = 0
    total_reward: float = 0.0
    chord_steps_total: int = 0
    chord_steps_exact: int = 0

    def __add__(self, other: "MultiKeyEpisodeCounters") -> "MultiKeyEpisodeCounters":
        return MultiKeyEpisodeCounters(
            hits_exact=self.hits_exact + other.hits_exact,
            hits_off_by_one=self.hits_off_by_one + other.hits_off_by_one,
            wrong_presses=self.wrong_presses + other.wrong_presses,
            missed_notes=self.missed_notes + other.missed_notes,
            total_notes=self.total_notes + other.total_notes,
            total_reward=self.total_reward + other.total_reward,
            chord_steps_total=self.chord_steps_total + other.chord_steps_total,
            chord_steps_exact=self.chord_steps_exact + other.chord_steps_exact,
        )


@dataclass
class MultiKeyEvaluationMetrics:
    """Final graded summary metrics for multi-key evaluation."""
    num_pieces: int
    total_notes: int
    precision: float
    recall: float
    f1: float
    exact_rate: float
    mean_reward: float
    chord_exact_rate: float


def compute_array_metrics(
    presses: np.ndarray,
    targets: np.ndarray,
) -> MultiKeyBinaryMetrics:
    """Compare binary arrays of key strikes against target keys.

    Args:
        presses: Binary array of shape (88,) or (num_steps, 88).
        targets: Binary array of shape (88,) or (num_steps, 88).

    Returns:
        MultiKeyBinaryMetrics with precision, recall, F1, exact rate, etc.
    """
    p = np.asarray(presses) > 0
    t = np.asarray(targets) > 0

    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch: presses {p.shape} vs targets {t.shape}")

    if p.ndim == 1:
        p = p[np.newaxis, :]
        t = t[np.newaxis, :]

    num_steps = p.shape[0]

    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t))

    total_targets = tp + fn
    total_presses = tp + fp

    # Edge cases: 0 presses and 0 targets = 1.0, 0 presses with targets = 0.0
    if total_presses == 0 and total_targets == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        precision = float(tp / total_presses) if total_presses > 0 else 0.0
        recall = float(tp / total_targets) if total_targets > 0 else 1.0
        denom = precision + recall
        f1 = float(2.0 * precision * recall / denom) if denom > 0 else 0.0

    exact_rate = float(tp / total_targets) if total_targets > 0 else 1.0

    # Step-level exact matches (all 88 keys correct)
    step_matches = int(np.sum(np.all(p == t, axis=1)))
    step_exact_rate = float(step_matches / num_steps) if num_steps > 0 else 1.0

    # Chord steps: steps where at least 2 notes are scheduled
    target_counts_per_step = np.sum(t, axis=1)
    chord_mask = target_counts_per_step >= 2
    total_chord_steps = int(np.sum(chord_mask))

    if total_chord_steps > 0:
        chord_matches = int(np.sum(np.all(p[chord_mask] == t[chord_mask], axis=1)))
        chord_exact_rate = float(chord_matches / total_chord_steps)
    else:
        chord_exact_rate = 1.0

    return MultiKeyBinaryMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        total_targets=total_targets,
        total_presses=total_presses,
        precision=precision,
        recall=recall,
        f1=f1,
        exact_rate=exact_rate,
        step_exact_rate=step_exact_rate,
        chord_exact_rate=chord_exact_rate,
        total_chord_steps=total_chord_steps,
    )


def compute_multikey_metrics(
    counters: MultiKeyEpisodeCounters, num_pieces: int = 1
) -> MultiKeyEvaluationMetrics:
    """Compute precision, recall, F1, exact_rate, and mean reward from episode counters."""
    tp = counters.hits_exact + counters.hits_off_by_one
    fp = counters.wrong_presses
    fn = counters.missed_notes

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    denom = precision + recall
    f1 = (2.0 * precision * recall / denom) if denom > 0 else 0.0
    exact_rate = (counters.hits_exact / counters.total_notes) if counters.total_notes > 0 else 0.0
    mean_reward = (counters.total_reward / num_pieces) if num_pieces > 0 else 0.0

    chord_exact_rate = (
        (counters.chord_steps_exact / counters.chord_steps_total)
        if counters.chord_steps_total > 0
        else 1.0
    )

    return MultiKeyEvaluationMetrics(
        num_pieces=num_pieces,
        total_notes=counters.total_notes,
        precision=precision,
        recall=recall,
        f1=f1,
        exact_rate=exact_rate,
        mean_reward=mean_reward,
        chord_exact_rate=chord_exact_rate,
    )


def evaluate_multi_player(
    player,
    scores: Union[Score, List[Score]],
    split_items: Optional[List[dict]] = None,
    seed: int = 42,
) -> Dict[str, MultiKeyEvaluationMetrics]:
    """Evaluate any multi-key player in MultiKeyPianoEnv across one or more pieces.

    Args:
        player: Object with `act(observation) -> np.ndarray` or `predict(obs)`.
        scores: Score object or list of Score objects.
        split_items: Optional metadata list containing 'level' tags.
        seed: Environment random seed.

    Returns:
        Dict mapping level strings ('Level 1M', 'Overall') to MultiKeyEvaluationMetrics.
    """
    scores_list = scores if isinstance(scores, list) else [scores]
    if split_items is None:
        split_items = [{"level": "custom"} for _ in scores_list]

    standard_rewards = RewardConfig(
        hit_exact=1.0,
        hit_off_by_one=0.5,
        wrong_press=-0.5,
        miss=-1.0,
    )
    env = MultiKeyPianoEnv(scores=scores_list, seed=seed, reward_config=standard_rewards)

    level_counters: Dict[str, List[MultiKeyEpisodeCounters]] = {}
    all_counters: List[MultiKeyEpisodeCounters] = []

    for i, item in enumerate(split_items):
        obs, info = env.reset(options={"piece_index": i})
        total_reward = 0.0
        terminated = False
        truncated = False

        # Build ground-truth step target counts to evaluate chords
        step_target_counts: Dict[int, int] = {}
        for target in env.targets:
            s_step = target["start_step"]
            step_target_counts[s_step] = step_target_counts.get(s_step, 0) + 1

        chord_steps_total = sum(1 for cnt in step_target_counts.values() if cnt >= 2)
        chord_steps_exact = 0

        while not (terminated or truncated):
            current_step = env.current_step
            # Support either .act(obs) or SB3 .predict(obs)
            if hasattr(player, "act"):
                action = player.act(obs)
            else:
                action, _ = player.predict(obs, deterministic=True)

            # Check chord exactness before step modifies state
            if step_target_counts.get(current_step, 0) >= 2:
                # Due targets at current step
                due_pitches = set(
                    t["pitch"] for t in env.targets if t["start_step"] == current_step
                )
                act_arr = np.asarray(action)
                struck_pitches = set(
                    21 + idx for idx in range(NUM_PIANO_KEYS) if act_arr[idx] > 0
                )
                if struck_pitches == due_pitches:
                    chord_steps_exact += 1

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        ep_counter = MultiKeyEpisodeCounters(
            hits_exact=info["hits_exact"],
            hits_off_by_one=info["hits_off_by_one"],
            wrong_presses=info["wrong_presses"],
            missed_notes=info["missed_notes"],
            total_notes=info["total_notes"],
            total_reward=total_reward,
            chord_steps_total=chord_steps_total,
            chord_steps_exact=chord_steps_exact,
        )

        lvl_label = f"Level {item['level']}"
        if lvl_label not in level_counters:
            level_counters[lvl_label] = []
        level_counters[lvl_label].append(ep_counter)
        all_counters.append(ep_counter)

    results: Dict[str, MultiKeyEvaluationMetrics] = {}
    for lvl, cnts in level_counters.items():
        total_cnt = MultiKeyEpisodeCounters()
        for c in cnts:
            total_cnt = total_cnt + c
        results[lvl] = compute_multikey_metrics(total_cnt, num_pieces=len(cnts))

    overall_cnt = MultiKeyEpisodeCounters()
    for c in all_counters:
        overall_cnt = overall_cnt + c
    results["Overall"] = compute_multikey_metrics(overall_cnt, num_pieces=len(all_counters))

    return results


def evaluate_multikey_real_piece(
    player,
    score: Score,
    seed: int = 42,
) -> Tuple[MultiKeyEpisodeCounters, int]:
    """Step a multi-key player through a real MIDI piece in MultiKeyPianoEnv."""
    standard_rewards = RewardConfig(
        hit_exact=1.0,
        hit_off_by_one=0.5,
        wrong_press=-0.5,
        miss=-1.0,
    )
    env = MultiKeyPianoEnv(scores=[score], seed=seed, reward_config=standard_rewards)
    obs, info = env.reset(options={"piece_index": 0})

    step_target_counts: Dict[int, int] = {}
    for t in env.targets:
        s = t["start_step"]
        step_target_counts[s] = step_target_counts.get(s, 0) + 1

    chord_steps_total = sum(1 for cnt in step_target_counts.values() if cnt >= 2)
    chord_steps_exact = 0

    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        current_step = env.current_step
        action = player.act(obs)
        act_arr = np.asarray(action)

        if step_target_counts.get(current_step, 0) >= 2:
            due_pitches = set(t["pitch"] for t in env.targets if t["start_step"] == current_step)
            struck_pitches = set(21 + idx for idx in range(NUM_PIANO_KEYS) if act_arr[idx] > 0)
            if struck_pitches == due_pitches:
                chord_steps_exact += 1

        obs, reward, terminated, truncated, info = env.step(act_arr)
        total_reward += reward

    counter = MultiKeyEpisodeCounters(
        hits_exact=info["hits_exact"],
        hits_off_by_one=info["hits_off_by_one"],
        wrong_presses=info["wrong_presses"],
        missed_notes=info["missed_notes"],
        total_notes=info["total_notes"],
        total_reward=total_reward,
        chord_steps_total=chord_steps_total,
        chord_steps_exact=chord_steps_exact,
    )
    return counter, chord_steps_total

