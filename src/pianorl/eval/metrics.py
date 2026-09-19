from dataclasses import dataclass
from typing import Iterable


@dataclass
class EpisodeCounters:
    """Raw event counts accumulated across one or more evaluated episodes."""
    hits_exact: int = 0
    hits_off_by_one: int = 0
    wrong_presses: int = 0
    missed_notes: int = 0
    total_notes: int = 0
    total_reward: float = 0.0

    def __add__(self, other: "EpisodeCounters") -> "EpisodeCounters":
        return EpisodeCounters(
            hits_exact=self.hits_exact + other.hits_exact,
            hits_off_by_one=self.hits_off_by_one + other.hits_off_by_one,
            wrong_presses=self.wrong_presses + other.wrong_presses,
            missed_notes=self.missed_notes + other.missed_notes,
            total_notes=self.total_notes + other.total_notes,
            total_reward=self.total_reward + other.total_reward,
        )


@dataclass
class EvaluationMetrics:
    """Grading metrics computed from aggregated episode counters."""
    num_pieces: int
    total_notes: int
    precision: float
    recall: float
    f1: float
    exact_rate: float
    mean_reward: float


def compute_metrics(counters: EpisodeCounters, num_pieces: int = 1) -> EvaluationMetrics:
    """Compute precision, recall, F1, exact_rate, and mean reward from counters.

    Formulas:
        TP = hits_exact + hits_off_by_one
        FP = wrong_presses
        FN = missed_notes
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        exact_rate = hits_exact / total_notes
        mean_reward = total_reward / num_pieces
    """
    tp = counters.hits_exact + counters.hits_off_by_one
    fp = counters.wrong_presses
    fn = counters.missed_notes

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    exact_rate = (counters.hits_exact / counters.total_notes) if counters.total_notes > 0 else 0.0
    mean_reward = (counters.total_reward / num_pieces) if num_pieces > 0 else 0.0

    return EvaluationMetrics(
        num_pieces=num_pieces,
        total_notes=counters.total_notes,
        precision=precision,
        recall=recall,
        f1=f1,
        exact_rate=exact_rate,
        mean_reward=mean_reward,
    )


def aggregate_and_compute(counters_list: Iterable[EpisodeCounters]) -> EvaluationMetrics:
    """Sum episode counters across multiple pieces, then compute aggregate metrics."""
    total_counters = EpisodeCounters()
    num_pieces = 0
    for c in counters_list:
        total_counters = total_counters + c
        num_pieces += 1
    return compute_metrics(total_counters, num_pieces=num_pieces)
