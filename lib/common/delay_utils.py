import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

TIMING_KEYS: List[str] = [
    "gen_crop_cameras",
    "warp_make_inputs",
    "feature_extractor",
    "temporal_rnn",
    "skeleton_encoder",
    "pose_regressor",
    "crop_total",
    "ume_total",
    "overall_total",
]


@dataclass
class TimingStats:
    totals: Dict[str, float] = field(
        default_factory=lambda: {key: 0.0 for key in TIMING_KEYS}
    )
    frame_count: int = 0

    def add_sample(self, timing_sample: Dict[str, float]) -> None:
        self.frame_count += 1
        for key in TIMING_KEYS:
            self.totals[key] += timing_sample.get(key, 0.0)

    def merge(self, other: Optional["TimingStats"]) -> None:
        if other is None or other.frame_count == 0:
            return
        self.frame_count += other.frame_count
        for key in TIMING_KEYS:
            self.totals[key] += other.totals.get(key, 0.0)

    def averages(self) -> Dict[str, float]:
        if self.frame_count == 0:
            return {key: 0.0 for key in TIMING_KEYS}
        return {
            key: self.totals[key] / self.frame_count
            for key in TIMING_KEYS
        }


def format_summary_lines(script_name: str, stats: TimingStats) -> List[str]:
    averages = stats.averages()
    header = datetime.datetime.now().isoformat()
    lines = [
        f"[{header}] {script_name}",
        f"Frames processed: {stats.frame_count}",
    ]
    lines.append("Average timings (ms):")
    for key in TIMING_KEYS:
        lines.append(f"  {key}: {averages[key] * 1000.0:.3f}")
    return lines


def append_delay_log(script_name: str, stats: TimingStats, log_path: str) -> None:
    lines = format_summary_lines(script_name, stats)
    with open(log_path, "a", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
        fp.write("\n")

