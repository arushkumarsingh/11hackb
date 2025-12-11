"""
Simple data models for timeline cuts.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TimeCut:
    """Represents a single time cut with start and end times."""
    start: float
    end: float


@dataclass
class TimeLineCutList:
    """Container for a list of time cuts."""
    timeline: List[TimeCut]
    
    def __len__(self):
        return len(self.timeline)

