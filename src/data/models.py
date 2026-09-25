# src/data/models.py

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Segment:
    """Represents a timed transcription segment (e.g. from Whisper-style ASR).

    Holds start/end timestamps, bilingual text (JA/EN), and optional model quality metrics.
    All optional fields default to None when absent.
    """
    id: int
    start: float
    end: float
    ja: str = ""
    en: str = ""
    confidence: Optional[float] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    compression_ratio: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Segment":
        """Build a Segment from a raw dict, ignoring unknown keys.

        Only the fields declared on the dataclass are taken from `d`.
        Missing keys become None (or the field's default).
        """
        fields = cls.__dataclass_fields__.keys()

        return cls(**{k: d[k] for k in fields if k in d})

    # Alias for backwards compatibility
    from_dic = from_dict
