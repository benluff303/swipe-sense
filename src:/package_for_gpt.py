# Outputs a JSON package for the GPT prompt

#from __future__ import annotations

from typing import List, Dict, Any
import pandas as pd
import re

_slug_re = re.compile(r"[^a-z0-9]+")

def _slugify(name: str) -> str:
    s = name.strip().lower()
    return _slug_re.sub("-", s).strip("-")

def to_api_payload(
    ranked_locations: pd.DataFrame,
    phrases: List[str],
) -> Dict[str, Any]:
    """Return API payload (no version field):
    {
      "locations": [{rank, name, score, id}, ...],
      "phrases": [...]
    }
    """
    if "rank" in ranked_locations.columns:
        ranked_locations = ranked_locations.sort_values("rank")

    loc_records = []
    for _, row in ranked_locations.iterrows():
        name = str(row["location"])
        loc_records.append({
            "rank": int(row["rank"]),
            "name": name,
            "score": float(row.get("combined_score", 0.0)),
            "id": _slugify(name),
        })

    return {
        "locations": loc_records,
        "phrases": list(phrases),
    }
