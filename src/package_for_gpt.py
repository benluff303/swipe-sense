# Outputs a JSON package for the GPT prompt

from typing import List
import pandas as pd
import json

def to_api_payload(
    ranked_locations: pd.DataFrame,
    phrases: List[str],
) -> str:

    """Return API payload as a JSON string:
    {
      "locations": [{rank, name}, ...],
      "phrases": [...]
    }

    Arguments are ranked_locations (dataframe) and a list of phrase strings

    """

    if "rank" in ranked_locations.columns:
        ranked_locations = ranked_locations.sort_values("rank")

    loc_records = []
    for _, row in ranked_locations.iterrows():
        name = str(row["location"])
        loc_records.append({
            "rank": int(row["rank"]),
            "name": name,
        })

    payload = {
        "locations": loc_records,
        "phrases": list(phrases),
    }

    return json.dumps(payload, indent=2, ensure_ascii=False)
