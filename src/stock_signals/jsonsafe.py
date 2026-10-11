"""Convert values to plain JSON-safe Python (for the API and saved summaries)."""

import math
from datetime import datetime
from typing import Any


def clean(value: Any) -> Any:
    """NaN/inf → None, numpy/pandas scalars → Python, datetimes → ISO strings."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, str | bytes):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
