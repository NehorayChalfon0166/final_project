"""JSON encoder that handles numpy scalars/arrays."""
import json

import numpy as np


class NpEncoder(json.JSONEncoder):
    """Convert numpy scalars/arrays so json.dump handles training metrics."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
