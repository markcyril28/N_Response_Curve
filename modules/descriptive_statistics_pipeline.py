from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from n_response_curve.pipeline.descriptive_statistics_pipeline import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
