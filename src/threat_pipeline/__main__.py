"""Allow running as: python -m threat_pipeline"""

import threat_pipeline._suppress  # noqa: F401  — must run before torch/tf

from threat_pipeline.cli import main

main()
