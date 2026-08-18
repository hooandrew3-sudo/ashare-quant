"""兼容入口：python run_pipeline.py --demo"""

import sys

from quant.cli import main

if __name__ == "__main__":
    sys.exit(main())
