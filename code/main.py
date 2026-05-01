#!/usr/bin/env python3
"""
Eval entry-point wrapper (recommended by HACKERRANK README).
Delegates to agent.py orchestration pipeline.
"""

import sys

from agent import main

if __name__ == "__main__":
    main(sys.argv)
