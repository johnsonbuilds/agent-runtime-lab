#!/usr/bin/env python3
"""Extract tool distribution statistics from runtime.log."""

import json
import re
import sys
from collections import Counter
from pathlib import Path


def parse_tool_calls(log_path: str) -> dict:
    """Parse runtime.log and extract tool call statistics."""
    tool_counts = Counter()
    
    with open(log_path, 'r') as f:
        for line in f:
            if 'tool.dispatch' in line:
                match = re.search(r"tool='([^']+)'", line)
                if match:
                    tool_counts[match.group(1)] += 1
    
    total = sum(tool_counts.values())
    
    tool_distribution = {}
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        tool_distribution[tool] = {
            "count": count,
            "share": f"{round(count / total * 100, 2)}%" if total > 0 else "0%"
        }
    
    return {
        "tool_distribution": tool_distribution,
        "total_calls": total
    }


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "runtime.log"
    
    if not Path(log_path).exists():
        print(json.dumps({"error": f"File not found: {log_path}"}))
        sys.exit(1)
    
    stats = parse_tool_calls(log_path)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
