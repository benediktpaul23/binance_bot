#!/usr/bin/env python
"""
Script to run backtest with a custom start time.
"""

import sys
from datetime import datetime, timezone
from utils.Backtest import main, parse_datetime

if __name__ == "__main__":
    # Check if a custom start time was provided as command line argument
    if len(sys.argv) > 1:
        start_time_str = sys.argv[1]
        custom_start_time = parse_datetime(start_time_str)
        
        if custom_start_time:
            print(f"Running backtest with custom start time: {custom_start_time}")
            main(custom_start_time=custom_start_time)
        else:
            print("Invalid start time format. Using current time.")
            main()
    else:
        print("No custom start time provided. Using current time.")
        main()