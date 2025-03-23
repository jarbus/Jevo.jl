import sys
from datetime import datetime

def time_diff(logfile):
    with open(logfile, 'r') as f:
        lines = f.read().strip().split('\n')

    # Parse first and last lines
    first_line = lines[0].split()
    last_line = lines[-1].split()

    # Combine date and time fields
    first_dt_str = f"{first_line[0]} {first_line[1]}"
    last_dt_str = f"{last_line[0]} {last_line[1]}"

    # Convert to datetime objects
    fmt = "%y-%m-%d %H:%M:%S"  # Corrected format
    first_dt = datetime.strptime(first_dt_str, fmt)
    last_dt = datetime.strptime(last_dt_str, fmt)

    return last_dt - first_dt

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <logfile>")
        sys.exit(1)

    logfile = sys.argv[1]
    difference = time_diff(logfile)
    print(logfile, difference)
