import sys
import re
import json

experiment_name = sys.argv[1]
trial_number = sys.argv[2]

performer_pattern = re.compile(
    r'^(\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) Operator Performer took ([\d.]+) seconds$'
)
wave_pattern = re.compile(
    r'^(\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) gen=40 (\S+Wave): '
    r'\|([\d.]+), ([\d.]+) ± ([\d.]+), ([\d.]+)\|, (\d+) samples$'
)

def parse_log_from_stdin():
    result = []
    current_performer = None

    for line in sys.stdin:
        line = line.strip()
        perf_match = performer_pattern.match(line)
        if perf_match:
            _, time_taken = perf_match.groups()
            current_performer = {"time_taken": float(time_taken), "experiment_name": experiment_name, "trial_number": trial_number}
            continue

        wave_match = wave_pattern.match(line)
        if wave_match and current_performer is not None:
            _, wave_name, low, mean, std, high, samples = wave_match.groups()
            current_performer.update({
                "wave_name": wave_name,
                "low": float(low),
                "mean": float(mean),
                "std": float(std),
                "high": float(high),
                "samples": int(samples),
            })
            result.append(current_performer)
            current_performer = None

    json_result = json.dumps(result)
    print(json_result)
parse_log_from_stdin()
