import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def parse_time(timestr):
    """Convert HH:MM:SS to total seconds."""
    h, m, s = map(int, timestr.strip().split(':'))
    return h*3600 + m*60 + s

def main(filename):
    times_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            # Each line format: <parent_dir>/<some_subdir>/run.log HH:MM:SS
            path, time_str = line.rsplit(' ', 1)
            parent_dir = path.split('/')[0]
            total_seconds = parse_time(time_str)
            times_dict.setdefault(parent_dir, []).append(total_seconds)

    # Prepare data for plotting
    labels = list(times_dict.keys())
    data = [times_dict[label] for label in labels]

    plt.boxplot(data, labels=labels, vert=True)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Directory")
    plt.ylabel("Time (seconds)")
    plt.title("Box and Whisker Plot of Run Times")
    plt.tight_layout()
    plt.savefig("times.png")

if __name__ == "__main__":
    main("times.txt")

