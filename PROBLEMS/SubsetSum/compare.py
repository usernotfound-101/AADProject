import time
import os
import matplotlib.pyplot as plt

from brute_force import subset_sum_bruteforce
from greedy_approx import subset_sum_greedy
from FPTAS import subset_sum_FPTAS
from dp_approach import subset_sum_DP


def load_instances(path):
    instances = []
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    for i in range(0, len(lines), 2):
        arr = list(map(int, lines[i].split()))
        target = int(lines[i + 1])
        instances.append((arr, target))
    return instances


def evaluate_instances(instances, allow_brute, repeats=2):
    sizes = []
    times = {"brute": [], "greedy": [], "fptas": [], "dp": []}
    err = {"greedy": [], "fptas": []}

    for arr, target in instances:
        sizes.append(len(arr))
        if len(arr) > 27:
            allow_brute = False
        opt_sum = target

        t = 0
        for _ in range(repeats):
            start = time.perf_counter()
            subset_sum_DP(arr, target)
            t += time.perf_counter() - start
        times["dp"].append(t / repeats)

        t = 0
        for _ in range(repeats):
            start = time.perf_counter()
            ok, subset = subset_sum_greedy(arr, target)
            g = sum(subset)
            t += time.perf_counter() - start
        times["greedy"].append(t / repeats)
        err["greedy"].append(1 - (g / opt_sum))

        t = 0
        for _ in range(repeats):
            start = time.perf_counter()
            ok, fsum = subset_sum_FPTAS(arr, target, eps=0.05)
            t += time.perf_counter() - start
        times["fptas"].append(t / repeats)
        err["fptas"].append(1 - (fsum / opt_sum))

        if allow_brute:
            t = 0
            for _ in range(repeats):
                start = time.perf_counter()
                subset_sum_bruteforce(arr, target)
                t += time.perf_counter() - start
            times["brute"].append(t / repeats)
        else:
            times["brute"].append(None)

    return sizes, times, err


def average_by_length(sizes, metric_dict):
    unique = sorted(set(sizes))
    avg = {k: [] for k in metric_dict}

    for u in unique:
        indices = [i for i, s in enumerate(sizes) if s == u]
        for key in metric_dict:
            vals = [metric_dict[key][i] for i in indices if metric_dict[key][i] is not None]
            avg[key].append(sum(vals) / len(vals) if vals else None)

    return unique, avg


def plot_results(title, sizes, times, err, include_brute):
    os.makedirs("results", exist_ok=True)

    sizes, avg_times = average_by_length(sizes, times)
    _, avg_err = average_by_length(sizes, err)

    plt.figure(figsize=(8, 5))
    if include_brute:
        plt.plot(sizes, avg_times["brute"], marker="o", label="Brute Force")
    plt.plot(sizes, avg_times["greedy"], marker="o", label="Greedy (1/2)")
    plt.plot(sizes, avg_times["fptas"], marker="o", label="FPTAS")
    plt.plot(sizes, avg_times["dp"], marker="o", label="DP (OPT)")
    plt.yscale("log")
    plt.xlabel("Instance size (n)")
    plt.ylabel("Runtime (log seconds)")
    plt.title(f"{title} — Runtime Comparison")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{title}_runtime.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(sizes, avg_err["greedy"], marker="o", label="Greedy Error")
    plt.plot(sizes, avg_err["fptas"], marker="o", label="FPTAS Error")
    plt.xlabel("Instance size (n)")
    plt.ylabel("Error (1 - accuracy)")
    plt.ylim(0, 0.15)
    plt.title(f"{title} — Error Comparison")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{title}_error.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    easy_instances = load_instances("data/easy.txt")
    hard_instances = load_instances("data/hard.txt")
    worst_instances = load_instances("data/worst_case.txt")

    sizes_e, times_e, err_e = evaluate_instances(easy_instances, allow_brute=True)
    sizes_h, times_h, err_h = evaluate_instances(hard_instances, allow_brute=True)
    sizes_w, times_w, err_w = evaluate_instances(worst_instances, allow_brute=False)

    plot_results("EASY_Dataset", sizes_e, times_e, err_e, include_brute=True)
    plot_results("HARD_Dataset", sizes_h, times_h, err_h, include_brute=True)
    plot_results("HARDEST_Dataset", sizes_w, times_w, err_w, include_brute=False)

    print("Graphs saved in results/")
