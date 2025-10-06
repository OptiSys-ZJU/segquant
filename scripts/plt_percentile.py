import os
import torch
import matplotlib.pyplot as plt


def visualize_anomalies(folder, idx=-1):
    linear_names = torch.load(os.path.join(folder, "linear_names.pt"))
    before_stat, after_stat = torch.load(os.path.join(folder, "linear_input_stats.pt"))

    indices = range(len(linear_names)) if idx < 0 else [idx]

    for i in indices:
        pair_path = os.path.join(folder, f"pairs_{i}_chunk_0.pt")
        if not os.path.exists(pair_path):
            print(f"[Skip] Missing {pair_path}")
            continue

        data = torch.load(pair_path)
        anomaly_scores = data["anomaly_scores"].float().cpu()

        # ---- before ----
        stats_before = before_stat[linear_names[i]]
        mean_b = stats_before["mean"].cpu()
        min_b = stats_before["min"].cpu()
        max_b = stats_before["max"].cpu()
        p1_b = stats_before["1%"].cpu()
        p99_b = stats_before["99%"].cpu()
        p25_b = stats_before["25%"].cpu()
        p75_b = stats_before["75%"].cpu()

        # ---- after ----
        stats_after = after_stat[linear_names[i]]
        mean_a = stats_after["mean"].cpu()
        min_a = stats_after["min"].cpu()
        max_a = stats_after["max"].cpu()
        p1_a = stats_after["1%"].cpu()
        p99_a = stats_after["99%"].cpu()
        p25_a = stats_after["25%"].cpu()
        p75_a = stats_after["75%"].cpu()

        channels = torch.arange(len(anomaly_scores))

        fig, (ax1, ax2, ax3) = plt.subplots(
            3,
            1,
            figsize=(14, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 2, 1]},
        )

        # ---- before ----
        ax1.fill_between(
            channels, min_b, max_b, color="green", alpha=0.2, label="min-max"
        )
        ax1.fill_between(channels, p1_b, p99_b, color="gray", alpha=0.3, label="1%-99%")
        ax1.fill_between(
            channels, p25_b, p75_b, color="blue", alpha=0.3, label="25%-75%"
        )
        ax1.plot(channels, mean_b, label="Mean", color="blue", alpha=0.8, linewidth=0.8)
        ax1.set_ylabel("Before stats")
        ax1.set_title(f"Layer {i}: {linear_names[i]}")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # ---- after ----
        ax2.fill_between(
            channels, min_a, max_a, color="green", alpha=0.2, label="min-max"
        )
        ax2.fill_between(channels, p1_a, p99_a, color="gray", alpha=0.3, label="1%-99%")
        ax2.fill_between(
            channels, p25_a, p75_a, color="blue", alpha=0.3, label="25%-75%"
        )
        ax2.plot(channels, mean_a, label="Mean", color="blue", alpha=0.8, linewidth=0.8)
        ax2.set_ylabel("After stats")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # ---- 统一 before/after y 轴范围 ----
        ymin = min(min_b.min().item(), min_a.min().item())
        ymax = max(max_b.max().item(), max_a.max().item())
        ax1.set_ylim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)

        # ---- anomaly scores ----
        ax3.plot(
            channels, anomaly_scores, label="Anomaly score", color="red", linewidth=0.8
        )
        ax3.set_xlabel("Input channel index")
        ax3.set_ylabel("Anomaly score")
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_dir = os.path.join(folder, "pics")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"viz_layer{i}.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"[Saved] {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", type=str, required=True, help="Path to folder containing .pt files"
    )
    parser.add_argument("--idx", type=int, default=-1, help="Layer index, -1 for all")
    args = parser.parse_args()

    visualize_anomalies(args.folder, args.idx)
