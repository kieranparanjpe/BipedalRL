import os
import pandas as pd

def summarise(directory, instance=(0, 1)):
    rows = []
    for i in range(instance[0], instance[1]):
        episodeStats = pd.read_csv(os.path.join(directory, f"episode_data_{i}.csv"))

        mean_reward_start = episodeStats["total_reward_per_timestep"].head(40).mean()
        mean_reward_end = episodeStats["total_reward_per_timestep"].tail(40).mean()
        reward_end_stability = episodeStats["total_reward_per_timestep"].tail(40).std()

        timestepStats = pd.read_csv(os.path.join(directory, f"timestep_data_{i}.csv"))

        mean_abs_td_error_end = timestepStats["abs(td error)"].tail(20000).mean()
        mean_alpha_beta_diff_end = timestepStats["mean(abs(alpha-beta))"].tail(20000).mean()

        row = {
            "instance": i,
            "mean_reward_start_40": mean_reward_start,
            "mean_reward_end_40": mean_reward_end,
            "reward_delta": mean_reward_end - mean_reward_start,
            "reward_end_stability_40_std": reward_end_stability,
            "mean_abs_td_error_end_20000": mean_abs_td_error_end,
            "mean_abs_alpha_beta_diff_end_20000": mean_alpha_beta_diff_end,
        }

        # include hyperparams from first row if present
        for col in episodeStats.columns:
            if col.startswith("H_"):
                row[col] = episodeStats[col].iloc[0]

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    return summary_df


if __name__ == "__main__":
    directory = "../train_information/go2/instance_26_03_23_05_25_41"
    summary = summarise(os.path.join(directory, "raw_data"), (0, 36))
    summary.to_csv(os.path.join(directory, "summary_statistics.csv"))