import os
import pandas as pd

def _mean_head_tail(df: pd.DataFrame, col: str, n: int):
    if col not in df.columns or len(df) == 0:
        return None, None
    return df[col].head(n).mean(), df[col].tail(n).mean()

def _mean_episode_length_head_tail(timestep_df: pd.DataFrame, n: int):
    if "episode" not in timestep_df.columns or len(timestep_df) == 0:
        return None, None
    episode_lengths = timestep_df.groupby("episode", sort=False).size()
    if len(episode_lengths) == 0:
        return None, None
    return episode_lengths.head(n).mean(), episode_lengths.tail(n).mean()


def summarise(directory, instance=(0, 1), episode_n: int = 40, timestep_n: int = 2000):
    rows = []
    for i in range(instance[0], instance[1]):
        episodeStats = pd.read_csv(os.path.join(directory, f"episode_data_{i}.csv"))
        mean_reward_start, mean_reward_end = _mean_head_tail(episodeStats, 'total_reward_per_timestep', episode_n)
        reward_end_stability = episodeStats["total_reward_per_timestep"].tail(episode_n).std()

        timestepStats = pd.read_csv(os.path.join(directory, f"timestep_data_{i}.csv"))
        mean_episode_length_start, mean_episode_length_end = _mean_episode_length_head_tail(timestepStats, episode_n)

        mean_abs_td_error_start, mean_abs_td_error_end = _mean_head_tail(timestepStats, "abs(td error)", timestep_n)
        mean_alpha_beta_diff_start, mean_alpha_beta_diff_end = _mean_head_tail(timestepStats, "mean(abs(alpha-beta))", episode_n)

        policy_entropy_start, policy_entropy_end = _mean_head_tail(timestepStats, "policy_entropy", timestep_n)
        action_mean_start, action_mean_end = _mean_head_tail(timestepStats, "action_mean", timestep_n)
        action_std_start, action_std_end = _mean_head_tail(timestepStats, "action_std", timestep_n)
        action_l2_start, action_l2_end = _mean_head_tail(timestepStats, "action_l2_norm", timestep_n)

        policy_grad_before_start, policy_grad_before_end = _mean_head_tail(
            timestepStats, "policy_grad_norm_before_clip", timestep_n
        )
        policy_grad_after_start, policy_grad_after_end = _mean_head_tail(
            timestepStats, "policy_grad_norm_after_clip", timestep_n
        )
        value_grad_before_start, value_grad_before_end = _mean_head_tail(
            timestepStats, "value_grad_norm_before_clip", timestep_n
        )
        value_grad_after_start, value_grad_after_end = _mean_head_tail(
            timestepStats, "value_grad_norm_after_clip", timestep_n
        )
        policy_update_before_start, policy_update_before_end = _mean_head_tail(
            timestepStats, "policy_update_norm_before_clip", timestep_n
        )
        policy_update_after_start, policy_update_after_end = _mean_head_tail(
            timestepStats, "policy_update_norm_after_clip", timestep_n
        )
        value_update_before_start, value_update_before_end = _mean_head_tail(
            timestepStats, "value_update_norm_before_clip", timestep_n
        )
        value_update_after_start, value_update_after_end = _mean_head_tail(
            timestepStats, "value_update_norm_after_clip", timestep_n
        )

        policy_trace_before_start, policy_trace_before_end = _mean_head_tail(
            timestepStats, "policy_trace_norm_before_clip", timestep_n
        )
        policy_trace_after_start, policy_trace_after_end = _mean_head_tail(
            timestepStats, "policy_trace_norm_after_clip", timestep_n
        )
        value_trace_before_start, value_trace_before_end = _mean_head_tail(
            timestepStats, "value_trace_norm_before_clip", timestep_n
        )
        value_trace_after_start, value_trace_after_end = _mean_head_tail(
            timestepStats, "value_trace_norm_after_clip", timestep_n
        )

        row = {
            "instance": i,
            f"mean_reward_start_{episode_n}": mean_reward_start,
            f"mean_reward_end_{episode_n}": mean_reward_end,
            "reward_delta": mean_reward_end - mean_reward_start,
            f"reward_end_stability_{episode_n}_std": reward_end_stability,
            f"mean_episode_length_start_{episode_n}": mean_episode_length_start,
            f"mean_episode_length_end_{episode_n}": mean_episode_length_end,
            f"mean_abs_td_error_start_{timestep_n}": mean_abs_td_error_start,
            f"mean_abs_td_error_end_{timestep_n}": mean_abs_td_error_end,
            f"mean_abs_alpha_beta_diff_start_{timestep_n}": mean_alpha_beta_diff_start,
            f"mean_abs_alpha_beta_diff_end_{timestep_n}": mean_alpha_beta_diff_end,
            f"mean_policy_entropy_start_{timestep_n}": policy_entropy_start,
            f"mean_policy_entropy_end_{timestep_n}": policy_entropy_end,
            f"mean_action_mean_start_{timestep_n}": action_mean_start,
            f"mean_action_mean_end_{timestep_n}": action_mean_end,
            f"mean_action_std_start_{timestep_n}": action_std_start,
            f"mean_action_std_end_{timestep_n}": action_std_end,
            f"mean_action_l2_norm_start_{timestep_n}": action_l2_start,
            f"mean_action_l2_norm_end_{timestep_n}": action_l2_end,
            f"mean_policy_grad_norm_before_clip_start_{timestep_n}": policy_grad_before_start,
            f"mean_policy_grad_norm_before_clip_end_{timestep_n}": policy_grad_before_end,
            f"mean_policy_grad_norm_after_clip_start_{timestep_n}": policy_grad_after_start,
            f"mean_policy_grad_norm_after_clip_end_{timestep_n}": policy_grad_after_end,
            f"mean_value_grad_norm_before_clip_start_{timestep_n}": value_grad_before_start,
            f"mean_value_grad_norm_before_clip_end_{timestep_n}": value_grad_before_end,
            f"mean_value_grad_norm_after_clip_start_{timestep_n}": value_grad_after_start,
            f"mean_value_grad_norm_after_clip_end_{timestep_n}": value_grad_after_end,
            f"mean_policy_update_norm_before_clip_start_{timestep_n}": policy_update_before_start,
            f"mean_policy_update_norm_before_clip_end_{timestep_n}": policy_update_before_end,
            f"mean_policy_update_norm_after_clip_start_{timestep_n}": policy_update_after_start,
            f"mean_policy_update_norm_after_clip_end_{timestep_n}": policy_update_after_end,
            f"mean_value_update_norm_before_clip_start_{timestep_n}": value_update_before_start,
            f"mean_value_update_norm_before_clip_end_{timestep_n}": value_update_before_end,
            f"mean_value_update_norm_after_clip_start_{timestep_n}": value_update_after_start,
            f"mean_value_update_norm_after_clip_end_{timestep_n}": value_update_after_end,
            f"mean_policy_trace_norm_before_clip_start_{timestep_n}": policy_trace_before_start,
            f"mean_policy_trace_norm_before_clip_end_{timestep_n}": policy_trace_before_end,
            f"mean_policy_trace_norm_after_clip_start_{timestep_n}": policy_trace_after_start,
            f"mean_policy_trace_norm_after_clip_end_{timestep_n}": policy_trace_after_end,
            f"mean_value_trace_norm_before_clip_start_{timestep_n}": value_trace_before_start,
            f"mean_value_trace_norm_before_clip_end_{timestep_n}": value_trace_before_end,
            f"mean_value_trace_norm_after_clip_start_{timestep_n}": value_trace_after_start,
            f"mean_value_trace_norm_after_clip_end_{timestep_n}": value_trace_after_end,
        }

        # include hyperparams from first row if present
        for col in episodeStats.columns:
            if col.startswith("H_"):
                row[col] = episodeStats[col].iloc[0]

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    return summary_df


if __name__ == "__main__":
    directory = "../train_information/go2/instance_26_03_24_18_16_52"
    summary = summarise(os.path.join(directory, "raw_data"), (0, 32), timestep_n=20000, episode_n=40)
    summary.to_csv(os.path.join(directory, "summary_statistics.csv"))
