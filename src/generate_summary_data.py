import os
import pandas as pd
from tqdm import tqdm

METRIC_COLUMN_MAP = {
    "abs_td_error": "abs(td error)",
    "abs_alpha_beta_diff": "mean(abs(alpha-beta))",
    "policy_entropy": "policy_entropy",
    "action_mean": "action_mean",
    "action_std": "action_std",
    "action_l2_norm": "action_l2_norm",
    "policy_grad_norm_before_clip": "policy_grad_norm_before_clip",
    "policy_grad_norm_after_clip": "policy_grad_norm_after_clip",
    "value_grad_norm_before_clip": "value_grad_norm_before_clip",
    "value_grad_norm_after_clip": "value_grad_norm_after_clip",
    "policy_update_norm_before_clip": "policy_update_norm_before_clip",
    "policy_update_norm_after_clip": "policy_update_norm_after_clip",
    "value_update_norm_before_clip": "value_update_norm_before_clip",
    "value_update_norm_after_clip": "value_update_norm_after_clip",
    "policy_trace_norm_before_clip": "policy_trace_norm_before_clip",
    "policy_trace_norm_after_clip": "policy_trace_norm_after_clip",
    "value_trace_norm_before_clip": "value_trace_norm_before_clip",
    "value_trace_norm_after_clip": "value_trace_norm_after_clip",
}


def _mean_head_tail(df: pd.DataFrame, col: str, n: int):
    if col not in df.columns or len(df) == 0:
        return None, None
    return df[col].head(n).mean(), df[col].tail(n).mean()


def _reward_stats(episode_df: pd.DataFrame, episode_n: int):
    start, end = _mean_head_tail(episode_df, "total_reward_per_timestep", episode_n)
    stability = None
    if "total_reward_per_timestep" in episode_df.columns and len(episode_df) > 0:
        stability = episode_df["total_reward_per_timestep"].tail(episode_n).std()
    delta = None
    if start is not None and end is not None:
        delta = end - start
    return start, end, delta, stability


def _episode_length_stats_from_timestep(timestep_df: pd.DataFrame, episode_n: int):
    if "episode" not in timestep_df.columns or len(timestep_df) == 0:
        return None, None
    episode_lengths = timestep_df.groupby("episode", sort=False).size()
    if len(episode_lengths) == 0:
        return None, None
    return float(episode_lengths.head(episode_n).mean()), float(episode_lengths.tail(episode_n).mean())


def _episode_length_stats_from_episode(episode_df: pd.DataFrame, episode_n: int):
    return _mean_head_tail(episode_df, "episode_length", episode_n)


def _metric_summary(df: pd.DataFrame, window_n: int, suffix_n: int):
    summary = {}
    for metric_name, source_col in METRIC_COLUMN_MAP.items():
        start, end = _mean_head_tail(df, source_col, window_n)
        summary[f"mean_{metric_name}_start_{suffix_n}"] = start
        summary[f"mean_{metric_name}_end_{suffix_n}"] = end
    return summary


def _extract_hparams(episode_df: pd.DataFrame):
    out = {}
    for col in episode_df.columns:
        if col.startswith("H_"):
            out[col] = episode_df[col].iloc[0]
    return out


def _build_base_row(instance_i: int, episode_df: pd.DataFrame, episode_n: int):
    mean_reward_start, mean_reward_end, reward_delta, reward_end_stability = _reward_stats(episode_df, episode_n)
    return {
        "instance": instance_i,
        f"mean_reward_start_{episode_n}": mean_reward_start,
        f"mean_reward_end_{episode_n}": mean_reward_end,
        "reward_delta": reward_delta,
        f"reward_end_stability_{episode_n}_std": reward_end_stability,
    }


def _summarise_instance_with_timestep(directory: str, instance_i: int, episode_n: int, timestep_n: int):
    episode_df = pd.read_csv(os.path.join(directory, f"episode_data_{instance_i}.csv"))
    timestep_df = pd.read_csv(os.path.join(directory, f"timestep_data_{instance_i}.csv"))

    row = _build_base_row(instance_i, episode_df, episode_n)
    ep_len_start, ep_len_end = _episode_length_stats_from_timestep(timestep_df, episode_n)
    row[f"mean_episode_length_start_{episode_n}"] = ep_len_start
    row[f"mean_episode_length_end_{episode_n}"] = ep_len_end
    row |= _metric_summary(timestep_df, timestep_n, timestep_n)
    row |= _extract_hparams(episode_df)
    return row


def _summarise_instance_episode_only(directory: str, instance_i: int, episode_n: int):
    episode_df = pd.read_csv(os.path.join(directory, f"episode_data_{instance_i}.csv"))

    row = _build_base_row(instance_i, episode_df, episode_n)
    ep_len_start, ep_len_end = _episode_length_stats_from_episode(episode_df, episode_n)
    row[f"mean_episode_length_start_{episode_n}"] = ep_len_start
    row[f"mean_episode_length_end_{episode_n}"] = ep_len_end
    row |= _metric_summary(episode_df, episode_n, episode_n)
    row |= _extract_hparams(episode_df)
    return row


def summarise(directory, instance=(0, 1), episode_n: int = 40, timestep_n: int = 2000):
    rows = []
    for i in tqdm(range(instance[0], instance[1])):
        episode_path = os.path.join(directory, f"episode_data_{i}.csv")
        if not os.path.exists(episode_path):
            continue

        timestep_path = os.path.join(directory, f"timestep_data_{i}.csv")
        if os.path.exists(timestep_path):
            row = _summarise_instance_with_timestep(directory, i, episode_n, timestep_n)
        else:
            row = _summarise_instance_episode_only(directory, i, episode_n)

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    return summary_df


if __name__ == "__main__":
    directory = "../train_information/cube/instance_26_03_25_07_18_27"
    summary = summarise(os.path.join(directory, "raw_data"), (0, 2), timestep_n=20000, episode_n=40)
    summary.to_csv(os.path.join(directory, "summary_statistics.csv"))
