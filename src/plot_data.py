import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_single_instance(directory, instance):
    episodeStats = pd.read_csv(os.path.join(directory, f"episode_data_{instance}.csv"))
    timestepStats = pd.read_csv(os.path.join(directory, f"timestep_data_{instance}.csv"))

    if len(episodeStats) > 0:
        for col in [c for c in episodeStats.columns if c != "episode" and 'H_' not in c and 'Unnamed' not in c]:
            episodeStats.plot(x="episode", y=[col])
            plt.title(f'{col} vs episode')
            plt.show()

    if len(timestepStats) > 0:
        for col in [c for c in timestepStats.columns if c != "timestep" and 'H_' not in c and 'Unnamed' not in c]:
            timestepStats.plot(x="timestep", y=[col])
            plt.title(f'{col} vs timestep')
            plt.show()

def plot_data(directory, instance=(0, 1)):
    for i in range(instance[0], instance[1]):
        plot_single_instance(directory, i)

if __name__ == "__main__":
    plot_data("../train_information/go2/instance_26_03_23_04_31_16/raw_data", (0, 1))