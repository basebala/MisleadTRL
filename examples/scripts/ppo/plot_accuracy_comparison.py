from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

def get_rewards_from_run(
    run_name: str,
    reward_metric: str = "train/objective/rlhf_reward",
) -> Tuple[float, float]:
    """
    Retrieve the first and last reward values from a Weights & Biases run.
    
    Args:
        run_name (str): The name of the W&B run to retrieve data from
        reward_metric (str, optional): The name of the reward metric in W&B.
            Defaults to "train/objective/rlhf_reward".
    
    Returns:
        Tuple[float, float]: A tuple containing (first_reward, last_reward)
    """
    api = wandb.Api()
    run = api.run(run_name)

    # Grab the entire history
    df = run.history(pandas=True)

    # Get reward-only subset
    rewards_df = df[["_step", reward_metric]].dropna(subset=[reward_metric])
    
    if len(rewards_df) == 0:
        raise ValueError(f"No reward data found for metric '{reward_metric}' in run {run_name}")
    
    # Get first and last reward values
    first_reward = rewards_df.iloc[0][reward_metric]
    last_reward = rewards_df.iloc[-1][reward_metric]
    
    print(f"Retrieved first reward {first_reward} and last reward {last_reward} from run {run_name}.")
    return first_reward, last_reward


def get_accuracy_from_run(
    run_name: str,
    accuracy_metric: str = "qa/accuracy_overall",
) -> float:
    """
    Retrieve the single accuracy value from a Weights & Biases run.

    Args:
        run_name (str): The name of the W&B run to retrieve data from
        accuracy_metric (str, optional): The name of the accuracy metric in W&B.
            Defaults to "qa/accuracy_overall".

    Returns:
        float: The accuracy value from the run
    """
    api = wandb.Api()
    run = api.run(run_name)

    # Grab the entire history
    df = run.history(pandas=True)

    # Get accuracy-only subset
    acc_df = df[["_step", accuracy_metric]].dropna(subset=[accuracy_metric])

    # Extract the single accuracy value
    if len(acc_df) == 0:
        raise ValueError(f"No accuracy data found for metric '{accuracy_metric}' in run {run_name}")
    
    accuracy = acc_df[accuracy_metric].iloc[0]

    print(f"Retrieved accuracy value {accuracy} from run {run_name}.")
    return accuracy


def plot_rewards_and_accuracies(
    first_reward: float,
    last_reward: float,
    left_accuracy: float,
    right_accuracy: float,
) -> None:
    """
    Create and save a bar chart comparing initial and maximum rewards and accuracies.

    The plot shows two groups of bars:
    1. Reward values (R^train) - comparing initial policy vs RLHF policy
    2. Accuracy values (R*) - comparing initial policy vs RLHF policy

    Args:
        first_reward (float): First reward value (for π_init)
        last_reward (float): Last reward value (for π_rlhf)
        left_accuracy (float): Accuracy value from left run (for π_init)
        right_accuracy (float): Accuracy value from right run (for π_rlhf)

    Returns:
        None: The function saves the plot to "accuracy_comparison.png"
    """
    # Increase the global font size
    plt.rcParams.update({"font.size": 16})

    # If the accuracies are provided in the interval [0, 1], convert them to the interval [0, 100]
    if left_accuracy <= 1:
        left_accuracy = left_accuracy * 100
    if right_accuracy <= 1:
        right_accuracy = right_accuracy * 100

    # Calculate reward scale so that first_reward appears at 79% and last_reward at 89%
    # We want: (first_reward - reward_min) / (reward_max - reward_min) = 0.79
    # And: (last_reward - reward_min) / (reward_max - reward_min) = 0.89
    # Solving: reward_min = (0.89 * first_reward - 0.79 * last_reward) / (0.89 - 0.79)
    #          reward_max = reward_min + (last_reward - first_reward) / (0.89 - 0.79)
    reward_min = (0.89 * first_reward - 0.79 * last_reward) / 0.10
    reward_max = reward_min + (last_reward - first_reward) / 0.10
    reward_base = reward_min

    # Fixed accuracy scale: 40 to 75
    accuracy_min = 40
    accuracy_max = 75
    accuracy_base = 40

    # Create figure and axes - match plot_reproduction_comparison.py exactly
    fig, ax2 = plt.subplots(figsize=(6, 6))
    ax1 = ax2.twinx()

    # Set width of bars - match plot_reproduction_comparison.py
    bar_width = 0.3
    offset = 0.5

    # Set positions for bars - reversed from previous version
    r1 = np.array([0, offset + bar_width])  # Positions for first bars in each group
    r2 = np.array(
        [bar_width, offset + 2 * bar_width]
    )  # Positions for second bars in each group

    # Define colors using RGB values
    blue_color = (113 / 255, 193 / 255, 209 / 255)  # (113,193,209)
    orange_color = (241 / 255, 180 / 255, 90 / 255)  # (241,180,90)

    # Create bars - reversed from previous version (reward on left, accuracy on right)
    # Reward bars on the left (ax1) - using actual values, scale adjusted so they appear at 79% and 89%
    ax1.bar(
        r1[0],
        first_reward - reward_base,
        width=bar_width,
        color=blue_color,
        label="$π_{init}$",
        bottom=reward_base,
        zorder=1,
    )
    ax1.bar(
        r2[0],
        last_reward - reward_base,
        width=bar_width,
        color=orange_color,
        label="$π_{rlhf}$",
        bottom=reward_base,
        zorder=1,
    )

    # Accuracy bars on the right (ax2)
    ax2.bar(r1[1], left_accuracy - accuracy_base, width=bar_width, color=blue_color, bottom=accuracy_base, zorder=1)
    ax2.bar(r2[1], right_accuracy - accuracy_base, width=bar_width, color=orange_color, bottom=accuracy_base, zorder=1)

    # Add labels and title
    ax1.set_xlabel("Metrics")
    ax1.set_ylabel("Reward (for $R^{train}$)")
    ax2.set_ylabel("Accuracy (for $R^*$)")

    # Set x-ticks with proper font size
    ax2.set_xticks([bar_width / 2, offset + 1.5 * bar_width])
    ax2.set_xticklabels(["$R^{train}$", "$R^*$"], fontsize=24)

    # Set axis limits - fixed scales (match plot_reproduction_comparison.py)
    ax1.set_ylim(reward_min, reward_max)
    ax2.set_ylim(accuracy_min, accuracy_max)
    
    # Don't set x-limits explicitly - let matplotlib handle it like plot_reproduction_comparison.py
    # This ensures natural spacing that matches
    
    # Get x-limits after matplotlib sets them naturally, then extend slightly for the line
    # Force a draw to get the actual limits
    fig.canvas.draw()
    x_min, x_max = ax2.get_xlim()
    # Extend slightly for the line to span both sections
    x_max_extended = x_max + 0.05
    
    # Add red dotted line at 50% accuracy labeled "random" - spans full width, drawn on top after everything
    # Draw on ax2 (accuracy axis) with high zorder
    line = ax2.plot([x_min, x_max], [50, 50], color='red', linestyle='--', linewidth=1.5, label='random', zorder=10)
    # Bring ax2 to front so line appears above ax1 bars
    ax2.set_zorder(ax1.get_zorder() + 1)
    ax2.patch.set_visible(False)  # Make ax2 background transparent so ax1 shows through
    

    # Add legends at the top - make them smaller
    ax1.legend(loc="upper left", bbox_to_anchor=(0, 1), fontsize=12)
    # Make the random line in legend smaller by using shorter handlelength and smaller font
    ax2.legend(loc="upper right", bbox_to_anchor=(1, 1), handlelength=1.5, fontsize=12)

    # Adjust layout
    fig.tight_layout()

    # Save plot
    plt.savefig("ablation_comparison.png", dpi=300)


if __name__ == "__main__":
    # Same run as plotter.py for rewards
    REWARD_RUN_NAME = "chandna-uc-berkeley-electrical-engineering-computer-sciences/huggingface/e40psuhy"
    
    # Two runs for accuracies
    LEFT_ACCURACY_RUN_NAME = (
        "chandna-uc-berkeley-electrical-engineering-computer-sciences/huggingface/7odvpk71"
    )
    RIGHT_ACCURACY_RUN_NAME = (
        "chandna-uc-berkeley-electrical-engineering-computer-sciences/huggingface/4uhvu93d"
    )

    # Get rewards from the reward run (first and last)
    first_reward, last_reward = get_rewards_from_run(run_name=REWARD_RUN_NAME)
    
    # Get accuracies from the two accuracy runs
    left_accuracy = get_accuracy_from_run(run_name=LEFT_ACCURACY_RUN_NAME)
    right_accuracy = get_accuracy_from_run(run_name=RIGHT_ACCURACY_RUN_NAME)

    plot_rewards_and_accuracies(first_reward, last_reward, left_accuracy, right_accuracy)
