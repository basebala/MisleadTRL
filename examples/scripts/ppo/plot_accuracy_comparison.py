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

    # Determine axis limits dynamically to better fit the provided ranges
    reward_min = min(first_reward, last_reward)
    reward_max = max(first_reward, last_reward)
    reward_margin = max(0.05, 0.05 * (reward_max - reward_min))

    accuracy_min = min(left_accuracy, right_accuracy)
    accuracy_max = max(left_accuracy, right_accuracy)
    accuracy_margin = max(2.0, 0.05 * (accuracy_max - accuracy_min))
    accuracy_base = accuracy_min - accuracy_margin

    # Create figure and axes
    fig, ax2 = plt.subplots(figsize=(6, 6))
    ax1 = ax2.twinx()

    # Set width of bars
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

    # Baseline for reward bars so the small range is visible
    reward_base = reward_min - reward_margin

    # Create bars - reversed from previous version (reward on left, accuracy on right)
    # Reward bars on the left (ax1)
    ax1.bar(
        r1[0],
        first_reward - reward_base,
        width=bar_width,
        color=blue_color,
        label="$π_{init}$",
        bottom=reward_base,
    )
    ax1.bar(
        r2[0],
        last_reward - reward_base,
        width=bar_width,
        color=orange_color,
        label="$π_{rlhf}$",
        bottom=reward_base,
    )

    # Accuracy bars on the right (ax2)
    ax2.bar(r1[1], left_accuracy - accuracy_base, width=bar_width, color=blue_color, bottom=accuracy_base)
    ax2.bar(r2[1], right_accuracy - accuracy_base, width=bar_width, color=orange_color, bottom=accuracy_base)

    # Add labels and title
    ax1.set_xlabel("Metrics")
    ax1.set_ylabel("Reward (for $R^{train}$)")
    ax2.set_ylabel("Accuracy (for $R^*$)")

    # Set x-ticks with proper font size
    ax2.set_xticks([bar_width / 2, offset + 1.5 * bar_width])
    ax2.set_xticklabels(["$R^{train}$", "$R^*$"], fontsize=24)

    # Set axis limits
    ax1.set_ylim(reward_min - reward_margin, reward_max + reward_margin)
    ax2.set_ylim(accuracy_min - accuracy_margin, accuracy_max + accuracy_margin)

    # Add legend
    ax1.legend(loc="upper right")

    # Adjust layout
    fig.tight_layout()

    # Save plot
    plt.savefig("accuracy_comparison.png", dpi=300)


if __name__ == "__main__":
    # Same run as plotter.py for rewards
    REWARD_RUN_NAME = "chandna-uc-berkeley-electrical-engineering-computer-sciences/huggingface/xk976apx"
    
    # Two runs for accuracies
    LEFT_ACCURACY_RUN_NAME = (
        "chandna-uc-berkeley-electrical-engineering-computer-sciences/huggingface/ygyekzne"
    )
    RIGHT_ACCURACY_RUN_NAME = (
        "chandna-uc-berkeley-electrical-engineering-computer-sciences/huggingface/9rl3khng"
    )

    # Get rewards from the reward run (first and last)
    first_reward, last_reward = get_rewards_from_run(run_name=REWARD_RUN_NAME)
    
    # Get accuracies from the two accuracy runs
    left_accuracy = get_accuracy_from_run(run_name=LEFT_ACCURACY_RUN_NAME)
    right_accuracy = get_accuracy_from_run(run_name=RIGHT_ACCURACY_RUN_NAME)

    plot_rewards_and_accuracies(first_reward, last_reward, left_accuracy, right_accuracy)
