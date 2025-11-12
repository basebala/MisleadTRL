from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import wandb


import pandas as pd
import wandb
from typing import List, Tuple

def get_rewards_and_accuracies_separate_callbacks(
    run_name: str,
    reward_metric: str = "train/objective/rlhf_reward",
    accuracy_metric: str = "train/eval/qa_accuracy",
) -> Tuple[List[float], List[float]]:
    api = wandb.Api()
    run = api.run(run_name)

    # Grab the entire history
    df = run.history(pandas=True)

    # Get reward-only and accuracy-only subsets
    rewards_df = df[["_step", reward_metric]].dropna(subset=[reward_metric])
    acc_df = df[["_step", accuracy_metric]].dropna(subset=[accuracy_metric])

    # Align: for each accuracy step, take the latest reward before or at that step
    aligned_rewards = []
    aligned_accuracies = []

    for _, row in acc_df.iterrows():
        step = row["_step"]
        acc = row[accuracy_metric]

        # Find the most recent reward step before or equal to this
        past_rewards = rewards_df[rewards_df["_step"] <= step]
        if len(past_rewards) == 0:
            continue
        last_reward = past_rewards.iloc[-1][reward_metric]

        aligned_rewards.append(last_reward)
        aligned_accuracies.append(acc)

    print(f"Aligned {len(aligned_rewards)} points between reward & accuracy callbacks.")
    return aligned_rewards, aligned_accuracies


def plot_rewards_and_accuracies(rewards: List[float], accuracies: List[float]) -> None:
    """
    Create and save a bar chart comparing initial and maximum rewards and accuracies.

    The plot shows two groups of bars:
    1. Reward values (R^train) - comparing initial policy vs RLHF policy
    2. Accuracy values (R*) - comparing initial policy vs RLHF policy

    Args:
        rewards (List[float]): List of reward values from training
        accuracies (List[float]): List of accuracy values from training

    Returns:
        None: The function saves the plot to "rewards_and_accuracies.png"
    """
    # Increase the global font size
    plt.rcParams.update({"font.size": 16})

    # If the accuracies are provided in the interval [0, 1], convert them to the interval [0, 100]
    if max(accuracies) <= 1:
        accuracies = [acc * 100 for acc in accuracies]

    # Find the position of the maximum reward
    max_reward_idx = np.argmax(rewards)

    # Extract the first and max reward values
    first_reward = rewards[0]
    max_reward = rewards[max_reward_idx]

    # Extract the first accuracy and accuracy at max reward
    first_accuracy = accuracies[0]
    max_accuracy = accuracies[max_reward_idx]

    # Determine axis limits dynamically to better fit the provided ranges
    reward_min = min(first_reward, max_reward)
    reward_max = max(first_reward, max_reward)
    reward_margin = max(0.05, 0.05 * (reward_max - reward_min))

    accuracy_min = min(first_accuracy, max_accuracy)
    accuracy_max = max(first_accuracy, max_accuracy)
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
        max_reward - reward_base,
        width=bar_width,
        color=orange_color,
        label="$π_{rlhf}$",
        bottom=reward_base,
    )

    # Accuracy bars on the right (ax2)
    ax2.bar(r1[1], first_accuracy - accuracy_base, width=bar_width, color=blue_color, bottom=accuracy_base)
    ax2.bar(r2[1], max_accuracy - accuracy_base, width=bar_width, color=orange_color, bottom=accuracy_base)

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
    plt.savefig("rewards_and_accuracies.png", dpi=300)


if __name__ == "__main__":
    RUN_NAME = "chandna-uc-berkeley-electrical-engineering-computer-sciences/huggingface/xk976apx"
    rewards, accuracies = get_rewards_and_accuracies_separate_callbacks(run_name=RUN_NAME)
    plot_rewards_and_accuracies(rewards, accuracies)