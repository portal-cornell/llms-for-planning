import argparse
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import re

def plot_bar_graph_reached_goal(df, title, save_path):
    """Plots and saves a bar graph of the number of times the goal was reached.
    
    Parameters:
        df (pd.DataFrame)
            The DataFrame containing the results.
        title (str)
            The title of the graph.
        save_path (str)
            The path to save the graph.
    
    Side Effects:
        - Saves the graph to the save path.
    """
    grouped = df.groupby("optimal_length")["reached_goal"].sum()
    grouped.plot(kind="bar", rot=0)
    plt.xlabel("Optimal Plan Length")
    plt.ylabel("Number of Times Goal Reached")
    plt.title(title)
    plt.savefig(os.path.join(save_path, "reached_goal_results.png"))

def get_stats(log_dir, results_csv, log_file, hydra_log_file):
    """Returns the statistics for the log files.
    
    Parameters:
        log_dir (str)
            The directory containing the log files.
        results_csv (str)
            The name of the results CSV.
        log_file (str)
            The name of the log file.
        hydra_log_file (str)
            The name of the Hydra log file.
    
    Returns:
        df (pd.DataFrame)
            The DataFrame containing the results.
        llm_calls (int)
            The number of LLM calls.
        llm_token_usage (int)
            The number of LLM tokens used.
    """
    df = pd.DataFrame()
    llm_calls = 0
    llm_token_usage = 0
    invalid_actions = {}
    for dirpath, dirnames, filenames in os.walk(log_dir):
        for filename in filenames:
            if filename == results_csv:
                csv_file = os.path.join(dirpath, filename)
                with open(csv_file, "r") as f:
                    new_df = pd.read_csv(f)
                    df = pd.concat([df, new_df], ignore_index=True)
            elif filename == log_file:
                curr_log_file = os.path.join(dirpath, filename)
                with open(curr_log_file, "r") as f:
                    log = f.readlines()
                llm_calls += sum(["RESPONSE" in line for line in log]) # Sum True values
            elif filename == hydra_log_file:
                curr_hydra_log_file = os.path.join(dirpath, filename)
                with open(curr_hydra_log_file, "r") as f:
                    hydra_log = f.readlines()
                regex = re.compile(r"Prompt Tokens: (\d+)")
                for line in hydra_log:
                    match = regex.search(line)
                    if match:
                        llm_token_usage += int(match.group(1))
    return df, llm_calls, llm_token_usage

def get_invalid_actions(log_dir, log_file):
    """Returns the invalid actions for the log files.
    
    Parameters:
        log_dir (str)
            The directory containing the log files.
        log_file (str)
            The name of the log file.
    
    Returns:
        invalid_actions (dict)
            The dictionary containing the invalid actions.
    """
    instance_to_invalid_count = {}
    for dirpath, dirnames, filenames in os.walk(log_dir):
        for filename in filenames:
            if filename == log_file:
                curr_log_file = os.path.join(dirpath, filename)
                with open(curr_log_file, "r") as f:
                    log = f.readlines()
                invalid_regex = re.compile(r"Error Feedback: The action '(.+)' at index")
                basename = os.path.basename(dirpath)
                invalid_actions = []
                for line in log:
                    match = invalid_regex.search(line)
                    if match:
                        invalid_action = match.group(1)
                        if invalid_action not in invalid_actions:
                            instance_to_invalid_count[basename] = instance_to_invalid_count.get(basename, 0) + 1
    return instance_to_invalid_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="The directory containing the log files.")
    parser.add_argument("--results_csv", type=str, required=True, help="The name of the results CSV.")
    parser.add_argument("--log_file", type=str, required=True, help="The name of the log file.")
    parser.add_argument("--hydra_log_file", type=str, help="The name of the Hydra log file.")
    viz_types = ["bar_graph_reached_goal", "print_stats"]
    parser.add_argument("--viz_types", type=str, nargs="+", required=True, choices=viz_types, help="The types of visualizations to create.")
    parser.add_argument("--title", type=str, help="[bar_graph_reached_goal] The title of the graph.")
    args = parser.parse_args()

    df, llm_calls, llm_token_usage = get_stats(args.log_dir, args.results_csv, args.log_file, args.hydra_log_file)

    if "bar_graph_reached_goal" in args.viz_types:
        plot_bar_graph_reached_goal(df, args.title, args.log_dir)
    if "print_stats" in args.viz_types:
        print("Printing stats...")
        success_df = df[df["reached_goal"] == True]
        success_df = df[(df["reached_goal"] == True) & (df["total_edges_expanded"] <= 20)]
        print(f"Success rate: {len(success_df)} / {len(df)}")
        print(f"Average nodes expanded: {df['total_nodes_expanded'].mean()}")
        print(f"Median nodes expanded: {df['total_nodes_expanded'].median()}")
        print(f"Std nodes expanded: {df['total_nodes_expanded'].std()}")
        print(f"Average edges expanded: {df['total_edges_expanded'].mean()}")
        print(f"Median edges expanded: {df['total_edges_expanded'].median()}")
        print(f"Std edges expanded: {df['total_edges_expanded'].std()}")
        print(f"LLM calls: {llm_calls}")
        print(f"LLM token usage: {llm_token_usage}")

