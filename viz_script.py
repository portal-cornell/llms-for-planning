import argparse
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="The directory containing the log files.")
    parser.add_argument("--results_csv", type=str, required=True, help="The name of the results CSV.")
    viz_types = ["bar_graph_reached_goal", "print_stats"]
    parser.add_argument("--viz_types", type=str, nargs="+", required=True, choices=viz_types, help="The types of visualizations to create.")
    parser.add_argument("--title", type=str, help="[bar_graph_reached_goal] The title of the graph.")
    args = parser.parse_args()

    df = pd.DataFrame()
    for dirpath, dirnames, filenames in os.walk(args.log_dir):
        for filename in filenames:
            if filename == args.results_csv:
                csv_file = os.path.join(dirpath, filename)
                with open(csv_file, "r") as f:
                    new_df = pd.read_csv(f)
                    df = pd.concat([df, new_df], ignore_index=True)
    # import pdb; pdb.set_trace()
    if "bar_graph_reached_goal" in args.viz_types:
        plot_bar_graph_reached_goal(df, args.title, args.log_dir)
    if "print_stats" in args.viz_types:
        print("Printing stats...")
        success_df = df[df["reached_goal"] == True]
        print(f"Success rate: {len(success_df)} / {len(df)}")
        print(f"Average nodes expanded: {success_df['total_nodes_expanded'].mean()}")
        print(f"Median nodes expanded: {success_df['total_nodes_expanded'].median()}")
        print(f"Average edges expanded: {success_df['total_edges_expanded'].mean()}")
        print(f"Median edges expanded: {success_df['total_edges_expanded'].median()}")

