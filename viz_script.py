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
    parser.add_argument("--title", type=str, required=True, help="The title of the graph.")
    parser.add_argument("--viz_types", type=str, nargs="+", required=True, choices=["bar_graph_reached_goal"], help="The types of visualizations to create.")
    args = parser.parse_args()

    df = pd.DataFrame()
    for dirpath, dirnames, filenames in os.walk(args.log_dir):
        for filename in filenames:
            if filename == args.results_csv:
                csv_file = os.path.join(dirpath, filename)
                with open(csv_file, "r") as f:
                    new_df = pd.read_csv(f)
                    df = pd.concat([df, new_df], ignore_index=True)
    
    if "bar_graph_reached_goal" in args.viz_types:
        plot_bar_graph_reached_goal(df, args.title, args.log_dir)