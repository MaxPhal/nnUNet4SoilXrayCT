import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
import numpy as np 

class TrainingLogProcessor:
    def __init__(self, input_folder, verbose=False):
        self.input_folder = input_folder
        self.verbose = verbose
        self.patterns = {
            'epoch': re.compile(r'Epoch (\d+)'),
            'train_loss': re.compile(r'train_loss (-?[\d.]+)'),
            'val_loss': re.compile(r'val_loss (-?[\d.]+)'),
            'pseudo_dice': re.compile(r'New best EMA pseudo Dice: ([\d.]+)'),
            'epoch_time': re.compile(r'Epoch time: ([\d.]+) s')
        }

    def get_file_list(self):
        return [f for f in os.listdir(self.input_folder) if os.path.isfile(os.path.join(self.input_folder, f))]

    def parse_log_file(self, txt_file):
        data = []
        with open(os.path.join(self.input_folder, txt_file), 'r') as file:
            epoch_data = {}
            for line in file:
                for key, pattern in self.patterns.items():
                    match = pattern.search(line)
                    if match:
                        if key == 'epoch':
                            if epoch_data:
                                data.append(epoch_data)
                            epoch_data = {'epoch': int(match.group(1))}
                        else:
                            epoch_data[key] = float(match.group(1))
            if epoch_data:
                data.append(epoch_data)
        
        df = pd.DataFrame(data)
        df['fold'] = os.path.splitext(txt_file)[0]  # Extract file name without extension
        return df

    def process_logs(self):
        df_list = [self.parse_log_file(txt_file) for txt_file in self.get_file_list()]
        df_combined = pd.concat(df_list, ignore_index=True)
        return df_combined

    def summarize_results(self, df_combined):
        summary = df_combined.groupby("epoch")[["train_loss", "val_loss"]].agg(["mean", "std"])
        return summary

    def save_summary(self, summary, output_folder):
        summary.to_csv(os.path.join(output_folder, 'summary.csv'), index=True)

    def plot_results(self, summary, output_folder):
        epochs = summary.index
        mean_train_loss = summary[("train_loss", "mean")]
        std_train_loss = summary[("train_loss", "std")]
        mean_val_loss = summary[("val_loss", "mean")]
        std_val_loss = summary[("val_loss", "std")]
        
        plt.figure(figsize=(6.5, 5))
        sns.set(style="ticks")
        sns.set_context("paper", font_scale=1.75)
        plt.rcParams["font.family"] = "Arial"
        
        sns.lineplot(x=epochs, y=mean_train_loss, label="Train Loss", color=sns.color_palette(palette='Set2')[1])
        plt.fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, color=sns.color_palette(palette='Set2')[1], alpha=0.2)
        
        sns.lineplot(x=epochs, y=mean_val_loss, label="Validation Loss", color=sns.color_palette(palette='Set2')[2])
        plt.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, color=sns.color_palette(palette='Set2')[2], alpha=0.2)
        
        plt.yticks(np.arange(-1, 1.25, 0.5))      
        plt.ylim(-1, 1)
        plt.xlabel("Number of Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(output_folder, 'training_plots.svg'), format="svg", bbox_inches='tight', transparent = True)
        plt.show()

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Script to process training logs and generate plots.')
    parser.add_argument('-i', type=Path, required=True, help='Path to the input directory where training logs are stored')
    parser.add_argument('-o', type=Path, required=True, help='Path to the output directory where output files will be saved -- Should be different from input directory')
    parser.add_argument('-v', action='store_true', help='Increase output verbosity')
    args = parser.parse_args()

    # Process training logs
    processor = TrainingLogProcessor(args.i, args.v)
    df_combined = processor.process_logs()
    summary = processor.summarize_results(df_combined)

    # Save summary and plot results
    processor.save_summary(summary, args.o)
    processor.plot_results(summary, args.o)
    print(f"Training plot saved to {args.o}")

if __name__ == "__main__":
    main()