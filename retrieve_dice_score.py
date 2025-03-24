import json
import os
import csv
import argparse
from pathlib import Path

def get_labels(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        labels = data["labels"]
        return labels  
    
def aggregate_metrics(json_files, class_labels):
    total_metrics = {}
    foreground_mean = {"FP": 0, "FN": 0, "TP": 0, "TN": 0, "Dice": 0.0}
    
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            
            # Extract foreground_mean metrics (Pores)
            for key in ["FP", "FN", "TP", "TN"]:
                foreground_mean[key] += int(data.get("foreground_mean", {}).get(key, 0))
            
            # Aggregate mean metrics for all other labels
            for class_id, class_label in class_labels.items():
                if class_id == "0" or class_id == "1":  # Skip the first and second label as they correspond to "ToPredict" and "ForegroundMean"
                    continue

                # Map the "mean" dictionary values:
                mean_class_id = str(int(class_id)-1)  
                
                # Only process if the class_id exists in the "mean" data
                if mean_class_id not in data.get("mean", {}):
                    print(f"Warning: No metrics found for class {class_label} (ID {class_id}) in {file}. Skipping.")
                    continue
                
                # Get the metrics for the current class (if present)
                metrics = data.get("mean", {}).get(mean_class_id, {})
                
                if class_label not in total_metrics:
                    total_metrics[class_label] = {"FP": 0, "FN": 0, "TP": 0, "TN": 0, "Dice": 0.0}
                
                total_metrics[class_label]["FP"] += int(metrics.get("FP", 0))
                total_metrics[class_label]["FN"] += int(metrics.get("FN", 0))
                total_metrics[class_label]["TP"] += int(metrics.get("TP", 0))
                total_metrics[class_label]["TN"] += int(metrics.get("TN", 0))
    
    # Calculate Dice score for foreground_mean
    tp, fp, fn = foreground_mean["TP"], foreground_mean["FP"], foreground_mean["FN"]
    if tp + fp + fn > 0:
        foreground_mean["Dice"] = (2 * tp) / (2 * tp + fp + fn)

    # Calculate Dice score for each class
    for class_label, metrics in total_metrics.items():
        tp, fp, fn = metrics["TP"], metrics["FP"], metrics["FN"]
        if tp + fp + fn > 0:
            metrics["Dice"] = (2 * tp) / (2 * tp + fp + fn)
    
    return total_metrics, foreground_mean
 
def main():
    # Parsing arguments from command line
    parser = argparse.ArgumentParser(description='This is script to aggregate the metrics obtained after each training fold.')
    parser.add_argument('-i', type=Path, required=True, help='Path to the input directory')
    parser.add_argument('-v', action='store_true', help='Increase output verbosity')
    args = parser.parse_args()

    # Define path to JSON files
    json_files = [os.path.join(args.i, f) for f in os.listdir(args.i) if f.endswith(".json")][:5]  # Read five files

    # Read class labels from dataset_info.json
    cwd = os.getcwd()
    class_labels = get_labels(cwd + '\\dataset_info.json')

    # Aggregate metrics
    total_metrics, foreground_mean = aggregate_metrics(json_files, class_labels)

    # Write results to a CSV file
    csv_filename = 'aggregated_metrics.csv'

    # Open CSV file to write
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write headers
        writer.writerow(["Class", "FP", "FN", "TP", "TN", "Dice"])

        # Write foreground mean
        writer.writerow([class_labels["1"], foreground_mean['FP'], foreground_mean['FN'], foreground_mean['TP'], foreground_mean['TN'], f"{foreground_mean['Dice']:.4f}"])

        # Write total metrics for other classes
        for class_label, metrics in total_metrics.items():
            writer.writerow([class_label, metrics["FP"], metrics["FN"], metrics["TP"], metrics["TN"], f"{metrics['Dice']:.4f}"])

    print(f"Aggregated metrics were written to {cwd}\{csv_filename}")

if __name__ == "__main__":
    main()


