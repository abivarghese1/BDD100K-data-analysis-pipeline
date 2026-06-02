# main.py
from data_analysis.bdd_parser import load_dataset
from data_analysis.bdd_analysis import compute_stats, find_anomalies
from data_analysis.dashboard import plot_dashboard
from data_analysis.visualize_samples import visualize_interesting_samples
import json

print("Loading dataset...")
data = load_dataset()

print("Computing statistics...")
train_stats = compute_stats(data["train"])
val_stats = compute_stats(data["val"])
anomalies = find_anomalies(data["train"], train_stats)

print("Generating dashboard figures...")
plot_dashboard(train_stats, val_stats)

print("Generating sample visualizations...")
visualize_interesting_samples(data, anomalies)

# Save stats as JSON for later use in the model pipeline
with open("train_stats.json", "w") as f:
    json.dump(
        {
            k: dict(v) if hasattr(v, "items") else v
            for k, v in train_stats.items()
            if k != "bbox_areas"
        },
        f,
        indent=2,
        default=str,
    )

print("Done. Outputs: fig1–4_*.png, samples_*.png, train_stats.json")
