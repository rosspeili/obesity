import matplotlib.pyplot as plt
from mlxtend.plotting import plot_sequential_feature_selection
import os

def plot_sfs_results(metric_dict: dict, filename: str, title: str):
    """Plots SFS/SBS results and saves to results directory."""
    fig1 = plot_sequential_feature_selection(metric_dict, kind='std_dev')
    plt.title(title)
    
    os.makedirs('results', exist_ok=True)
    filepath = os.path.join('results', filename)
    plt.savefig(filepath)
    plt.clf() # Clear graph before next runs
