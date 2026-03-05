import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import warnings

from src.data_loader import load_and_preprocess_data
from src.models.logistic_regression import evaluate_base_model
from src.models.sfs import run_sfs
from src.models.sbs import run_sbs
from src.models.rfe import run_rfe
from src.visualization import plot_sfs_results
from src.evaluation_metrics import evaluate_and_save_model

warnings.filterwarnings('ignore') # Ignore convergence warnings for clean CLI output

app = typer.Typer(help="ftf by rosspeili - Wrapper Methods for Obesity Dataset")
console = Console()


def print_advanced_metrics(report, cm, title):
    """Utility to print advanced metrics elegantly."""
    console.print(f"[bold underline]{title} - Detailed Metrics[/bold underline]")
    console.print(f"Accuracy: {report['accuracy']:.4f}")
    console.print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    
    cm_table = Table(title="Confusion Matrix", show_header=True)
    cm_table.add_column("True \\ Pred", justify="center")
    cm_table.add_column("Non-Obese (0)", justify="center", style="green")
    cm_table.add_column("Obese (1)", justify="center", style="red")
    
    cm_table.add_row("Non-Obese (0)", str(cm[0][0]), str(cm[0][1]))
    cm_table.add_row("Obese (1)", str(cm[1][0]), str(cm[1][1]))
    console.print(cm_table)
    console.print()


@app.command()
def run_all(
    data_path: str = typer.Option(
        "ObesityDataSet_raw_and_data_sinthetic.csv", "--data-path", "-d", help="Path to the Obesity dataset CSV"
    ),
    save_models: bool = typer.Option(
        False, "--save", "-s", help="Save the best models to the 'models/' directory"
    ),
    advanced_metrics: bool = typer.Option(
        False, "--verbose", "-v", help="Display full classification reports and confusion matrices"
    )
):
    """
    Run the entire robust feature selection pipeline: Base LR (tuned), SFS, SBS, and RFE.
    """
    console.print(Panel.fit("[bold blue]ftf by rosspeili[/bold blue] - Obesity Dataset Feature Selection (MLOps Edition)", border_style="blue"))
    
    with console.status("[bold green]Loading and preprocessing data (80/20 split)...[/bold green]"):
        try:
            X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
            console.print(f"[green]✓ Data loaded successfully.[/green] Shape: Train={X_train.shape}, Test={X_test.shape}\n")
        except FileNotFoundError as exc:
            console.print(f"[bold red]Error:[/bold red] Dataset '{data_path}' not found.")
            raise typer.Exit(code=1) from exc

    # 1. Base Logistic Regression Model (with GridSearchCV)
    with console.status("[bold yellow]Tuning base Logistic Regression model via GridSearchCV...[/bold yellow]"):
        base_model, base_train_acc, base_test_acc = evaluate_base_model(X_train, y_train, X_test, y_test)
        
        if save_models or advanced_metrics:
            rep, cm, _ = evaluate_and_save_model(base_model, X_test, y_test, "base_lr_tuned", "models/" if save_models else "/tmp")
            
    base_panel = Panel(
        f"Train Accuracy: [bold white]{base_train_acc:.4f}[/bold white] | Test Accuracy (All {X_train.shape[1]} Features): [bold cyan]{base_test_acc:.4f}[/bold cyan]\n"
        f"Best Params: {base_model.best_params_}",
        title="[bold]1. Base Logistic Regression (Tuned)[/bold]",
        border_style="yellow"
    )
    console.print(base_panel)
    if advanced_metrics: print_advanced_metrics(rep, cm, "Base Model")

    # 2. Sequential Forward Selection (SFS)
    with console.status("[bold magenta]Running Sequential Forward Selection (SFS) with 5-Fold CV...[/bold magenta]"):
        sfs, _, sfs_features, sfs_train_acc, sfs_test_acc = run_sfs(X_train, y_train, X_test, y_test)
        plot_sfs_results(sfs.get_metric_dict(), 'sfs_accuracy.png', 'Step Forward Selection (SFS) CV Accuracy')
        
    sfs_feature_str = "\n".join([f"  • {f}" for f in sfs_features])
    sfs_panel = Panel(
        f"[bold]Features Chosen ({len(sfs_features)}):[/bold]\n{sfs_feature_str}\n\n"
        f"Train Accuracy: [bold white]{sfs_train_acc:.4f}[/bold white] | Test Accuracy (SFS Subset): [bold cyan]{sfs_test_acc:.4f}[/bold cyan]\n"
        f"Plot saved to: [italic]results/sfs_accuracy.png[/italic]",
        title="[bold]2. Sequential Forward Selection[/bold]",
        border_style="magenta"
    )
    console.print(sfs_panel)
    console.print()

    # 3. Sequential Backward Selection (SBS)
    with console.status("[bold cyan]Running Sequential Backward Selection (SBS) with 5-Fold CV...[/bold cyan]"):
        sbs, _, sbs_features, sbs_train_acc, sbs_test_acc = run_sbs(X_train, y_train, X_test, y_test)
        plot_sfs_results(sbs.get_metric_dict(), 'sbs_accuracy.png', 'Step Backward Selection (SBS) CV Accuracy')
        
    sbs_feature_str = "\n".join([f"  • {f}" for f in sbs_features])
    sbs_panel = Panel(
        f"[bold]Features Chosen ({len(sbs_features)}):[/bold]\n{sbs_feature_str}\n\n"
        f"Train Accuracy: [bold white]{sbs_train_acc:.4f}[/bold white] | Test Accuracy (SBS Subset): [bold cyan]{sbs_test_acc:.4f}[/bold cyan]\n"
        f"Plot saved to: [italic]results/sbs_accuracy.png[/italic]",
        title="[bold]3. Sequential Backward Floating Selection[/bold]",
        border_style="cyan"
    )
    console.print(sbs_panel)
    console.print()

    # 4. Recursive Feature Elimination (RFE)
    with console.status("[bold green]Running Recursive Feature Elimination (RFE) securely...[/bold green]"):
        rfe_model, rfe_features, rfe_train_acc, rfe_test_acc = run_rfe(X_train, y_train, X_test, y_test)
        if save_models or advanced_metrics:
            rep, cm, _ = evaluate_and_save_model(rfe_model, X_test, y_test, "rfe_pipeline", "models/" if save_models else "/tmp")
        
    rfe_feature_str = "\n".join([f"  • {f}" for f in rfe_features])
    rfe_panel = Panel(
        f"[bold]Features Chosen ({len(rfe_features)}):[/bold]\n{rfe_feature_str}\n\n"
        f"Train Accuracy: [bold white]{rfe_train_acc:.4f}[/bold white] | Test Accuracy (RFE Subset): [bold cyan]{rfe_test_acc:.4f}[/bold cyan]",
        title="[bold]4. Recursive Feature Elimination[/bold]",
        border_style="green"
    )
    console.print(rfe_panel)
    if advanced_metrics: print_advanced_metrics(rep, cm, "RFE Model")
    
    # Final Comparison Summary table
    table = Table(title="Generalization Accuracy Comparison Summary", show_header=True, header_style="bold underline")
    table.add_column("Method", style="bold")
    table.add_column("Features Count", justify="center")
    table.add_column("Train Accuracy", justify="right", style="white")
    table.add_column("Test Accuracy", justify="right", style="cyan")
    
    table.add_row("Tuned Base LR Model", str(X_train.shape[1]), f"{base_train_acc:.4f}", f"{base_test_acc:.4f}")
    table.add_row("SFS", str(len(sfs_features)), f"{sfs_train_acc:.4f}", f"{sfs_test_acc:.4f}")
    table.add_row("SBS", str(len(sbs_features)), f"{sbs_train_acc:.4f}", f"{sbs_test_acc:.4f}")
    table.add_row("RFE", str(len(rfe_features)), f"{rfe_train_acc:.4f}", f"{rfe_test_acc:.4f}")
    
    console.print(table)
    
    if save_models:
        console.print("\n[bold cyan]Models successfully evaluated and serialized to `models/`![/bold cyan]")
    
    console.print("\n[bold green]Pipeline completed successfully![/bold green] 🎉\n")


if __name__ == "__main__":
    app()

