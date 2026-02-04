import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def plot_km_curve(df, time_col='OS', status_col='Status', score_col='RiskScore',
                  quantile=0.5, title='Kaplan-Meier Survival Curve',
                  figsize=(7, 5), show_pvalue=True, fontsize=12, save_path=None):
    """
    Plot Kaplan-Meier survival curve and display log-rank test p-value.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing survival data with at least OS, Status, RiskScore columns.
    time_col : str
        Column name for survival time.
    status_col : str
        Column name for survival status (1=event, 0=censored).
    score_col : str
        Column name for risk score.
    quantile : float, default=0.5
        Grouping threshold (default median split for high/low risk).
        Can be set to 0.33 / 0.67 for tertile split.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    show_pvalue : bool
        Whether to display p-value on the plot.
    fontsize : int
        Font size for text in the plot.
    save_path : str, optional
        Path to save the figure. If None, display the plot.

    Returns:
    ----------
    p_value : float
        Log-rank test p-value.
    """

    # Check if columns exist
    for col in [time_col, status_col, score_col]:
        if col not in df.columns:
            raise ValueError(f"DataFrame missing column: {col}")

    # Remove missing values
    df = df[[time_col, status_col, score_col]].dropna().copy()

    # Split into high/low risk groups by quantile
    threshold = df[score_col].quantile(quantile)
    df['Group'] = ['High' if x > threshold else 'Low' for x in df[score_col]]

    # Kaplan-Meier fitter
    kmf = KaplanMeierFitter()
    plt.figure(figsize=figsize)

    colors = {'High': 'red', 'Low': 'green'}
    for group in ['High', 'Low']:
        ix = df['Group'] == group
        kmf.fit(
            durations=df.loc[ix, time_col],
            event_observed=df.loc[ix, status_col],
            label=f"{group} risk"
        )
        kmf.plot_survival_function(ci_show=True, color=colors[group])

    # log-rank test
    results = logrank_test(
        df.loc[df['Group'] == 'High', time_col],
        df.loc[df['Group'] == 'Low', time_col],
        event_observed_A=df.loc[df['Group'] == 'High', status_col],
        event_observed_B=df.loc[df['Group'] == 'Low', status_col]
    )
    p_value = results.p_value

    # Format plot
    plt.title(title, fontsize=fontsize+2)
    plt.xlabel("Time", fontsize=fontsize)
    plt.ylabel("Survival Probability", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(alpha=0.3, linestyle='--')

    # Display p-value on plot
    if show_pvalue:
        if p_value < 1e-4:
            p_text = "p-value < 0.0001"
        else:
            p_text = f"p-value = {p_value:.4f}"
        plt.text(
            0.95, 0.95, p_text,
            transform=plt.gca().transAxes,
            fontsize=fontsize,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )

    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

    print(f"Log-rank test p-value = {p_value:.4e}")
    return p_value


def main():
    parser = argparse.ArgumentParser(description='Plot Kaplan-Meier survival curve from CSV file')
    parser.add_argument('--csv', type=str, default='results.csv',
                        help='Path to CSV file containing survival data')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the output figure')
    parser.add_argument('--time_col', type=str, default='OS',
                        help='Column name for survival time (default: OS)')
    parser.add_argument('--status_col', type=str, default='Status',
                        help='Column name for survival status (default: Status)')
    parser.add_argument('--score_col', type=str, default='RiskScore',
                        help='Column name for risk score (default: RiskScore)')
    parser.add_argument('--quantile', type=float, default=0.5,
                        help='Quantile threshold for grouping (default: 0.5)')
    parser.add_argument('--title', type=str, default='TCGA-COAD',
                        help='Plot title (default: Kaplan-Meier Survival Curve)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[7, 5],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Figure size in inches (default: 7 5)')
    parser.add_argument('--fontsize', type=int, default=12,
                        help='Font size for text (default: 12)')
    parser.add_argument('--no_pvalue', action='store_true',
                        help='Do not display p-value on the plot')
    
    args = parser.parse_args()
    
    # Read CSV file
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    
    df = pd.read_csv(args.csv)
    
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.csv), 'km_curve.png')
    
    # Plot KM curve
    plot_km_curve(
        df=df,
        time_col=args.time_col,
        status_col=args.status_col,
        score_col=args.score_col,
        quantile=args.quantile,
        title=args.title,
        figsize=tuple(args.figsize),
        show_pvalue=not args.no_pvalue,
        fontsize=args.fontsize,
        save_path=args.output
    )


if __name__ == '__main__':
    main()
