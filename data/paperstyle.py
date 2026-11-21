import seaborn as sns

# Okabe-Ito palette
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]

def use(context: str = "paper"):
    sns.set_theme(
        context=context,            # size
        style="whitegrid",          # background
        palette=OKABE_ITO,
        rc={
            # grid
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.color": "#D3D7DC",
            "axes.axisbelow": True,
            # spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.8,
            # fonts
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,         # ASCII minus
            "mathtext.fontset": "dejavuserif",   # math font
            # markers
            "lines.markersize": 5,
            "lines.linewidth": 1.6,
            # ticks
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3, "ytick.major.size": 3,
            # legend
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#D3D7DC",
            # export
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        },
    )
