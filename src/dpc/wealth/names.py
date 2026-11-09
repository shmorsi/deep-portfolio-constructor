"""Strategy names, labels and colours for the wealth study: one source for the tables, figures, site export and dashboard.

Light on purpose (no NumPy, no PyTorch), so the dashboard can import it from a light install.
"""

LABELS = {
    "market": "Market (hold the index)",
    "equal_weight": "Equal weight",
    "min_variance": "Minimum variance",
    "risk_parity": "Risk parity",
    "growth_classical": "Classical growth optimiser",
    "momentum_tilt": "Momentum tilt",
    "deep": "Deep Growth Network",
    "deep_trade_all": "Deep, no learned holding",
    "deep_mlp": "Deep, no attention",
    "deep_unanchored": "Deep, no market anchor",
    "deep_pair": "Deep Growth Network, seeds 42 and 43",
}
MAIN = ["market", "equal_weight", "min_variance", "risk_parity", "growth_classical", "momentum_tilt", "deep"]
ABLATIONS = ["deep_trade_all", "deep_mlp", "deep_unanchored"]
REFERENCE = ["deep_pair"]  # the primary network with the ablations' seeds only: their like-for-like comparison
SECONDARY = ["equal_weight", "momentum_tilt", "growth_classical"]
COLORS = {"market": "#111827", "equal_weight": "#65a30d", "min_variance": "#0d9488", "risk_parity": "#7c3aed",
          "growth_classical": "#ea580c", "momentum_tilt": "#db2777", "deep": "#1d6fe0",
          "deep_trade_all": "#93c5fd", "deep_mlp": "#60a5fa", "deep_unanchored": "#3b82f6", "deep_pair": "#1e40af"}
