"""
Scripted preference pairs — rule-based evaluation results.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SCRIPTED_CSV = ROOT / "data" / "preferences" / "scripted_pairs.csv"


def main() -> None:
    st.title("🔧 Scripted Preferences")
    st.caption("Rule-based preference labels generated automatically from rollout metadata.")
    st.divider()

    if not SCRIPTED_CSV.exists() or SCRIPTED_CSV.stat().st_size == 0:
        st.info(
            "No scripted pairs found. Run the scripted evaluator to generate them:\n\n"
            "```bash\npython scripts/gen_scripted_pairs.py\n```",
            icon="📭",
        )
        return

    df = pd.read_csv(SCRIPTED_CSV)

    # Stats
    total = len(df)
    a_wins = int((df["label"] == 1.0).sum())
    ties   = int((df["label"] == 0.5).sum())
    b_wins = int((df["label"] == 0.0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total pairs", total)
    c2.metric("A wins", a_wins)
    c3.metric("Ties", ties)
    c4.metric("B wins", b_wins)

    st.divider()
    st.dataframe(df, width="stretch", hide_index=True)


main()
