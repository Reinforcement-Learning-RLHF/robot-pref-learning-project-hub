"""
Results dashboard — label progress and inter-annotator agreement overview.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
HUMAN_CSV = ROOT / "data" / "preferences" / "human_pairs.csv"
SCRIPTED_CSV = ROOT / "data" / "preferences" / "scripted_pairs.csv"
VLM_CSV = ROOT / "data" / "preferences" / "vlm_pairs.csv"
REGISTRY_CSV = ROOT / "data" / "rollout_registry.csv"


@st.cache_data
def load_registry() -> pd.DataFrame:
    df = pd.read_csv(REGISTRY_CSV, encoding="utf-8-sig")
    df["success_label"] = df["success_label"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    return df


def load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists() and path.stat().st_size > 0:
        df = pd.read_csv(path)
        return df if len(df) > 0 else None
    return None


def agreement_rate(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    """Fraction of shared pairs where both sources agree on the label."""
    merged = pd.merge(
        df_a[["traj_A", "traj_B", "label"]],
        df_b[["traj_A", "traj_B", "label"]],
        on=["traj_A", "traj_B"],
        suffixes=("_a", "_b"),
    )
    if merged.empty:
        return float("nan")
    return float((merged["label_a"] == merged["label_b"]).mean())


def main() -> None:
    st.set_page_config(page_title="Results Dashboard", layout="wide", page_icon="📊")
    st.title("Results Dashboard")

    # ── Coverage ──────────────────────────────────────────────────────────────
    st.header("Label Coverage")

    registry = load_registry()
    n_success = int(registry["success_label"].sum())
    n_failure = int((~registry["success_label"]).sum())
    total_possible = n_success * n_failure

    human = load_csv(HUMAN_CSV)
    scripted = load_csv(SCRIPTED_CSV)
    vlm = load_csv(VLM_CSV)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total possible pairs", total_possible)
    col2.metric("Human labels", len(human) if human is not None else 0)
    col3.metric("Scripted labels", len(scripted) if scripted is not None else 0)
    col4.metric("VLM labels", len(vlm) if vlm is not None else 0)

    # ── Human labels table ────────────────────────────────────────────────────
    st.header("Human Labels")
    if human is not None:
        label_map = {1.0: "A better", 0.0: "B better", 0.5: "Tie"}
        display = human.copy()
        display["preference"] = display["label"].map(label_map)
        st.dataframe(
            display[["traj_A", "traj_B", "preference", "timestamp"]],
            width="stretch",
        )
        st.download_button(
            "Download human_pairs.csv",
            data=HUMAN_CSV.read_bytes(),
            file_name="human_pairs.csv",
            mime="text/csv",
        )
    else:
        st.info("No human labels yet. Use the labeling page to get started.")

    # ── Agreement stats ───────────────────────────────────────────────────────
    if human is not None and (scripted is not None or vlm is not None):
        st.header("Inter-source Agreement")
        pairs_info: list[tuple[str, pd.DataFrame | None, pd.DataFrame | None]] = [
            ("Human vs Scripted", human, scripted),
            ("Human vs VLM", human, vlm),
        ]
        if scripted is not None and vlm is not None:
            pairs_info.append(("Scripted vs VLM", scripted, vlm))

        active = [(name, a, b) for name, a, b in pairs_info if a is not None and b is not None]
        if active:
            cols = st.columns(len(active))
            for col, (name, df_a, df_b) in zip(cols, active):
                rate = agreement_rate(df_a, df_b)
                col.metric(name, f"{rate:.1%}" if rate == rate else "no shared pairs")


if __name__ == "__main__":
    main()
