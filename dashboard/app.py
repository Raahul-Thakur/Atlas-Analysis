"""Streamlit dashboard for the ATLAS Open Data starter analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from atlas_analysis.physics import ObjectCuts
from atlas_analysis.pipeline import AnalysisConfig, run_analysis
from atlas_analysis.schema import describe_inspection, list_root_files
from atlas_analysis.data_access import get_urls


st.set_page_config(page_title="ATLAS Open Data Analysis", layout="wide")
st.title("ATLAS Open Data Analysis")

with st.sidebar:
    st.header("Input")
    use_magic = st.checkbox("Use atlasopenmagic URLs")
    input_folder = st.text_input("ROOT file folder", "data")
    pattern = st.text_input("File pattern", "sample_1lep.root")
    release = st.text_input("ATLAS release", "2024r-pp", disabled=not use_magic)
    dataset = st.text_input("Dataset key", "data", disabled=not use_magic)
    skim = st.text_input("Skim", "noskim", disabled=not use_magic)
    protocol = st.selectbox("Protocol", ["https", "root", "eos"], disabled=not use_magic)
    tree_name = st.text_input("Tree name", "")
    max_files = st.number_input("Max files", min_value=0, value=1, step=1)
    max_events = st.number_input("Max events per file", min_value=0, value=5000, step=1000)

    st.header("Channel")
    channel = st.selectbox(
        "Analysis channel",
        ["inclusive", "z_ee", "z_mumu", "w_enu", "w_munu", "jets", "top"],
        format_func=lambda x: {
            "inclusive": "inclusive",
            "z_ee": "Z -> ee",
            "z_mumu": "Z -> mumu",
            "w_enu": "W -> e nu",
            "w_munu": "W -> mu nu",
            "jets": "jets",
            "top": "top-like",
        }[x],
    )

    st.header("Object Cuts")
    electron_pt = st.slider("Electron pT [GeV]", 5.0, 80.0, 25.0, 1.0)
    electron_eta = st.slider("Electron |eta|", 1.0, 3.0, 2.47, 0.01)
    muon_pt = st.slider("Muon pT [GeV]", 5.0, 80.0, 25.0, 1.0)
    muon_eta = st.slider("Muon |eta|", 1.0, 3.0, 2.5, 0.01)
    jet_pt = st.slider("Jet pT [GeV]", 10.0, 120.0, 30.0, 1.0)
    jet_eta = st.slider("Jet |eta|", 1.0, 5.0, 2.5, 0.1)
    met_cut = st.slider("MET [GeV]", 0.0, 120.0, 25.0, 1.0)
    w_mt_min = st.slider("W transverse mass min [GeV]", 0.0, 120.0, 30.0, 1.0)
    z_mass_min, z_mass_max = st.slider("Z mass window [GeV]", 50.0, 130.0, (80.0, 100.0), 1.0)
    apply_triggers = st.checkbox("Apply channel trigger cuts")

    st.header("Output")
    plot_type = st.selectbox(
        "Plot type",
        [
            "electron_pt",
            "muon_pt",
            "lepton_eta",
            "dilepton_mass",
            "z_ee_dilepton_mass",
            "z_mumu_dilepton_mass",
            "z_ee_os_ss_mass",
            "z_mumu_os_ss_mass",
            "emu_control_mass",
            "w_transverse_mass",
            "w_enu_transverse_mass",
            "w_munu_transverse_mass",
            "met",
            "jet_multiplicity",
            "leading_jet_pt",
            "btag_jet_multiplicity",
        ],
    )
    output_folder = st.text_input("Output folder", "outputs")
    run_button = st.button("Run analysis", type="primary")

files = list_root_files(input_folder, pattern, int(max_files) or None) if Path(input_folder).exists() and not use_magic else []
st.caption(f"Matched {len(files)} ROOT file(s).")

if files:
    with st.expander("Inspect first ROOT file"):
        if st.button("Inspect branches"):
            try:
                st.code(describe_inspection(files[0]))
            except Exception as exc:
                st.error(str(exc))

if run_button:
    try:
        with st.spinner("Running ATLAS analysis..."):
            input_files = get_urls(release, dataset, skim, protocol, limit=int(max_files) or None) if use_magic else None
            result = run_analysis(
                AnalysisConfig(
                    input_folder=input_folder,
                    input_files=input_files,
                    pattern=pattern,
                    output_folder=output_folder,
                    tree_name=tree_name.strip() or None,
                    max_files=int(max_files) or None,
                    max_events=int(max_events) or None,
                    channel=channel,
                    cuts=ObjectCuts(
                        electron_pt=electron_pt,
                        electron_abs_eta=electron_eta,
                        muon_pt=muon_pt,
                        muon_abs_eta=muon_eta,
                        jet_pt=jet_pt,
                        jet_abs_eta=jet_eta,
                        met=met_cut,
                        w_mt_min=w_mt_min,
                        z_mass_min=z_mass_min,
                        z_mass_max=z_mass_max,
                    ),
                    apply_triggers=apply_triggers,
                )
            )
        st.success(f"Wrote outputs to {result.output_folder}")
        st.write("Resolved branch map")
        st.json(result.branch_map.present())
    except Exception as exc:
        st.error(str(exc))

out = Path(output_folder)
cols = st.columns([1, 1])
with cols[0]:
    st.subheader("Summary")
    summary_path = out / "summary.csv"
    if summary_path.exists():
        st.dataframe(pd.read_csv(summary_path), use_container_width=True)
    else:
        st.info("Run the analysis to create summary.csv.")
    branch_map_path = out / "branch_map.json"
    if branch_map_path.exists():
        st.subheader("Branch Map")
        st.json(branch_map_path.read_text(encoding="utf-8"))

with cols[1]:
    st.subheader("Selected Events")
    selected_path = out / "selected_events.csv"
    if selected_path.exists():
        st.dataframe(pd.read_csv(selected_path).head(200), use_container_width=True)
    else:
        st.info("Run the analysis to create selected_events.csv.")
    cutflow_path = out / "cutflow.csv"
    if cutflow_path.exists():
        st.subheader("Cutflow")
        st.dataframe(pd.read_csv(cutflow_path), use_container_width=True)

st.subheader("Downloads")
csv_paths = sorted(out.glob("*.csv")) if out.exists() else []
download_cols = st.columns(3)
for index, csv_path in enumerate(csv_paths):
    with download_cols[index % 3]:
        st.download_button(
            csv_path.name,
            data=csv_path.read_bytes(),
            file_name=csv_path.name,
            mime="text/csv",
        )

st.subheader("Plot")
plot_path = out / "plots" / f"{plot_type}.png"
if plot_path.exists():
    st.image(str(plot_path), use_container_width=True)
else:
    st.info(f"No plot found yet: {plot_path}")
