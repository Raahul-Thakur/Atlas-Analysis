# ATLAS Open Data Analysis Pipeline

This project is a portfolio-ready starter analysis for CERN ATLAS Open Data ROOT ntuples. It reads ROOT files with `uproot`, discovers likely trees and branches, applies flexible lepton, jet, MET, and event selections, produces physics plots, exports CSV summaries, and includes a Streamlit dashboard.

## ATLAS in One Paragraph

ATLAS is a general-purpose detector at the Large Hadron Collider. It is designed to study high-energy proton-proton collisions and reconstruct particles such as electrons, muons, photons, hadronic jets, missing transverse energy, and heavy-flavor signatures. Its physics program includes Standard Model measurements, Higgs physics, top-quark physics, electroweak processes, and searches for new particles.

## ATLAS vs ALICE

ATLAS and ALICE both sit at the LHC, but they are optimized for different physics. ALICE is built for heavy-ion collisions, where the goal is often to study hot, dense QCD matter and quark-gluon plasma signatures. ATLAS is broader and higher-rate for proton-proton physics, with strong lepton, jet, MET, b-tagging, trigger, and calorimeter capabilities aimed at hard-scattering processes such as `Z -> ll`, `W -> l nu`, top production, Higgs decays, and beyond-the-Standard-Model searches.

## Physics Concepts

- **Leptons**: Electrons and muons are clean signatures for electroweak bosons such as W and Z particles.
- **Jets**: Collimated sprays of particles created by quarks and gluons after hadronization.
- **MET**: Missing transverse energy, usually a sign of invisible particles such as neutrinos or detector effects.
- **Invariant mass**: A Lorentz-invariant quantity built from two or more particles. A `Z -> ee` or `Z -> mumu` event peaks near 91.2 GeV.
- **Transverse mass**: A mass-like observable using transverse momentum and MET. It is useful for `W -> l nu`, where the neutrino is not fully reconstructed.
- **b-tagging**: Identification of jets likely produced by bottom quarks. It is central in top-quark and Higgs analyses.

## What the Plots Show

- **Electron pT and muon pT**: Momentum scale and selection threshold behavior for reconstructed leptons.
- **Lepton eta**: Detector acceptance and angular distribution of selected leptons.
- **Dilepton invariant mass**: Reconstructs resonance structure, especially the Z peak near 91.2 GeV.
- **W transverse mass**: Shows the broad W-boson transverse-mass shape using a lepton and MET.
- **MET distribution**: Highlights invisible momentum, neutrino-rich events, or possible detector tails.
- **Jet multiplicity**: Separates simple electroweak events from jet-rich QCD or top-like events.
- **Leading jet pT**: Shows the hardest jet scale in each event.
- **b-tagged jet multiplicity**: Useful when a b-tag branch exists, especially for top-like selections.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The project also includes `atlasopenmagic`, the ATLAS Open Data helper package for discovering dataset metadata and streamable ROOT URLs.

## Data Layout

Place ATLAS Open Data ROOT files under `data/`:

```text
data/
  sample_1lep.root
```

`main` includes a compact 50,000-event sample for local use and Streamlit deployment. The full `data_A.1lep.root` dataset is stored with Git LFS on the `full-data-lfs` branch so deployment does not need to clone 1.6 GB. Other local `.root` files are ignored by Git.

The code is intentionally schema-flexible. It searches for common educational/open-data branch names for leptons, jets, MET, weights, triggers, event IDs, and b-tags. If a branch is absent, the pipeline keeps running where possible and reports missing channel-critical branches.

## Fetch URLs with ATLAS Open Magic

ATLAS recommends `atlasopenmagic` for metadata and URL discovery. You can use it through this project's CLI:

```powershell
python -m atlas_analysis.cli magic releases
python -m atlas_analysis.cli magic skims --release 2024r-pp
python -m atlas_analysis.cli magic datasets --release 2024r-pp
```

Write a short URL list:

```powershell
python -m atlas_analysis.cli magic urls --release 2024r-pp --dataset data --skim noskim --protocol https --limit 1 --output data\atlas_urls.txt
```

Then stream those ROOT files directly with uproot:

```powershell
python -m atlas_analysis.cli run --url-file data\atlas_urls.txt --max-events 10000 --channel inclusive
```

For local-file analysis, place downloaded ROOT files in `data/`. For streamed analysis, keep a URL list in `data/atlas_urls.txt`.

## Inspect a ROOT File

```powershell
python -m atlas_analysis.cli inspect data\sample.root
```

This lists trees, entry counts, and branch previews so you can choose a tree explicitly if needed.

You can also inspect a remote URL from `atlasopenmagic`:

```powershell
python -m atlas_analysis.cli inspect "simplecache::https://opendata.cern.ch/..."
```

## Run the CLI

```powershell
python -m atlas_analysis.cli run --input-folder data --pattern "*.root" --max-files 2 --max-events 10000 --channel inclusive
```

Examples:

```powershell
python -m atlas_analysis.cli run --input-folder data --channel z_ee --electron-pt 25
python -m atlas_analysis.cli run --input-folder data --channel z_mumu --muon-pt 25
python -m atlas_analysis.cli run --input-folder data --channel w_enu --met 30
python -m atlas_analysis.cli run --input-folder data --channel top --jet-pt 30 --met 30
```

Useful Z-window and trigger options:

```powershell
python -m atlas_analysis.cli run --url-file data\atlas_urls.txt --channel z_mumu --z-mass-min 80 --z-mass-max 100
python -m atlas_analysis.cli run --url-file data\atlas_urls.txt --channel z_mumu --apply-triggers
```

For a more defensible W analysis, use the one-lepton skim:

```powershell
python -m atlas_analysis.cli magic urls --release 2020e-13tev --dataset data --skim 1lep --protocol https --limit 1 --output data\atlas_w_urls.txt
python -m atlas_analysis.cli run --url-file data\atlas_w_urls.txt --channel w_munu --met 30 --w-mt-min 30 --apply-triggers
```

For large `1lep` files, direct HTTPS streaming can be more reliable than local `simplecache`:

```powershell
python -m atlas_analysis.cli magic urls --release 2020e-13tev --dataset data --skim 1lep --protocol https --no-cache --limit 1 --output data\atlas_w_urls.txt
```

If remote streaming still fails, download the ROOT file once and analyze it locally:

```powershell
python -m atlas_analysis.cli download --url-file data\atlas_w_urls.txt --output-folder data --limit 1
python -m atlas_analysis.cli run --input-folder data --pattern "data_A.1lep.root" --channel w_munu --met 30 --w-mt-min 30 --apply-triggers
```

Outputs are written to `outputs/` by default:

- `summary.csv`
- `lepton_summary.csv`
- `cutflow.csv`
- `lepton_label_counts.csv`
- `z_mass_spectrum.csv`
- `z_candidates.csv`
- `emu_control_candidates.csv`
- `os_ss_mass_spectrum.csv`
- `emu_mass_spectrum.csv`
- `w_transverse_mass.csv`
- `w_selected_transverse_mass.csv`
- `jet_summary.csv`
- `selected_events.csv`
- `run_config.json`
- `input_files.txt`
- `branch_map.json`
- `validation_report.md`
- `plots/*.png`

## Run the Dashboard

```powershell
streamlit run dashboard\app.py
```

The dashboard lets you choose the ROOT folder, file pattern, tree name, max files, max events, analysis channel, object pT cuts, plot type, and output folder.
It defaults to the deployable `data/sample_1lep.root` sample.
It also has an `atlasopenmagic` mode where you can enter a release, dataset key, skim, and protocol to stream files without manually downloading them first.
It previews `summary.csv`, `selected_events.csv`, `cutflow.csv`, the resolved branch map, channel-specific plots, and provides CSV download buttons.

Additional validation plots include:

- `z_mumu_os_ss_mass.png`
- `z_ee_os_ss_mass.png`
- `emu_control_mass.png`

The opposite-sign Z spectrum should show the Z peak much more clearly than the same-sign spectrum, while the e-mu control spectrum should not show a strong Z resonance. This is a useful sanity check for real ATLAS data.

## Notebook Demo

Open:

```text
notebooks/atlas_z_peak_demo.ipynb
```

The notebook fetches a URL with `atlasopenmagic`, inspects the ROOT file, runs `Z -> mumu`, reads the generated CSVs, and draws the Z peak from `z_mass_spectrum.csv`.

## Tests

```powershell
python -m pytest
```

The tests cover invariant mass, transverse mass, and ATLAS `lep_type` splitting for shared `lep_*` lepton branches.

## Example Result Gallery

Using:

```powershell
python -m atlas_analysis.cli magic urls --release 2020e-13tev --dataset data --skim exactly2lep --protocol https --limit 1 --output data\atlas_urls.txt
python -m atlas_analysis.cli run --url-file data\atlas_urls.txt --max-events 10000 --channel z_mumu
```

on:

```text
data_A.exactly2lep.root
```

the analysis selects a clean `Z -> mumu` sample from the first 10,000 events. The dimuon mass spectrum peaks near the expected Z boson mass of about 91.2 GeV. The channel-specific plot is written to:

```text
outputs/plots/z_mumu_dilepton_mass.png
```

The combined dilepton plot is:

```text
outputs/plots/dilepton_mass.png
```

## Normalization and Weights

If an event-weight branch is found, the pipeline stores and sums it. If no luminosity, cross-section, sum-of-weights, or sample metadata is available, plots and CSV tables should be interpreted as raw or internally weighted counts, not publication-normalized yields. The CSV histogram exports include raw and weighted count columns where possible.

## Limitations

This is a starter analysis, not a publication-quality ATLAS workflow. A full ATLAS analysis would need validated calibrations, object scale factors, trigger matching, pileup reweighting, luminosity metadata, systematic uncertainties, overlap removal, sample bookkeeping, background estimation, control regions, statistical modeling, and collaboration-approved selections. This project is built to make Open Data exploration reproducible and readable while leaving those professional analysis layers explicit.
