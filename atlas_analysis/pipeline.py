"""End-to-end ATLAS Open Data starter analysis pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import awkward as ak
import numpy as np
import pandas as pd
import uproot
from tqdm import tqdm

from .physics import ObjectCuts, count_true, invariant_mass, leading_or_nan, select_objects, to_gev, transverse_mass
from .plots import histogram_table, save_histogram, save_overlay_histogram
from .schema import BranchMap, atlas_access_paths, choose_tree, list_root_files, resolve_branches

Channel = Literal["inclusive", "z_ee", "z_mumu", "w_enu", "w_munu", "jets", "top"]


@dataclass
class AnalysisConfig:
    input_folder: str = "data"
    input_files: list[str] | None = None
    pattern: str = "*.root"
    output_folder: str = "outputs"
    tree_name: str | None = None
    max_files: int | None = None
    max_events: int | None = None
    channel: Channel = "inclusive"
    cuts: ObjectCuts = field(default_factory=ObjectCuts)
    make_plots: bool = True
    apply_triggers: bool = False


@dataclass
class AnalysisResult:
    output_folder: Path
    files: list[str]
    branch_map: BranchMap
    summary: pd.DataFrame
    selected_events: pd.DataFrame
    cutflow: pd.DataFrame


def _branches_to_read(branch_map: BranchMap) -> list[str]:
    return sorted(set(branch_map.present().values()))


def _read_file(path: str, config: AnalysisConfig, first_map: BranchMap | None = None):
    last_error: Exception | None = None
    tried = atlas_access_paths(path)
    for access_path in tried:
        try:
            tree_info = choose_tree(access_path, config.tree_name)
            break
        except Exception as exc:
            last_error = exc
    else:
        attempts = "\n".join(f"- {item}" for item in tried)
        raise OSError(f"Could not inspect ROOT file after trying:\n{attempts}\nLast error: {last_error}") from last_error
    branch_map = first_map or resolve_branches(tree_info.branches)
    branches = [branch for branch in _branches_to_read(branch_map) if branch in tree_info.branches]
    if not branches:
        raise ValueError(f"No recognizable ATLAS analysis branches found in {path} tree '{tree_info.tree_name}'.")
    for access_path in tried:
        try:
            with uproot.open(access_path) as root_file:
                tree = root_file[tree_info.tree_name]
                events = tree.arrays(branches, entry_stop=config.max_events, library="ak")
            return events, branch_map, tree_info.tree_name
        except Exception as exc:
            last_error = exc
    attempts = "\n".join(f"- {item}" for item in tried)
    raise OSError(f"Could not read ROOT file after trying:\n{attempts}\nLast error: {last_error}") from last_error


def _weights(events, branch_map: BranchMap) -> np.ndarray:
    if branch_map.weight and branch_map.weight in events.fields:
        weights = np.asarray(ak.to_numpy(events[branch_map.weight]), dtype=float)
        finite = np.isfinite(weights)
        if finite.any() and np.any(weights[finite] != 0):
            return weights
    return np.ones(len(events), dtype=float)


def _source_type(path: str) -> str:
    text = str(path).lower()
    name = Path(urlparse(str(path).removeprefix("simplecache::")).path).name.lower()
    if "/data/" in text or name.startswith("data_") or name.startswith("data."):
        return "data"
    if "/mc/" in text or name.startswith("mc_") or name.startswith("mc."):
        return "mc"
    return "unknown"


def _build_objects(events, branch_map: BranchMap, cuts: ObjectCuts):
    objects = {"electrons": None, "muons": None, "jets": None}
    if branch_map.electron_pt and branch_map.electron_pt in events.fields:
        flavor_mask = True
        if branch_map.lepton_type in events.fields and branch_map.electron_pt == branch_map.muon_pt:
            flavor_mask = abs(events[branch_map.lepton_type]) == 11
        mask = select_objects(
            events[branch_map.electron_pt],
            events[branch_map.electron_eta] if branch_map.electron_eta in events.fields else None,
            events[branch_map.electron_quality] if branch_map.electron_quality in events.fields else None,
            events[branch_map.electron_iso] if branch_map.electron_iso in events.fields else None,
            cuts.electron_pt,
            cuts.electron_abs_eta,
            cuts.max_iso_ratio,
        )
        mask = mask & flavor_mask
        objects["electrons"] = {
            "pt": to_gev(events[branch_map.electron_pt][mask]),
            "eta": events[branch_map.electron_eta][mask] if branch_map.electron_eta in events.fields else None,
            "phi": events[branch_map.electron_phi][mask] if branch_map.electron_phi in events.fields else None,
            "e": to_gev(events[branch_map.electron_e][mask]) if branch_map.electron_e in events.fields else None,
            "charge": events[branch_map.electron_charge][mask] if branch_map.electron_charge in events.fields else None,
        }
    if branch_map.muon_pt and branch_map.muon_pt in events.fields:
        flavor_mask = True
        if branch_map.lepton_type in events.fields and branch_map.electron_pt == branch_map.muon_pt:
            flavor_mask = abs(events[branch_map.lepton_type]) == 13
        mask = select_objects(
            events[branch_map.muon_pt],
            events[branch_map.muon_eta] if branch_map.muon_eta in events.fields else None,
            events[branch_map.muon_quality] if branch_map.muon_quality in events.fields else None,
            events[branch_map.muon_iso] if branch_map.muon_iso in events.fields else None,
            cuts.muon_pt,
            cuts.muon_abs_eta,
            cuts.max_iso_ratio,
        )
        mask = mask & flavor_mask
        objects["muons"] = {
            "pt": to_gev(events[branch_map.muon_pt][mask]),
            "eta": events[branch_map.muon_eta][mask] if branch_map.muon_eta in events.fields else None,
            "phi": events[branch_map.muon_phi][mask] if branch_map.muon_phi in events.fields else None,
            "e": to_gev(events[branch_map.muon_e][mask]) if branch_map.muon_e in events.fields else None,
            "charge": events[branch_map.muon_charge][mask] if branch_map.muon_charge in events.fields else None,
        }
    if branch_map.jet_pt and branch_map.jet_pt in events.fields:
        mask = to_gev(events[branch_map.jet_pt]) >= cuts.jet_pt
        if branch_map.jet_eta in events.fields:
            mask = mask & (abs(events[branch_map.jet_eta]) <= cuts.jet_abs_eta)
        objects["jets"] = {
            "pt": to_gev(events[branch_map.jet_pt][mask]),
            "eta": events[branch_map.jet_eta][mask] if branch_map.jet_eta in events.fields else None,
            "btag": events[branch_map.jet_btag][mask] if branch_map.jet_btag in events.fields else None,
        }
    return objects


def _pair_masses(obj, sign: str = "opposite"):
    if not obj or obj["eta"] is None or obj["phi"] is None:
        return np.array([])
    pairs = ak.combinations(ak.zip(obj), 2, fields=["a", "b"])
    masses = invariant_mass(
        pairs.a.pt,
        pairs.a.eta,
        pairs.a.phi,
        pairs.b.pt,
        pairs.b.eta,
        pairs.b.phi,
        pairs.a.e if obj["e"] is not None else None,
        pairs.b.e if obj["e"] is not None else None,
    )
    if obj["charge"] is not None:
        product = pairs.a.charge * pairs.b.charge
        if sign == "opposite":
            masses = masses[product < 0]
        elif sign == "same":
            masses = masses[product > 0]
    return ak.to_numpy(ak.flatten(masses, axis=None))


def _pair_candidates(obj, frame: pd.DataFrame, channel: str, cuts: ObjectCuts) -> pd.DataFrame:
    if not obj or obj["eta"] is None or obj["phi"] is None:
        return pd.DataFrame()
    pairs = ak.combinations(ak.zip(obj), 2, fields=["a", "b"])
    masses = invariant_mass(
        pairs.a.pt,
        pairs.a.eta,
        pairs.a.phi,
        pairs.b.pt,
        pairs.b.eta,
        pairs.b.phi,
        pairs.a.e if obj["e"] is not None else None,
        pairs.b.e if obj["e"] is not None else None,
    )
    charge_product = pairs.a.charge * pairs.b.charge if obj["charge"] is not None else ak.ones_like(masses) * -1
    pair_event_index = ak.broadcast_arrays(ak.local_index(obj["pt"], axis=0), obj["pt"])[0]
    pair_event_index = ak.combinations(pair_event_index, 2, fields=["a", "b"]).a
    flat_idx = ak.to_numpy(ak.flatten(pair_event_index, axis=None))
    flat_mass = ak.to_numpy(ak.flatten(masses, axis=None))
    flat_pt1 = ak.to_numpy(ak.flatten(pairs.a.pt, axis=None))
    flat_pt2 = ak.to_numpy(ak.flatten(pairs.b.pt, axis=None))
    flat_charge_product = ak.to_numpy(ak.flatten(charge_product, axis=None))
    if len(flat_mass) == 0:
        return pd.DataFrame()
    leading_pt = np.maximum(flat_pt1, flat_pt2)
    subleading_pt = np.minimum(flat_pt1, flat_pt2)
    candidates = pd.DataFrame(
        {
            "source_file": frame.iloc[flat_idx]["source_file"].to_numpy(),
            "event_number": frame.iloc[flat_idx]["event_number"].to_numpy(),
            "channel": channel,
            "mass_gev": flat_mass,
            "leading_lepton_pt_gev": leading_pt,
            "subleading_lepton_pt_gev": subleading_pt,
            "opposite_sign": flat_charge_product < 0,
            "in_z_window": (flat_mass >= cuts.z_mass_min) & (flat_mass <= cuts.z_mass_max),
        }
    )
    candidates["same_sign"] = flat_charge_product > 0
    return candidates.reset_index(drop=True)


def _cross_candidates(electrons, muons, frame: pd.DataFrame, cuts: ObjectCuts) -> pd.DataFrame:
    if not electrons or not muons or electrons["eta"] is None or muons["eta"] is None or electrons["phi"] is None or muons["phi"] is None:
        return pd.DataFrame()
    pairs = ak.cartesian({"electron": ak.zip(electrons), "muon": ak.zip(muons)}, axis=1)
    masses = invariant_mass(
        pairs.electron.pt,
        pairs.electron.eta,
        pairs.electron.phi,
        pairs.muon.pt,
        pairs.muon.eta,
        pairs.muon.phi,
        pairs.electron.e if electrons["e"] is not None else None,
        pairs.muon.e if muons["e"] is not None else None,
    )
    charge_product = pairs.electron.charge * pairs.muon.charge if electrons["charge"] is not None and muons["charge"] is not None else ak.ones_like(masses) * -1
    event_index = ak.broadcast_arrays(ak.local_index(electrons["pt"], axis=0), masses)[0]
    flat_idx = ak.to_numpy(ak.flatten(event_index, axis=None))
    flat_mass = ak.to_numpy(ak.flatten(masses, axis=None))
    if len(flat_mass) == 0:
        return pd.DataFrame()
    flat_ept = ak.to_numpy(ak.flatten(pairs.electron.pt, axis=None))
    flat_mpt = ak.to_numpy(ak.flatten(pairs.muon.pt, axis=None))
    flat_charge_product = ak.to_numpy(ak.flatten(charge_product, axis=None))
    return pd.DataFrame(
        {
            "source_file": frame.iloc[flat_idx]["source_file"].to_numpy(),
            "event_number": frame.iloc[flat_idx]["event_number"].to_numpy(),
            "channel": "e-mu",
            "mass_gev": flat_mass,
            "leading_lepton_pt_gev": np.maximum(flat_ept, flat_mpt),
            "subleading_lepton_pt_gev": np.minimum(flat_ept, flat_mpt),
            "opposite_sign": flat_charge_product < 0,
            "same_sign": flat_charge_product > 0,
            "in_z_window": (flat_mass >= cuts.z_mass_min) & (flat_mass <= cuts.z_mass_max),
        }
    )


def _w_mt(obj, met_et, met_phi):
    if not obj or obj["phi"] is None or met_et is None or met_phi is None:
        return np.array([])
    leading_pt = leading_or_nan(obj["pt"])
    leading_phi = leading_or_nan(obj["phi"])
    valid = np.isfinite(ak.to_numpy(leading_pt))
    mt = transverse_mass(leading_pt[valid], leading_phi[valid], met_et[valid], met_phi[valid])
    return np.asarray(ak.to_numpy(mt), dtype=float)


def _w_mt_per_event(obj, met_et, met_phi, n_events: int) -> np.ndarray:
    values = np.full(n_events, np.nan)
    if not obj or obj["phi"] is None or met_et is None or met_phi is None:
        return values
    leading_pt = leading_or_nan(obj["pt"])
    leading_phi = leading_or_nan(obj["phi"])
    valid = np.isfinite(ak.to_numpy(leading_pt))
    if np.any(valid):
        values[valid] = np.asarray(ak.to_numpy(transverse_mass(leading_pt[valid], leading_phi[valid], met_et[valid], met_phi[valid])), dtype=float)
    return values


def _has_same_sign_pairs(obj) -> np.ndarray | None:
    if not obj or obj["charge"] is None:
        return None
    pairs = ak.combinations(obj["charge"], 2, fields=["a", "b"])
    return ak.to_numpy(ak.any((pairs.a * pairs.b) > 0, axis=1))


def _event_frame(events, branch_map: BranchMap, objects, weights, source_file: str, source_kind: str) -> pd.DataFrame:
    n = len(events)
    met = to_gev(events[branch_map.met_et]) if branch_map.met_et in events.fields else np.full(n, np.nan)
    electrons = objects["electrons"]
    muons = objects["muons"]
    jets = objects["jets"]
    data = {
        "source_file": [source_file] * n,
        "source_type": [source_kind] * n,
        "event_number": ak.to_numpy(events[branch_map.event_number]) if branch_map.event_number in events.fields else np.arange(n),
        "weight": weights,
        "n_electrons": count_true(electrons["pt"] > -1) if electrons else np.zeros(n, dtype=int),
        "n_muons": count_true(muons["pt"] > -1) if muons else np.zeros(n, dtype=int),
        "n_jets": count_true(jets["pt"] > -1) if jets else np.zeros(n, dtype=int),
        "met_gev": ak.to_numpy(met),
        "leading_electron_pt_gev": ak.to_numpy(leading_or_nan(electrons["pt"])) if electrons else np.full(n, np.nan),
        "leading_muon_pt_gev": ak.to_numpy(leading_or_nan(muons["pt"])) if muons else np.full(n, np.nan),
        "leading_jet_pt_gev": ak.to_numpy(leading_or_nan(jets["pt"])) if jets else np.full(n, np.nan),
    }
    met_phi = events[branch_map.met_phi] if branch_map.met_phi in events.fields else None
    data["electron_mt_gev"] = _w_mt_per_event(electrons, met, met_phi, n)
    data["muon_mt_gev"] = _w_mt_per_event(muons, met, met_phi, n)
    same_sign = np.full(n, False)
    for obj in [electrons, muons]:
        flags = _has_same_sign_pairs(obj)
        if flags is not None:
            same_sign = same_sign | flags
    data["has_same_sign_leptons"] = same_sign
    if branch_map.electron_trigger in events.fields:
        data["trigE"] = ak.to_numpy(events[branch_map.electron_trigger]).astype(bool)
    else:
        data["trigE"] = np.full(n, True)
    if branch_map.muon_trigger in events.fields:
        data["trigM"] = ak.to_numpy(events[branch_map.muon_trigger]).astype(bool)
    else:
        data["trigM"] = np.full(n, True)
    if jets and jets["btag"] is not None:
        data["n_btag_jets"] = count_true(jets["btag"] != 0)
    return pd.DataFrame(data)


def _channel_mask(frame: pd.DataFrame, channel: Channel, cuts: ObjectCuts, apply_triggers: bool = False) -> pd.Series:
    trigger = pd.Series([True] * len(frame), index=frame.index)
    if apply_triggers:
        if channel in {"z_ee", "w_enu"}:
            trigger = frame["trigE"]
        elif channel in {"z_mumu", "w_munu"}:
            trigger = frame["trigM"]
        elif channel == "inclusive":
            trigger = frame["trigE"] | frame["trigM"]
    if channel == "z_ee":
        return trigger & (frame["n_electrons"] >= 2) & frame["has_os_ee"]
    if channel == "z_mumu":
        return trigger & (frame["n_muons"] >= 2) & frame["has_os_mumu"]
    if channel == "w_enu":
        return trigger & (frame["n_electrons"] == 1) & (frame["n_muons"] == 0) & (frame["met_gev"] >= cuts.met) & (frame["electron_mt_gev"] >= cuts.w_mt_min)
    if channel == "w_munu":
        return trigger & (frame["n_muons"] == 1) & (frame["n_electrons"] == 0) & (frame["met_gev"] >= cuts.met) & (frame["muon_mt_gev"] >= cuts.w_mt_min)
    if channel == "jets":
        return trigger & (frame["n_jets"] >= 1)
    if channel == "top":
        btag_ok = frame.get("n_btag_jets", pd.Series(np.zeros(len(frame), dtype=int))) >= 1
        return trigger & (frame["n_jets"] >= 3) & (frame["met_gev"] >= cuts.met) & ((frame["n_electrons"] + frame["n_muons"]) >= 1) & btag_ok
    return trigger


def _add_event_flags(frame: pd.DataFrame, candidates: pd.DataFrame, cuts: ObjectCuts) -> pd.DataFrame:
    frame = frame.copy()
    frame["has_os_ee"] = False
    frame["has_os_mumu"] = False
    frame["has_em_control"] = (frame["n_electrons"] >= 1) & (frame["n_muons"] >= 1)
    if "has_same_sign_leptons" not in frame:
        frame["has_same_sign_leptons"] = False
    frame["z_ee_in_window"] = False
    frame["z_mumu_in_window"] = False
    frame["best_z_ee_mass_gev"] = np.nan
    frame["best_z_mumu_mass_gev"] = np.nan
    if candidates.empty:
        return frame
    keyed = frame.reset_index().set_index(["source_file", "event_number"])
    for channel, os_col, window_col, mass_col in [
        ("Z->ee", "has_os_ee", "z_ee_in_window", "best_z_ee_mass_gev"),
        ("Z->mumu", "has_os_mumu", "z_mumu_in_window", "best_z_mumu_mass_gev"),
    ]:
        channel_candidates = candidates[candidates["channel"] == channel].copy()
        if channel_candidates.empty:
            continue
        channel_candidates["distance_to_z"] = (channel_candidates["mass_gev"] - 91.1876).abs()
        best = channel_candidates.sort_values("distance_to_z").drop_duplicates(["source_file", "event_number"])
        for row in best.itertuples(index=False):
            key = (row.source_file, row.event_number)
            if key in keyed.index:
                idx = keyed.loc[key, "index"]
                if row.opposite_sign:
                    frame.loc[idx, os_col] = True
                    frame.loc[idx, mass_col] = row.mass_gev
                    frame.loc[idx, window_col] = cuts.z_mass_min <= row.mass_gev <= cuts.z_mass_max
    return frame


def _lepton_label_counts(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"label": "opposite-sign ee", "count": int(frame["has_os_ee"].sum())},
        {"label": "opposite-sign mumu", "count": int(frame["has_os_mumu"].sum())},
        {"label": "e-mu control-like events", "count": int(frame["has_em_control"].sum())},
        {"label": "same-sign leptons", "count": int(frame["has_same_sign_leptons"].sum())},
    ]
    return pd.DataFrame(rows)


def _cutflow(frame: pd.DataFrame, channel: Channel, cuts: ObjectCuts, apply_triggers: bool) -> pd.DataFrame:
    flow = []
    current = pd.Series([True] * len(frame), index=frame.index)

    def add(step: str, mask: pd.Series) -> None:
        nonlocal current
        current = current & mask
        flow.append({"channel": channel, "step": step, "events": int(current.sum()), "weighted_events": float(frame.loc[current, "weight"].sum())})

    flow.append({"channel": channel, "step": "all events", "events": len(frame), "weighted_events": float(frame["weight"].sum())})
    if channel in {"z_ee", "w_enu"}:
        trigger_mask = frame["trigE"] if apply_triggers else pd.Series([True] * len(frame), index=frame.index)
    elif channel in {"z_mumu", "w_munu"}:
        trigger_mask = frame["trigM"] if apply_triggers else pd.Series([True] * len(frame), index=frame.index)
    elif channel == "inclusive":
        trigger_mask = (frame["trigE"] | frame["trigM"]) if apply_triggers else pd.Series([True] * len(frame), index=frame.index)
    else:
        trigger_mask = pd.Series([True] * len(frame), index=frame.index)
    add("trigger passed", trigger_mask)
    if channel == "z_ee":
        add(">=2 leptons", frame["n_electrons"] >= 2)
        add("opposite-sign leptons", frame["has_os_ee"])
        add("same-flavor leptons", frame["n_electrons"] >= 2)
        add("pT/eta cuts", frame["n_electrons"] >= 2)
        add("Z mass window", frame["z_ee_in_window"])
    elif channel == "z_mumu":
        add(">=2 leptons", frame["n_muons"] >= 2)
        add("opposite-sign leptons", frame["has_os_mumu"])
        add("same-flavor leptons", frame["n_muons"] >= 2)
        add("pT/eta cuts", frame["n_muons"] >= 2)
        add("Z mass window", frame["z_mumu_in_window"])
    elif channel in {"w_enu", "w_munu"}:
        lep_col = "n_electrons" if channel == "w_enu" else "n_muons"
        other_col = "n_muons" if channel == "w_enu" else "n_electrons"
        mt_col = "electron_mt_gev" if channel == "w_enu" else "muon_mt_gev"
        add("exactly one lepton", frame[lep_col] == 1)
        add("second lepton veto", frame[other_col] == 0)
        add("MET cut", frame["met_gev"] >= cuts.met)
        add("transverse mass cut", frame[mt_col] >= cuts.w_mt_min)
    else:
        add("object selection", _channel_mask(frame, channel, cuts, apply_triggers=False))
    final_mask = _channel_mask(frame, channel, cuts, apply_triggers=apply_triggers)
    if channel in {"z_ee", "z_mumu"}:
        z_col = "z_ee_in_window" if channel == "z_ee" else "z_mumu_in_window"
        final_mask = final_mask & frame[z_col]
    flow.append({"channel": channel, "step": "final selected", "events": int(final_mask.sum()), "weighted_events": float(frame.loc[final_mask, "weight"].sum())})
    return pd.DataFrame(flow)


def run_analysis(config: AnalysisConfig) -> AnalysisResult:
    files = config.input_files or list_root_files(config.input_folder, config.pattern, config.max_files)
    if not files:
        raise FileNotFoundError(f"No ROOT files matched '{config.pattern}' in {config.input_folder}")

    output = Path(config.output_folder)
    output.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    all_candidates: list[pd.DataFrame] = []
    z_ee, z_mumu, z_ee_ss, z_mumu_ss, emu_os, emu_ss, w_enu, w_munu = [], [], [], [], [], [], [], []
    electron_pts, muon_pts, lepton_etas, met_values, jet_mult, lead_jet_pts, btag_mult = [], [], [], [], [], [], []
    branch_map: BranchMap | None = None
    tree_names: set[str] = set()

    for path in tqdm(files, desc="Analyzing ROOT files"):
        events, branch_map, tree_name = _read_file(path, config, branch_map)
        tree_names.add(tree_name)
        missing = branch_map.required_for(config.channel)
        if missing:
            print(f"Warning: channel '{config.channel}' is missing logical branches: {', '.join(missing)}")
        weights = _weights(events, branch_map)
        objects = _build_objects(events, branch_map, config.cuts)
        source_name = Path(urlparse(str(path).removeprefix("simplecache::")).path).name or str(path)
        source_kind = _source_type(path)
        frame = _event_frame(events, branch_map, objects, weights, source_name, source_kind)
        file_candidates = []
        if objects["electrons"]:
            file_candidates.append(_pair_candidates(objects["electrons"], frame, "Z->ee", config.cuts))
        if objects["muons"]:
            file_candidates.append(_pair_candidates(objects["muons"], frame, "Z->mumu", config.cuts))
        if objects["electrons"] and objects["muons"]:
            file_candidates.append(_cross_candidates(objects["electrons"], objects["muons"], frame, config.cuts))
        candidates = pd.concat(file_candidates, ignore_index=True) if file_candidates else pd.DataFrame()
        frame = _add_event_flags(frame, candidates, config.cuts)
        all_frames.append(frame)
        if not candidates.empty:
            all_candidates.append(candidates)

        if objects["electrons"]:
            electron_pts.extend(ak.to_numpy(ak.flatten(objects["electrons"]["pt"], axis=None)))
            if objects["electrons"]["eta"] is not None:
                lepton_etas.extend(ak.to_numpy(ak.flatten(objects["electrons"]["eta"], axis=None)))
            z_ee.extend(_pair_masses(objects["electrons"]))
            z_ee_ss.extend(_pair_masses(objects["electrons"], sign="same"))
            if branch_map.met_et in events.fields and branch_map.met_phi in events.fields:
                w_enu.extend(_w_mt(objects["electrons"], to_gev(events[branch_map.met_et]), events[branch_map.met_phi]))
        if objects["muons"]:
            muon_pts.extend(ak.to_numpy(ak.flatten(objects["muons"]["pt"], axis=None)))
            if objects["muons"]["eta"] is not None:
                lepton_etas.extend(ak.to_numpy(ak.flatten(objects["muons"]["eta"], axis=None)))
            z_mumu.extend(_pair_masses(objects["muons"]))
            z_mumu_ss.extend(_pair_masses(objects["muons"], sign="same"))
            if branch_map.met_et in events.fields and branch_map.met_phi in events.fields:
                w_munu.extend(_w_mt(objects["muons"], to_gev(events[branch_map.met_et]), events[branch_map.met_phi]))
        if branch_map.met_et in events.fields:
            met_values.extend(ak.to_numpy(to_gev(events[branch_map.met_et])))
        if objects["jets"]:
            jet_mult.extend(ak.to_numpy(ak.num(objects["jets"]["pt"], axis=1)))
            lead_jet_pts.extend(ak.to_numpy(leading_or_nan(objects["jets"]["pt"])))
            if objects["jets"]["btag"] is not None:
                btag_mult.extend(count_true(objects["jets"]["btag"] != 0))

    all_events = pd.concat(all_frames, ignore_index=True)
    z_candidates = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame(
        columns=["source_file", "event_number", "channel", "mass_gev", "leading_lepton_pt_gev", "subleading_lepton_pt_gev", "opposite_sign", "in_z_window"]
    )
    selected_mask = _channel_mask(all_events, config.channel, config.cuts, config.apply_triggers)
    if config.channel == "z_ee":
        selected_mask = selected_mask & all_events["z_ee_in_window"]
    elif config.channel == "z_mumu":
        selected_mask = selected_mask & all_events["z_mumu_in_window"]
    selected = all_events[selected_mask].copy()
    cutflow = _cutflow(all_events, config.channel, config.cuts, config.apply_triggers)
    w_enu_selected = all_events.loc[_channel_mask(all_events, "w_enu", config.cuts, config.apply_triggers), "electron_mt_gev"].to_numpy()
    w_munu_selected = all_events.loc[_channel_mask(all_events, "w_munu", config.cuts, config.apply_triggers), "muon_mt_gev"].to_numpy()

    summary = pd.DataFrame(
        [
            {"metric": "files", "value": len(files)},
            {"metric": "trees", "value": ", ".join(sorted(tree_names))},
            {"metric": "events_selected", "value": len(selected)},
            {"metric": "sum_weights_selected", "value": float(selected["weight"].sum()) if len(selected) else 0.0},
            {"metric": "apply_triggers", "value": config.apply_triggers},
            {"metric": "z_mass_window_gev", "value": f"{config.cuts.z_mass_min}-{config.cuts.z_mass_max}"},
            {"metric": "w_mt_min_gev", "value": config.cuts.w_mt_min},
            {"metric": "source_types", "value": ", ".join(sorted(all_events["source_type"].dropna().unique()))},
            {"metric": "weight_mode", "value": "event weights if branch found, otherwise raw unit weights"},
            {"metric": "branch_map", "value": str(branch_map.present() if branch_map else {})},
        ]
    )
    lepton_summary = pd.DataFrame(
        [
            {"object": "electron", "count": len(electron_pts), "mean_pt_gev": np.nanmean(electron_pts) if electron_pts else np.nan},
            {"object": "muon", "count": len(muon_pts), "mean_pt_gev": np.nanmean(muon_pts) if muon_pts else np.nan},
        ]
    )
    jet_summary = pd.DataFrame(
        [{"count_events_with_jets": int(np.sum(np.asarray(jet_mult) > 0)) if jet_mult else 0, "mean_n_jets": np.nanmean(jet_mult) if jet_mult else np.nan}]
    )

    summary.to_csv(output / "summary.csv", index=False)
    lepton_summary.to_csv(output / "lepton_summary.csv", index=False)
    cutflow.to_csv(output / "cutflow.csv", index=False)
    z_candidates[z_candidates["channel"].isin(["Z->ee", "Z->mumu"])].to_csv(output / "z_candidates.csv", index=False)
    _lepton_label_counts(all_events).to_csv(output / "lepton_label_counts.csv", index=False)
    z_candidates[z_candidates["channel"] == "e-mu"].to_csv(output / "emu_control_candidates.csv", index=False)
    (output / "input_files.txt").write_text("\n".join(files) + "\n", encoding="utf-8")
    (output / "branch_map.json").write_text(json.dumps(branch_map.present() if branch_map else {}, indent=2), encoding="utf-8")
    run_config = asdict(config)
    run_config["cuts"] = asdict(config.cuts)
    (output / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    pd.concat(
        [
            histogram_table(z_ee, bins=80, range=(50, 130)).assign(channel="Z->ee"),
            histogram_table(z_mumu, bins=80, range=(50, 130)).assign(channel="Z->mumu"),
        ],
        ignore_index=True,
    ).to_csv(output / "z_mass_spectrum.csv", index=False)
    pd.concat(
        [
            histogram_table(z_ee, bins=80, range=(50, 130)).assign(channel="Z->ee", sign="opposite"),
            histogram_table(z_ee_ss, bins=80, range=(50, 130)).assign(channel="Z->ee", sign="same"),
            histogram_table(z_mumu, bins=80, range=(50, 130)).assign(channel="Z->mumu", sign="opposite"),
            histogram_table(z_mumu_ss, bins=80, range=(50, 130)).assign(channel="Z->mumu", sign="same"),
        ],
        ignore_index=True,
    ).to_csv(output / "os_ss_mass_spectrum.csv", index=False)
    if not z_candidates.empty:
        emu = z_candidates[z_candidates["channel"] == "e-mu"]
        emu_os = emu.loc[emu["opposite_sign"], "mass_gev"].to_numpy()
        emu_ss = emu.loc[emu["same_sign"], "mass_gev"].to_numpy()
    pd.concat(
        [
            histogram_table(emu_os, bins=80, range=(50, 130)).assign(channel="e-mu", sign="opposite"),
            histogram_table(emu_ss, bins=80, range=(50, 130)).assign(channel="e-mu", sign="same"),
        ],
        ignore_index=True,
    ).to_csv(output / "emu_mass_spectrum.csv", index=False)
    pd.concat(
        [
            histogram_table(w_enu, bins=60, range=(0, 180)).assign(channel="W->enu"),
            histogram_table(w_munu, bins=60, range=(0, 180)).assign(channel="W->munu"),
        ],
        ignore_index=True,
    ).to_csv(output / "w_transverse_mass.csv", index=False)
    pd.concat(
        [
            histogram_table(w_enu_selected, bins=60, range=(0, 180)).assign(channel="W->enu"),
            histogram_table(w_munu_selected, bins=60, range=(0, 180)).assign(channel="W->munu"),
        ],
        ignore_index=True,
    ).to_csv(output / "w_selected_transverse_mass.csv", index=False)
    jet_summary.to_csv(output / "jet_summary.csv", index=False)
    selected.to_csv(output / "selected_events.csv", index=False)

    if config.make_plots:
        plots = output / "plots"
        save_histogram(electron_pts, plots / "electron_pt.png", "Electron pT", "pT [GeV]", bins=60, range=(0, 250))
        save_histogram(muon_pts, plots / "muon_pt.png", "Muon pT", "pT [GeV]", bins=60, range=(0, 250))
        save_histogram(lepton_etas, plots / "lepton_eta.png", "Selected lepton eta", "eta", bins=50, range=(-3, 3))
        save_histogram(np.concatenate([np.asarray(z_ee), np.asarray(z_mumu)]), plots / "dilepton_mass.png", "Dilepton invariant mass with Z peak", "m_ll [GeV]", bins=80, range=(50, 130))
        save_histogram(z_ee, plots / "z_ee_dilepton_mass.png", "Z -> ee invariant mass", "m_ee [GeV]", bins=80, range=(50, 130))
        save_histogram(z_mumu, plots / "z_mumu_dilepton_mass.png", "Z -> mumu invariant mass", "m_mumu [GeV]", bins=80, range=(50, 130))
        save_overlay_histogram({"opposite sign": z_mumu, "same sign": z_mumu_ss}, plots / "z_mumu_os_ss_mass.png", "Z -> mumu OS vs SS mass", "m_mumu [GeV]", bins=80, range=(50, 130))
        save_overlay_histogram({"opposite sign": z_ee, "same sign": z_ee_ss}, plots / "z_ee_os_ss_mass.png", "Z -> ee OS vs SS mass", "m_ee [GeV]", bins=80, range=(50, 130))
        save_overlay_histogram({"opposite sign": emu_os, "same sign": emu_ss}, plots / "emu_control_mass.png", "e-mu control dilepton mass", "m_emu [GeV]", bins=80, range=(50, 130))
        save_histogram(np.concatenate([np.asarray(w_enu), np.asarray(w_munu)]), plots / "w_transverse_mass.png", "W transverse mass", "mT [GeV]", bins=60, range=(0, 180))
        save_histogram(w_enu_selected, plots / "w_enu_transverse_mass.png", "W -> e nu transverse mass", "mT [GeV]", bins=60, range=(0, 180))
        save_histogram(w_munu_selected, plots / "w_munu_transverse_mass.png", "W -> mu nu transverse mass", "mT [GeV]", bins=60, range=(0, 180))
        save_histogram(met_values, plots / "met.png", "Missing transverse energy", "MET [GeV]", bins=60, range=(0, 250))
        save_histogram(jet_mult, plots / "jet_multiplicity.png", "Jet multiplicity", "N jets", bins=np.arange(-0.5, 12.5, 1))
        save_histogram(lead_jet_pts, plots / "leading_jet_pt.png", "Leading jet pT", "pT [GeV]", bins=60, range=(0, 400))
        if btag_mult:
            save_histogram(btag_mult, plots / "btag_jet_multiplicity.png", "b-tagged jet multiplicity", "N b-tagged jets", bins=np.arange(-0.5, 8.5, 1))

    _write_validation_report(output, config, files, branch_map or BranchMap(), summary, cutflow, z_candidates, all_events)

    return AnalysisResult(output, files, branch_map or BranchMap(), summary, selected, cutflow)


def _write_validation_report(
    output: Path,
    config: AnalysisConfig,
    files: list[str],
    branch_map: BranchMap,
    summary: pd.DataFrame,
    cutflow: pd.DataFrame,
    z_candidates: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    selected_value = summary.loc[summary["metric"] == "events_selected", "value"].iloc[0] if not summary.empty else 0
    lines = [
        "# ATLAS Open Data Validation Report",
        "",
        f"- Channel: `{config.channel}`",
        f"- Files analyzed: `{len(files)}`",
        f"- Max events per file: `{config.max_events}`",
        f"- Source type(s): `{', '.join(sorted(events['source_type'].dropna().unique())) if 'source_type' in events else 'unknown'}`",
        f"- Trigger cuts applied: `{config.apply_triggers}`",
        f"- Electron trigger branch: `{branch_map.electron_trigger or 'missing'}`",
        f"- Muon trigger branch: `{branch_map.muon_trigger or 'missing'}`",
        f"- Selected events: `{selected_value}`",
        f"- Z mass window: `{config.cuts.z_mass_min}-{config.cuts.z_mass_max} GeV`",
        f"- W transverse-mass threshold: `{config.cuts.w_mt_min} GeV`",
        "",
        "## Scientific Checks",
        "",
    ]
    if config.channel in {"z_ee", "z_mumu"}:
        label = "Z->ee" if config.channel == "z_ee" else "Z->mumu"
        cand = z_candidates[(z_candidates["channel"] == label) & (z_candidates["opposite_sign"])]
        if not cand.empty:
            peak_window = cand[cand["in_z_window"]]
            mass_source = peak_window if not peak_window.empty else cand
            lines.extend(
                [
                    f"- Opposite-sign {label} candidates: `{len(cand)}`",
                    f"- Candidates inside Z window: `{len(peak_window)}`",
                    f"- Median in-window candidate mass: `{mass_source['mass_gev'].median():.3f} GeV`",
                    f"- Mean in-window candidate mass: `{mass_source['mass_gev'].mean():.3f} GeV`",
                    "- Interpretation: a peak near 91.2 GeV is expected for a valid Z-boson reconstruction.",
                ]
            )
    if config.channel in {"w_enu", "w_munu"}:
        lines.extend(
            [
                "- W selection requires exactly one selected lepton, vetoes the second lepton flavor, requires MET, and requires transverse mass.",
                "- For best W interpretation, use the ATLAS `1lep` skim rather than `exactly2lep`.",
            ]
        )
    if not config.apply_triggers:
        lines.append("- Trigger caveat: trigger branches were found if listed above, but trigger cuts were not applied in this run.")
    if "data" in set(events.get("source_type", pd.Series(dtype=str))):
        lines.append("- Weighting: input appears data-like; unit event weights are appropriate for raw Open Data plots.")
    lines.extend(
        [
            "",
            "## Cutflow",
            "",
            "```text",
            cutflow.to_string(index=False),
            "```",
            "",
            "## Scope",
            "",
            "This is an educational Open Data validation, not a publication-quality ATLAS result. It does not include full detector calibrations, systematic uncertainties, luminosity normalization, or collaboration-approved background modeling.",
            "",
        ]
    )
    output.joinpath("validation_report.md").write_text("\n".join(lines), encoding="utf-8")
