"""ROOT file inspection and flexible ATLAS ntuple branch discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import uproot
from uproot.behaviors.TBranch import HasBranches


@dataclass
class BranchMap:
    """Resolved branch names for physics objects.

    Values are actual branch names from a ROOT tree. Missing optional quantities
    are left as ``None`` and handled downstream with graceful fallbacks.
    """

    event_number: str | None = None
    run_number: str | None = None
    weight: str | None = None
    trigger: str | None = None
    electron_trigger: str | None = None
    muon_trigger: str | None = None
    electron_pt: str | None = None
    electron_eta: str | None = None
    electron_phi: str | None = None
    electron_e: str | None = None
    electron_charge: str | None = None
    electron_iso: str | None = None
    electron_quality: str | None = None
    muon_pt: str | None = None
    muon_eta: str | None = None
    muon_phi: str | None = None
    muon_e: str | None = None
    muon_charge: str | None = None
    muon_iso: str | None = None
    muon_quality: str | None = None
    jet_pt: str | None = None
    jet_eta: str | None = None
    jet_phi: str | None = None
    jet_e: str | None = None
    jet_btag: str | None = None
    met_et: str | None = None
    met_phi: str | None = None
    lepton_type: str | None = None

    def present(self) -> dict[str, str]:
        return {key: value for key, value in self.__dict__.items() if value}

    def required_for(self, channel: str) -> list[str]:
        base = {
            "inclusive": [],
            "z_ee": ["electron_pt", "electron_eta", "electron_phi"],
            "z_mumu": ["muon_pt", "muon_eta", "muon_phi"],
            "w_enu": ["electron_pt", "electron_phi", "met_et", "met_phi"],
            "w_munu": ["muon_pt", "muon_phi", "met_et", "met_phi"],
            "jets": ["jet_pt"],
            "top": ["jet_pt", "met_et"],
        }
        return [name for name in base.get(channel, []) if getattr(self, name) is None]


@dataclass
class TreeInfo:
    file: str
    tree_name: str
    branches: list[str] = field(default_factory=list)
    entries: int | None = None


BRANCH_CANDIDATES: dict[str, tuple[str, ...]] = {
    "event_number": ("eventNumber", "event_number", "EventNumber", "event"),
    "run_number": ("runNumber", "run_number", "RunNumber", "run"),
    "weight": ("mcWeight", "eventWeight", "weight", "scaleFactor", "totalWeight"),
    "trigger": ("passedTrigger", "event_trigger"),
    "electron_trigger": ("trigE", "trigger_e", "electron_trigger", "HLT_e"),
    "muon_trigger": ("trigM", "trigger_m", "muon_trigger", "HLT_mu"),
    "electron_pt": ("lep_pt", "el_pt", "electron_pt", "Electron_pt", "e_pt"),
    "electron_eta": ("lep_eta", "el_eta", "electron_eta", "Electron_eta", "e_eta"),
    "electron_phi": ("lep_phi", "el_phi", "electron_phi", "Electron_phi", "e_phi"),
    "electron_e": ("lep_E", "lep_e", "el_E", "el_e", "electron_E", "electron_e", "Electron_e"),
    "electron_charge": ("lep_charge", "el_charge", "electron_charge", "Electron_charge"),
    "electron_iso": ("lep_etcone20", "el_etcone20", "el_iso", "electron_iso", "Electron_pfRelIso04_all"),
    "electron_quality": ("lep_isTightID", "el_tight", "el_isTightID", "electron_tight", "Electron_tight"),
    "muon_pt": ("mu_pt", "muon_pt", "Muon_pt", "m_pt", "lep_pt"),
    "muon_eta": ("mu_eta", "muon_eta", "Muon_eta", "m_eta", "lep_eta"),
    "muon_phi": ("mu_phi", "muon_phi", "Muon_phi", "m_phi", "lep_phi"),
    "muon_e": ("mu_E", "mu_e", "muon_E", "muon_e", "Muon_e", "lep_E", "lep_e"),
    "muon_charge": ("mu_charge", "muon_charge", "Muon_charge", "lep_charge"),
    "muon_iso": ("mu_etcone20", "mu_iso", "muon_iso", "Muon_pfRelIso04_all", "lep_etcone20"),
    "muon_quality": ("mu_isTightID", "mu_tight", "muon_tight", "Muon_tight", "lep_isTightID"),
    "jet_pt": ("jet_pt", "Jet_pt", "jets_pt"),
    "jet_eta": ("jet_eta", "Jet_eta", "jets_eta"),
    "jet_phi": ("jet_phi", "Jet_phi", "jets_phi"),
    "jet_e": ("jet_E", "jet_e", "Jet_e", "jets_e"),
    "jet_btag": ("jet_MV2c10", "jet_btag", "jet_btagged", "Jet_btag", "btag"),
    "met_et": ("met_et", "met_Et", "MET_et", "met", "MissingET_MET"),
    "met_phi": ("met_phi", "MET_phi", "MissingET_phi"),
    "lepton_type": ("lep_type", "lepton_type", "Lepton_type"),
}


def is_remote_path(path: str | Path) -> bool:
    text = str(path)
    if text.startswith("simplecache::"):
        text = text.removeprefix("simplecache::")
    return bool(urlparse(text).scheme in {"root", "http", "https", "s3", "gs"})


def atlas_access_paths(path: str | Path) -> list[str]:
    """Return primary and fallback access paths for known ATLAS Open Data mirrors."""

    text = str(path)
    prefix = "simplecache::" if text.startswith("simplecache::") else ""
    bare = text.removeprefix("simplecache::")
    paths = [text]
    if prefix:
        paths.append(bare)
    marker = "https://opendata.cern.ch/eos/opendata/atlas/OutreachDatasets/2020-08-19/"
    if bare.startswith(marker):
        suffix = bare.removeprefix(marker)
        fallback = f"{prefix}https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/{suffix}"
        direct_fallback = f"https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/{suffix}"
        paths.append(fallback)
        paths.append(direct_fallback)
        record_url = _record_file_url(prefix, suffix)
        if record_url:
            paths.append(record_url)
        direct_record_url = _record_file_url("", suffix)
        if direct_record_url:
            paths.append(direct_record_url)
    legacy_marker = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/"
    if bare.startswith(legacy_marker):
        suffix = bare.removeprefix(legacy_marker)
        record_url = _record_file_url(prefix, suffix)
        if record_url:
            paths.append(record_url)
        direct_record_url = _record_file_url("", suffix)
        if direct_record_url:
            paths.append(direct_record_url)
    return list(dict.fromkeys(paths))


def _record_file_url(prefix: str, suffix: str) -> str | None:
    parts = suffix.split("/")
    if len(parts) < 3:
        return None
    skim, filename = parts[0], parts[-1]
    record_by_skim = {
        "1largeRjet1lep": "15000",
        "1lep": "15001",
        "1lep1tau": "15002",
        "2lep": "15003",
        "3lep": "15004",
        "4lep": "15005",
        "GamGam": "15006",
        "exactly2lep": "15007",
    }
    record = record_by_skim.get(skim)
    if not record:
        return None
    return f"{prefix}https://opendata.cern.ch/record/{record}/files/{filename}?download=1"


def list_root_files(folder: str | Path, pattern: str = "*.root", max_files: int | None = None) -> list[str]:
    files = [str(path) for path in sorted(Path(folder).expanduser().glob(pattern))]
    if max_files is not None:
        files = files[: max(0, max_files)]
    return files


def inspect_root_file(path: str | Path) -> list[TreeInfo]:
    """Return tree names, entry counts, and branch names for a ROOT file."""

    original_path = str(path)
    local_path = Path(path)
    if not is_remote_path(original_path) and not local_path.exists():
        raise FileNotFoundError(
            f"ROOT file not found: {path}\n"
            "Place a real ATLAS Open Data .root file there, or pass the path to an existing file."
        )
    if not is_remote_path(original_path) and not local_path.is_file():
        raise FileNotFoundError(f"Expected a ROOT file path, but got a directory: {path}")

    infos: list[TreeInfo] = []
    last_error: Exception | None = None
    tried = atlas_access_paths(original_path)
    for access_path in tried:
        try:
            root_file = uproot.open(access_path)
            break
        except Exception as exc:
            last_error = exc
    else:
        attempts = "\n".join(f"- {item}" for item in tried)
        raise OSError(f"Could not open ROOT file after trying:\n{attempts}\nLast error: {last_error}") from last_error

    with root_file:
        for key, obj in root_file.items():
            is_tree_like = isinstance(obj, HasBranches) or (hasattr(obj, "keys") and hasattr(obj, "num_entries"))
            if not is_tree_like:
                continue
            tree_name = key.split(";")[0]
            tree = root_file[tree_name]
            infos.append(
                TreeInfo(
                    file=original_path,
                    tree_name=tree_name,
                    branches=list(tree.keys()),
                    entries=tree.num_entries,
                )
            )
    return infos


def choose_tree(path: str | Path, requested_tree: str | None = None) -> TreeInfo:
    infos = inspect_root_file(path)
    if not infos:
        raise ValueError(f"No TTree objects were found in ROOT file: {path}")
    if requested_tree:
        for info in infos:
            if info.tree_name == requested_tree:
                return info
        available = ", ".join(info.tree_name for info in infos)
        raise ValueError(f"Tree '{requested_tree}' not found in {path}. Available trees: {available}")

    def score(info: TreeInfo) -> int:
        branches = {branch.lower() for branch in info.branches}
        keywords = ("lep", "el_", "mu_", "jet", "met", "weight", "trig")
        return sum(any(keyword in branch for branch in branches) for keyword in keywords)

    return max(infos, key=score)


def resolve_branches(branches: Iterable[str]) -> BranchMap:
    available = list(branches)
    lower_lookup = {branch.lower(): branch for branch in available}
    resolved: dict[str, str | None] = {}
    for logical_name, candidates in BRANCH_CANDIDATES.items():
        exact = next((candidate for candidate in candidates if candidate in available), None)
        if exact:
            resolved[logical_name] = exact
            continue
        fuzzy = next((lower_lookup[key] for key in lower_lookup if any(c.lower() == key for c in candidates)), None)
        if fuzzy:
            resolved[logical_name] = fuzzy
            continue
        contains = next(
            (
                branch
                for branch in available
                for candidate in candidates
                if candidate.lower() in branch.lower()
            ),
            None,
        )
        resolved[logical_name] = contains
    return BranchMap(**resolved)


def describe_inspection(path: str | Path) -> str:
    lines = [f"ROOT inspection for {path}"]
    for info in inspect_root_file(path):
        lines.append(f"- {info.tree_name}: {info.entries} entries, {len(info.branches)} branches")
        preview = ", ".join(info.branches[:30])
        lines.append(f"  branches: {preview}{' ...' if len(info.branches) > 30 else ''}")
    return "\n".join(lines)
