"""Physics helpers for starter ATLAS Open Data analyses."""

from __future__ import annotations

import math
from dataclasses import dataclass

import awkward as ak
import numpy as np


GEV = 1000.0


@dataclass
class ObjectCuts:
    electron_pt: float = 25.0
    electron_abs_eta: float = 2.47
    muon_pt: float = 25.0
    muon_abs_eta: float = 2.5
    jet_pt: float = 30.0
    jet_abs_eta: float = 2.5
    met: float = 25.0
    w_mt_min: float = 30.0
    max_iso_ratio: float | None = 0.15
    z_mass_min: float = 80.0
    z_mass_max: float = 100.0


def to_gev(values):
    array = ak.Array(values)
    flat = ak.flatten(array, axis=None)
    if len(flat) == 0:
        return array
    median = float(np.nanmedian(ak.to_numpy(flat)))
    return array / GEV if median > 500.0 else array


def safe_array(events: dict, branch: str | None, default=None):
    if branch is None:
        return default
    return events.get(branch, default)


def select_objects(pt, eta=None, quality=None, iso=None, cut_pt=25.0, cut_abs_eta=2.5, max_iso_ratio=0.15):
    mask = to_gev(pt) >= cut_pt
    if eta is not None:
        mask = mask & (abs(eta) <= cut_abs_eta)
    if quality is not None:
        mask = mask & (quality != 0)
    if iso is not None and max_iso_ratio is not None:
        ratio = to_gev(iso) / ak.where(to_gev(pt) > 0, to_gev(pt), math.inf)
        mask = mask & (ratio <= max_iso_ratio)
    return mask


def invariant_mass(pt1, eta1, phi1, pt2, eta2, phi2, e1=None, e2=None):
    pt1 = to_gev(pt1)
    pt2 = to_gev(pt2)
    if e1 is None or e2 is None:
        mass2 = 2.0 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2))
        return np.sqrt(np.maximum(mass2, 0.0))
    e1 = to_gev(e1)
    e2 = to_gev(e2)
    px1, py1, pz1 = pt1 * np.cos(phi1), pt1 * np.sin(phi1), pt1 * np.sinh(eta1)
    px2, py2, pz2 = pt2 * np.cos(phi2), pt2 * np.sin(phi2), pt2 * np.sinh(eta2)
    mass2 = (e1 + e2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2
    return np.sqrt(np.maximum(mass2, 0.0))


def transverse_mass(lepton_pt, lepton_phi, met_et, met_phi):
    lepton_pt = to_gev(lepton_pt)
    met_et = to_gev(met_et)
    return np.sqrt(np.maximum(2.0 * lepton_pt * met_et * (1.0 - np.cos(lepton_phi - met_phi)), 0.0))


def leading_or_nan(jagged):
    return ak.fill_none(ak.firsts(jagged), np.nan)


def count_true(jagged_bool):
    return ak.to_numpy(ak.sum(jagged_bool, axis=1))
