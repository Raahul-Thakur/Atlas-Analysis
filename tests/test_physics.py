import awkward as ak
import numpy as np

from atlas_analysis.physics import ObjectCuts, invariant_mass, transverse_mass
from atlas_analysis.pipeline import _build_objects
from atlas_analysis.schema import BranchMap


def test_invariant_mass_back_to_back_massless_leptons():
    mass = invariant_mass(
        np.array([45.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([45.0]),
        np.array([0.0]),
        np.array([np.pi]),
    )
    assert np.isclose(mass[0], 90.0)


def test_transverse_mass_known_value():
    mt = transverse_mass(
        np.array([40.0]),
        np.array([0.0]),
        np.array([40.0]),
        np.array([np.pi / 2.0]),
    )
    assert np.isclose(mt[0], np.sqrt(3200.0))


def test_shared_lep_type_branches_split_electrons_and_muons():
    events = ak.Array(
        {
            "lep_pt": [[45_000.0, 52_000.0, 18_000.0]],
            "lep_eta": [[0.1, -0.2, 0.3]],
            "lep_phi": [[0.0, 1.0, 2.0]],
            "lep_E": [[46_000.0, 53_000.0, 19_000.0]],
            "lep_charge": [[-1, 1, -1]],
            "lep_type": [[11, 13, 13]],
            "lep_isTightID": [[1, 1, 1]],
        }
    )
    branch_map = BranchMap(
        electron_pt="lep_pt",
        electron_eta="lep_eta",
        electron_phi="lep_phi",
        electron_e="lep_E",
        electron_charge="lep_charge",
        electron_quality="lep_isTightID",
        muon_pt="lep_pt",
        muon_eta="lep_eta",
        muon_phi="lep_phi",
        muon_e="lep_E",
        muon_charge="lep_charge",
        muon_quality="lep_isTightID",
        lepton_type="lep_type",
    )
    objects = _build_objects(events, branch_map, ObjectCuts(muon_pt=10.0))
    assert ak.to_list(objects["electrons"]["pt"]) == [[45.0]]
    assert ak.to_list(objects["muons"]["pt"]) == [[52.0, 18.0]]
