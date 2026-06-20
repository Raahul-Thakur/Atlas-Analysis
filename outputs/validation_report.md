# ATLAS Open Data Validation Report

- Channel: `w_munu`
- Files analyzed: `1`
- Max events per file: `10000`
- Source type(s): `data`
- Trigger cuts applied: `True`
- Electron trigger branch: `trigE`
- Muon trigger branch: `trigM`
- Selected events: `1693`
- Z mass window: `80.0-100.0 GeV`
- W transverse-mass threshold: `30.0 GeV`

## Scientific Checks

- W selection requires exactly one selected lepton, vetoes the second lepton flavor, requires MET, and requires transverse mass.
- For best W interpretation, use the ATLAS `1lep` skim rather than `exactly2lep`.
- Weighting: input appears data-like; unit event weights are appropriate for raw Open Data plots.

## Cutflow

```text
channel                step  events  weighted_events
 w_munu          all events   10000          10000.0
 w_munu      trigger passed    4517           4517.0
 w_munu  exactly one lepton    2563           2563.0
 w_munu  second lepton veto    2563           2563.0
 w_munu             MET cut    1757           1757.0
 w_munu transverse mass cut    1693           1693.0
 w_munu      final selected    1693           1693.0
```

## Scope

This is an educational Open Data validation, not a publication-quality ATLAS result. It does not include full detector calibrations, systematic uncertainties, luminosity normalization, or collaboration-approved background modeling.
