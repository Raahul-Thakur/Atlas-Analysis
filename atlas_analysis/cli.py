"""Command line interface for the ATLAS Open Data starter analysis."""

from __future__ import annotations

import argparse
import sys

from .data_access import available_datasets, available_releases, available_skims, download_urls, get_urls, read_url_file, write_url_file
from .physics import ObjectCuts
from .pipeline import AnalysisConfig, run_analysis
from .schema import describe_inspection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reusable ATLAS Open Data ROOT analysis.")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Inspect trees and branches in a ROOT file.")
    inspect.add_argument("root_file")

    magic = sub.add_parser("magic", help="Use atlasopenmagic to find ATLAS Open Data URLs.")
    magic_sub = magic.add_subparsers(dest="magic_command", required=True)
    magic_sub.add_parser("releases", help="List available atlasopenmagic releases.")
    skims = magic_sub.add_parser("skims", help="List skims for a release.")
    skims.add_argument("--release", default="2024r-pp")
    datasets = magic_sub.add_parser("datasets", help="List dataset keys for a release.")
    datasets.add_argument("--release", default="2024r-pp")
    urls = magic_sub.add_parser("urls", help="Write dataset file URLs to a text file.")
    urls.add_argument("--release", default="2024r-pp")
    urls.add_argument("--dataset", default="data")
    urls.add_argument("--skim", default="noskim")
    urls.add_argument("--protocol", choices=["https", "root", "eos"], default="https")
    urls.add_argument("--no-cache", action="store_true", help="Do not prefix URLs with simplecache::.")
    urls.add_argument("--limit", type=int)
    urls.add_argument("--output", default="data/atlas_urls.txt")

    download = sub.add_parser("download", help="Download ROOT files from a URL file for reliable local analysis.")
    download.add_argument("--url-file", required=True)
    download.add_argument("--output-folder", default="data")
    download.add_argument("--limit", type=int)
    download.add_argument("--no-resume", action="store_true")

    run = sub.add_parser("run", help="Run analysis over ROOT files.")
    run.add_argument("--input-folder", default="data")
    run.add_argument("--url-file", help="Text file containing local paths or remote ROOT URLs, one per line.")
    run.add_argument("--pattern", default="*.root")
    run.add_argument("--tree-name")
    run.add_argument("--output-folder", default="outputs")
    run.add_argument("--max-files", type=int)
    run.add_argument("--max-events", type=int)
    run.add_argument("--channel", choices=["inclusive", "z_ee", "z_mumu", "w_enu", "w_munu", "jets", "top"], default="inclusive")
    run.add_argument("--electron-pt", type=float, default=25.0)
    run.add_argument("--muon-pt", type=float, default=25.0)
    run.add_argument("--jet-pt", type=float, default=30.0)
    run.add_argument("--met", type=float, default=25.0)
    run.add_argument("--w-mt-min", type=float, default=30.0)
    run.add_argument("--electron-eta", type=float, default=2.47)
    run.add_argument("--muon-eta", type=float, default=2.5)
    run.add_argument("--jet-eta", type=float, default=2.5)
    run.add_argument("--z-mass-min", type=float, default=80.0)
    run.add_argument("--z-mass-max", type=float, default=100.0)
    run.add_argument("--apply-triggers", action="store_true", help="Apply trigE/trigM channel trigger cuts when available.")
    run.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "inspect":
            print(describe_inspection(args.root_file))
            return
        if args.command == "magic":
            if args.magic_command == "releases":
                available_releases()
                return
            if args.magic_command == "skims":
                print("\n".join(available_skims(args.release)))
                return
            if args.magic_command == "datasets":
                print("\n".join(available_datasets(args.release)))
                return
            if args.magic_command == "urls":
                urls = get_urls(args.release, args.dataset, args.skim, args.protocol, cache=False if args.no_cache else None, limit=args.limit)
                path = write_url_file(urls, args.output)
                print(f"Wrote {len(urls)} URL(s) to {path}")
                return
        if args.command == "download":
            urls = read_url_file(args.url_file, args.limit)
            paths = download_urls(urls, args.output_folder, resume=not args.no_resume)
            print("Downloaded:")
            for path in paths:
                print(path)
            return

        cuts = ObjectCuts(
            electron_pt=args.electron_pt,
            electron_abs_eta=args.electron_eta,
            muon_pt=args.muon_pt,
            muon_abs_eta=args.muon_eta,
            jet_pt=args.jet_pt,
            jet_abs_eta=args.jet_eta,
            met=args.met,
            w_mt_min=args.w_mt_min,
            z_mass_min=args.z_mass_min,
            z_mass_max=args.z_mass_max,
        )
        input_files = read_url_file(args.url_file, args.max_files) if args.url_file else None
        result = run_analysis(
            AnalysisConfig(
                input_folder=args.input_folder,
                input_files=input_files,
                pattern=args.pattern,
                output_folder=args.output_folder,
                tree_name=args.tree_name,
                max_files=args.max_files,
                max_events=args.max_events,
                channel=args.channel,
                cuts=cuts,
                make_plots=not args.no_plots,
                apply_triggers=args.apply_triggers,
            )
        )
        print(f"Wrote analysis outputs to {result.output_folder}")
        print(f"Resolved branches: {result.branch_map.present()}")
    except (FileNotFoundError, ImportError, ModuleNotFoundError, OSError, ValueError) as exc:
        if "HTTPFileSystem requires" in str(exc):
            print(
                "Error: HTTPS streaming needs extra fsspec dependencies.\n"
                "Install them with:\n"
                "python -m pip install requests aiohttp\n"
                "or rerun:\n"
                "python -m pip install -r requirements.txt",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
