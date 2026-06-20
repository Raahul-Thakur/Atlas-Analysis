"""Optional ATLAS Open Magic integration for dataset URL discovery."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from .schema import atlas_access_paths

def _load_magic():
    try:
        import atlasopenmagic as atom
    except ImportError as exc:
        raise ImportError(
            "atlasopenmagic is not installed. Install it with:\n"
            "python -m pip install atlasopenmagic\n"
            "or reinstall this project's requirements.txt."
        ) from exc
    return atom


def available_releases() -> dict:
    atom = _load_magic()
    return atom.available_releases()


def available_skims(release: str = "2024r-pp") -> list[str]:
    atom = _load_magic()
    atom.set_verbosity("warning")
    atom.set_release(release)
    return atom.available_skims()


def available_datasets(release: str = "2024r-pp") -> list[str]:
    atom = _load_magic()
    atom.set_verbosity("warning")
    atom.set_release(release)
    return atom.available_datasets()


def get_urls(
    release: str = "2024r-pp",
    dataset: str = "data",
    skim: str = "noskim",
    protocol: str = "https",
    cache: bool | None = None,
    limit: int | None = None,
) -> list[str]:
    atom = _load_magic()
    atom.set_verbosity("warning")
    atom.set_release(release)
    urls = atom.get_urls(dataset, skim=skim, protocol=protocol, cache=cache)
    if release == "2020e-13tev" and protocol == "https":
        marker = "https://opendata.cern.ch/eos/opendata/atlas/OutreachDatasets/2020-08-19/"
        urls = [
            url.replace(marker, "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/")
            if url.removeprefix("simplecache::").startswith(marker)
            else url
            for url in urls
        ]
    return urls[:limit] if limit else urls


def write_url_file(urls: list[str], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(urls) + "\n", encoding="utf-8")
    return path


def read_url_file(path: str | Path, max_files: int | None = None) -> list[str]:
    urls = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return urls[:max_files] if max_files else urls


def download_urls(urls: list[str], output_folder: str | Path = "data", limit: int | None = None, resume: bool = True) -> list[Path]:
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for url in urls[:limit] if limit else urls:
        downloaded.append(download_url(url, output, resume=resume))
    return downloaded


def download_url(url: str, output_folder: str | Path = "data", resume: bool = True) -> Path:
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    candidates = [candidate.removeprefix("simplecache::") for candidate in atlas_access_paths(url)]
    for candidate in candidates:
        filename = Path(urlparse(candidate).path).name
        if not filename:
            continue
        target = output / filename
        try:
            _stream_download(candidate, target, resume=resume)
            return target
        except Exception as exc:
            last_error = exc
    attempts = "\n".join(f"- {candidate}" for candidate in candidates)
    raise OSError(f"Could not download ROOT file after trying:\n{attempts}\nLast error: {last_error}") from last_error


def _stream_download(url: str, target: Path, resume: bool = True) -> None:
    expected_size = _remote_size(url)
    if expected_size is not None and target.exists() and target.stat().st_size >= expected_size:
        return
    headers = {}
    existing = target.stat().st_size if target.exists() else 0
    if resume and existing > 0:
        headers["Range"] = f"bytes={existing}-"
    with requests.get(url, stream=True, timeout=60, headers=headers, allow_redirects=True) as response:
        if response.status_code == 416:
            return
        if response.status_code not in {200, 206}:
            raise OSError(f"HTTP {response.status_code} for {url}")
        if existing and response.status_code == 200:
            existing = 0
            target.unlink(missing_ok=True)
        total = response.headers.get("content-length")
        total_size = expected_size or (int(total) + existing if total and total.isdigit() else None)
        mode = "ab" if existing and response.status_code == 206 else "wb"
        try:
            with target.open(mode) as handle:
                with tqdm(
                    total=total_size,
                    initial=existing,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {target.name}",
                ) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        progress.update(len(chunk))
        except requests.exceptions.ChunkedEncodingError as exc:
            raise OSError(f"Connection dropped while downloading {url}; rerun the same command to resume.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise OSError(f"Connection dropped while downloading {url}; rerun the same command to resume.") from exc
    if expected_size is not None and target.stat().st_size < expected_size:
        raise OSError(
            f"Incomplete download for {target.name}: got {target.stat().st_size} bytes, expected {expected_size}. "
            "Rerun the same command to resume."
        )


def _remote_size(url: str) -> int | None:
    try:
        response = requests.head(url, timeout=30, allow_redirects=True)
        if response.status_code >= 400:
            return None
        value = response.headers.get("content-length")
        return int(value) if value and value.isdigit() else None
    except requests.RequestException:
        return None
