#!/usr/bin/env python3
"""Read-only repository snapshot exporter.

Concatenates all text source files in the repo into a single .txt file
for easy sharing with language models that lack file browsing.

SAFETY:
- This script ONLY reads files. It never modifies, moves, or deletes any source file.
- The only file it writes is the output snapshot (and a .tmp predecessor).
- It does not follow symlinks.
- It skips sensitive files, binary files, and large files.
"""

import argparse
import datetime
import glob as globmod
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_TEXT_EXTENSIONS = {
    ".py", ".pyi",
    ".md", ".rst", ".txt",
    ".yaml", ".yml",
    ".json", ".jsonl",
    ".toml", ".cfg", ".ini", ".conf",
    ".sh", ".bash", ".zsh",
    ".sql",
    ".tex", ".bib",
    ".csv", ".tsv",
    ".xml", ".html", ".css", ".js", ".ts",
    ".env.example",
    ".gitignore", ".gitattributes",
    ".dockerignore",
    ".editorconfig",
    ".flake8",
    ".pre-commit-config.yaml",
    "Makefile",
    "Dockerfile",
}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".claude",
    ".venv",
    "venv",
    "node_modules",
    "runs",
    "artifacts",
    "outputs",
    "videos",
    "checkpoints",
    "wandb",
    ".tox",
    ".eggs",
    "*.egg-info",
    "dist",
    "build",
    ".mypy_cache",
}

SKIP_DIR_PREFIXES = (
    "data/sim/episodes",
    "data/sim/videos",
    "data/real",
    "logs",
    "runs/debug/repo_snapshot",
)

SKIP_FILE_PATTERNS = {
    "repo_snapshot.txt",
    "repo_snapshot.tmp",
    ".env",
}

SKIP_FILE_PREFIXES = (
    ".env.",
    "credentials",
    "secret",
    "token",
)

SKIP_EXTENSIONS = {
    # secrets / certs
    ".pem", ".key", ".crt",
    # databases
    ".sqlite", ".db",
    # serialized objects
    ".pkl", ".npy", ".npz", ".pt", ".pth", ".ckpt",
    # video
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
    # images
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".ico", ".tiff",
    # archives
    ".zip", ".tar", ".gz", ".7z", ".rar", ".bz2", ".xz",
    # compiled
    ".so", ".dll", ".dylib", ".o", ".a",
    # fonts
    ".ttf", ".otf", ".woff", ".woff2",
    # pdf
    ".pdf",
}

DEFAULT_MAX_FILE_SIZE_KB = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_text_extension(path: Path) -> bool:
    """Check if the file has a known text extension."""
    name = path.name
    # Special filenames without extensions
    if name in KNOWN_TEXT_EXTENSIONS:
        return True
    ext = path.suffix.lower()
    return ext in KNOWN_TEXT_EXTENSIONS


def _should_skip_dir(dirname: str) -> bool:
    return dirname in SKIP_DIRS or dirname.endswith(".egg-info")


def _should_skip_file(path: Path) -> bool:
    name = path.name
    if name in SKIP_FILE_PATTERNS:
        return True
    for prefix in SKIP_FILE_PREFIXES:
        if name.startswith(prefix):
            return True
    ext = path.suffix.lower()
    if ext in SKIP_EXTENSIONS:
        return True
    return False


def _try_read_text(filepath: Path, max_bytes: int) -> tuple[bool, str, str]:
    """Try to read a file as UTF-8 text.

    Returns (success, content_or_reason, encoding_used).
    """
    try:
        size = filepath.stat().st_size
    except OSError as e:
        return False, f"stat error: {e}", ""

    if size > max_bytes:
        return False, f"too large ({size / 1024:.1f} KB > {max_bytes / 1024:.0f} KB)", ""

    try:
        content = filepath.read_text(encoding="utf-8")
        return True, content, "utf-8"
    except UnicodeDecodeError:
        return False, "not valid UTF-8", ""
    except OSError as e:
        return False, f"read error: {e}", ""


def _is_within_root(path: Path, root: Path) -> bool:
    """Check that path is inside root (prevents reading files outside repo)."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _format_size(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.1f} KB"
    else:
        return f"{n_bytes / (1024 * 1024):.1f} MB"


def resolve_include_patterns(
    root: Path,
    include_paths: list[str],
    include_globs: list[str],
    include_from: str | None,
    print_skipped: bool,
) -> tuple[set[Path], list[str]]:
    """Resolve --include / --include-glob / --include-from into a set of
    absolute file paths.

    Returns:
        resolved: set of absolute paths to include
        missing: list of original patterns/paths that matched nothing
    """
    resolved: set[Path] = set()
    missing: list[str] = []

    all_patterns: list[str] = list(include_paths)
    all_patterns.extend(include_globs)

    if include_from is not None:
        from_path = Path(include_from)
        if not from_path.is_file():
            print(f"WARNING: --include-from file not found: {include_from}", file=sys.stderr)
            missing.append(include_from)
        else:
            for line in from_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                all_patterns.append(line)

    for pattern in all_patterns:
        if any(c in pattern for c in ("*", "?", "[")):
            # Treat as glob
            matches = globmod.glob(str(root / pattern), recursive=True)
            matched_any = False
            for m in matches:
                mp = Path(m)
                if mp.is_file() and not mp.is_symlink():
                    resolved.add(mp.resolve())
                    matched_any = True
            if not matched_any:
                missing.append(pattern)
                if print_skipped:
                    print(f"  MISSING [glob]: {pattern}", file=sys.stderr)
        else:
            # Treat as file or directory
            p = root / pattern
            if p.is_dir():
                for dirpath, dirnames, filenames in os.walk(p, followlinks=False):
                    for fname in filenames:
                        fp = Path(dirpath) / fname
                        if fp.is_file() and not fp.is_symlink():
                            resolved.add(fp.resolve())
            elif p.is_file():
                resolved.add(p.resolve())
            else:
                missing.append(pattern)
                if print_skipped:
                    print(f"  MISSING [path]: {pattern}", file=sys.stderr)

    # Enforce root boundary: reject any resolved path outside repo root
    root_resolved = root.resolve()
    bounded: set[Path] = set()
    for rp in resolved:
        if _is_within_root(rp, root_resolved):
            bounded.add(rp)
        else:
            try:
                rel = rp.relative_to(root_resolved)
            except ValueError:
                rel = rp
            missing.append(str(rel))
            if print_skipped:
                print(f"  SKIP [outside repo root]: {rp}", file=sys.stderr)
    return bounded, missing


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def collect_files(
    root: Path,
    max_file_size_bytes: int,
    include_data_metadata: bool,
    include_runs_debug: bool,
    print_skipped: bool,
    skip_resolved_paths: set[Path] | None = None,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], dict[str, int]]:
    """Walk the repo and collect text files.

    Args:
        skip_resolved_paths: set of resolved absolute paths to always skip
            (used to exclude the output snapshot and its .tmp from inclusion).

    Returns:
        included: list of (relative_path, content)
        skipped: list of (relative_path, reason)
        skip_reason_counts: dict mapping reason -> count
    """
    included = []
    skipped = []
    skip_reason_counts: dict[str, int] = {}
    if skip_resolved_paths is None:
        skip_resolved_paths = set()

    def _record_skip(rel: Path, reason: str):
        skipped.append((rel, reason))
        skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
        if print_skipped:
            print(f"  SKIP [{reason}]: {rel}", file=sys.stderr)

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune directories in-place so os.walk doesn't descend into them
        rel_dir = Path(dirpath).relative_to(root)
        rel_dir_str = str(rel_dir)

        # Check if we're inside a skippable prefix path
        should_skip_prefix = False
        for prefix in SKIP_DIR_PREFIXES:
            if rel_dir_str.startswith(prefix) or rel_dir_str == prefix:
                should_skip_prefix = True
                break

        pruned = []
        for d in dirnames:
            if d == "runs" and include_runs_debug:
                # Allow walker to enter runs/ when --include-runs-debug is set;
                # subdirectories under runs/ will be pruned below
                pruned.append(d)
                continue
            if rel_dir_str == "runs" and include_runs_debug:
                # Under runs/, only allow debug/ — skip everything else
                if d == "debug":
                    pruned.append(d)
                elif print_skipped:
                    print(f"  SKIP DIR: {rel_dir / d} (runs/debug only)", file=sys.stderr)
                continue
            if _should_skip_dir(d):
                if print_skipped:
                    print(f"  SKIP DIR: {rel_dir / d}", file=sys.stderr)
                continue
            pruned.append(d)
        dirnames[:] = pruned

        if should_skip_prefix and rel_dir_str != ".":
            dirnames.clear()
            _record_skip(rel_dir, "skipped directory prefix")
            continue

        for fname in filenames:
            filepath = Path(dirpath) / fname
            rel_path = filepath.relative_to(root)

            # Skip symlinks
            if filepath.is_symlink():
                _record_skip(rel_path, "symlink")
                continue

            # Skip the output snapshot file and its .tmp to prevent recursion
            try:
                resolved = filepath.resolve()
            except OSError:
                resolved = filepath
            if resolved in skip_resolved_paths:
                _record_skip(rel_path, "output snapshot file (anti-recursion)")
                continue

            # Skip any .txt file with "snapshot" in its name (old snapshots, etc.)
            if filepath.suffix.lower() == ".txt" and "snapshot" in filepath.stem.lower():
                _record_skip(rel_path, "snapshot .txt file (anti-recursion)")
                continue

            # Skip by filename / extension
            if _should_skip_file(filepath):
                _record_skip(rel_path, "skipped file pattern/extension")
                continue

            # data/sim/metadata gate
            rel_str = str(rel_path)
            if rel_str.startswith("data/sim/metadata/") and not include_data_metadata:
                _record_skip(rel_path, "data/sim/metadata (use --include-data-metadata)")
                continue

            # Must be a known text extension (or try fallback)
            is_known_text = _is_text_extension(filepath)

            if not is_known_text:
                # For unknown extensions, try reading as UTF-8
                ok, content_or_reason, _ = _try_read_text(filepath, max_file_size_bytes)
                if ok:
                    included.append((rel_path, content_or_reason))
                else:
                    _record_skip(rel_path, f"unknown extension + {content_or_reason}")
                continue

            # Known text extension — try to read
            ok, content_or_reason, _ = _try_read_text(filepath, max_file_size_bytes)
            if ok:
                included.append((rel_path, content_or_reason))
            else:
                _record_skip(rel_path, content_or_reason)

    return included, skipped, skip_reason_counts


def _guess_lang(filepath: Path) -> str:
    """Guess a markdown language tag from file extension."""
    ext = filepath.suffix.lower()
    mapping = {
        ".py": "python", ".pyi": "python",
        ".md": "markdown", ".rst": "rst",
        ".yaml": "yaml", ".yml": "yaml",
        ".json": "json", ".jsonl": "json",
        ".toml": "toml",
        ".sh": "bash", ".bash": "bash", ".zsh": "zsh",
        ".sql": "sql",
        ".tex": "latex", ".bib": "bibtex",
        ".xml": "xml", ".html": "html",
        ".css": "css", ".js": "javascript", ".ts": "typescript",
        ".ini": "ini", ".cfg": "ini", ".conf": "ini",
        ".csv": "csv", ".tsv": "csv",
        ".txt": "",
        ".dockerignore": "", ".gitignore": "",
    }
    name = filepath.name
    if name == "Makefile":
        return "makefile"
    if name == "Dockerfile":
        return "dockerfile"
    return mapping.get(ext, "")


def build_snapshot(
    included: list[tuple[Path, str]],
    skipped: list[tuple[Path, str]],
    skip_reason_counts: dict[str, int],
    root: Path,
    max_file_size_kb: int,
    snapshot_mode: str = "full",
    snapshot_title: str | None = None,
    missing_files: list[str] | None = None,
    include_sources: list[str] | None = None,
) -> str:
    """Build the full snapshot text."""
    now = datetime.datetime.now().isoformat(timespec="seconds")
    total_chars = sum(len(c) for _, c in included)
    if missing_files is None:
        missing_files = []
    if include_sources is None:
        include_sources = []

    header_lines = [
        "=" * 72,
        "REPOSITORY SNAPSHOT",
        "=" * 72,
        f"generated_at: {now}",
        f"repo_root: {root.resolve()}",
        f"snapshot_mode: {snapshot_mode}",
    ]
    if snapshot_title:
        header_lines.append(f"snapshot_title: {snapshot_title}")
    header_lines.extend([
        f"max_file_size_kb: {max_file_size_kb}",
        f"included_files_count: {len(included)}",
        f"skipped_files_count: {len(skipped)}",
        f"missing_files_count: {len(missing_files)}",
        f"total_chars: {total_chars:,}",
    ])
    if include_sources:
        header_lines.append("")
        header_lines.append("Include sources:")
        for src in include_sources:
            header_lines.append(f"  {src}")
    if missing_files:
        header_lines.append("")
        header_lines.append("Missing files (not found):")
        for mf in missing_files:
            header_lines.append(f"  {mf}")
    header_lines.append("")
    header_lines.append("Skipped summary by reason:")
    for reason, count in sorted(skip_reason_counts.items(), key=lambda x: -x[1]):
        header_lines.append(f"  {reason}: {count}")
    header_lines.append("=" * 72)
    header_lines.append("")

    parts = ["\n".join(header_lines)]

    for rel_path, content in included:
        lang = _guess_lang(rel_path)
        fence_open = f"```{lang}" if lang else "```"
        parts.append(f"===== FILE: {rel_path} =====")
        parts.append(fence_open)
        parts.append(content)
        parts.append("```")
        parts.append("")

    return "\n".join(parts)


def write_snapshot(output_path: Path, snapshot_text: str) -> None:
    """Atomically write snapshot: write .tmp then replace."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(snapshot_text, encoding="utf-8")
    tmp_path.replace(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read-only repo snapshot exporter for sharing with LLMs."
    )
    p.add_argument(
        "--root", type=str, default=".",
        help="Repository root directory (default: .)",
    )
    p.add_argument(
        "--out", type=str,
        default="runs/debug/repo_snapshot/repo_snapshot.txt",
        help="Output snapshot file path (default: runs/debug/repo_snapshot/repo_snapshot.txt)",
    )
    p.add_argument(
        "--max-file-size-kb", type=int,
        default=DEFAULT_MAX_FILE_SIZE_KB,
        help=f"Skip files larger than this (KB). Default: {DEFAULT_MAX_FILE_SIZE_KB}",
    )
    p.add_argument(
        "--include-data-metadata", action="store_true", default=False,
        help="Include data/sim/metadata/*.json (default: false)",
    )
    p.add_argument(
        "--include-runs-debug", action="store_true", default=False,
        help="Include files under runs/debug/ (default: false)",
    )
    p.add_argument(
        "--print-skipped", action="store_true", default=False,
        help="Print skipped files and reasons to stderr",
    )
    p.add_argument(
        "--snapshot-title", type=str, default=None,
        help="Title for this snapshot (written into header)",
    )
    p.add_argument(
        "--include", action="append", default=[],
        dest="include_paths",
        help="File or directory to include (can repeat). Enables focused mode.",
    )
    p.add_argument(
        "--include-glob", action="append", default=[],
        dest="include_globs",
        help="Glob pattern to include (can repeat). Enables focused mode.",
    )
    p.add_argument(
        "--include-from", type=str, default=None,
        dest="include_from",
        help="Read include list from a text file (one path/glob per line).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path

    max_bytes = args.max_file_size_kb * 1024

    # Detect focused mode
    is_focused = bool(args.include_paths or args.include_globs or args.include_from)

    # Build set of resolved paths to always skip (output file + its .tmp)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    skip_resolved: set[Path] = set()
    for p in (out_path, tmp_path):
        try:
            skip_resolved.add(p.resolve())
        except OSError:
            skip_resolved.add(p)

    # Build include sources list for header
    include_sources: list[str] = []
    if args.include_paths:
        include_sources.extend(f"--include {p}" for p in args.include_paths)
    if args.include_globs:
        include_sources.extend(f"--include-glob {g}" for g in args.include_globs)
    if args.include_from:
        include_sources.append(f"--include-from {args.include_from}")

    snapshot_mode = "focused" if is_focused else "full"
    print(f"Mode:          {snapshot_mode}", file=sys.stderr)
    print(f"Scanning repo: {root}", file=sys.stderr)
    print(f"Output:        {out_path}", file=sys.stderr)
    print(f"Max file size: {args.max_file_size_kb} KB", file=sys.stderr)
    if args.snapshot_title:
        print(f"Title:         {args.snapshot_title}", file=sys.stderr)
    if is_focused:
        print(f"Include paths: {len(args.include_paths)}", file=sys.stderr)
        print(f"Include globs: {len(args.include_globs)}", file=sys.stderr)
        if args.include_from:
            print(f"Include from:  {args.include_from}", file=sys.stderr)
    else:
        print(f"Include data/sim/metadata: {args.include_data_metadata}", file=sys.stderr)
        print(f"Include runs/debug: {args.include_runs_debug}", file=sys.stderr)
    print(file=sys.stderr)

    if is_focused:
        # --- Focused mode: only collect explicitly specified files ---
        resolved_paths, missing_files = resolve_include_patterns(
            root=root,
            include_paths=args.include_paths,
            include_globs=args.include_globs,
            include_from=args.include_from,
            print_skipped=args.print_skipped,
        )

        included: list[tuple[Path, str]] = []
        skipped: list[tuple[Path, str]] = []
        skip_reason_counts: dict[str, int] = {}

        def _record_skip(rel: Path, reason: str):
            skipped.append((rel, reason))
            skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
            if args.print_skipped:
                print(f"  SKIP [{reason}]: {rel}", file=sys.stderr)

        for filepath in sorted(resolved_paths):
            rel_path = filepath.relative_to(root)

            # Always skip symlinks
            if filepath.is_symlink():
                _record_skip(rel_path, "symlink")
                continue

            # Always skip the output file and its .tmp (anti-recursion)
            if filepath in skip_resolved:
                _record_skip(rel_path, "output snapshot file (anti-recursion)")
                continue

            # Always skip old snapshot .txt files (anti-recursion)
            if filepath.suffix.lower() == ".txt" and "snapshot" in filepath.stem.lower():
                _record_skip(rel_path, "snapshot .txt file (anti-recursion)")
                continue

            # Always skip sensitive files
            if _should_skip_file(filepath):
                _record_skip(rel_path, "skipped file pattern/extension")
                continue

            # Check size
            try:
                size = filepath.stat().st_size
            except OSError as e:
                _record_skip(rel_path, f"stat error: {e}")
                continue
            if size > max_bytes:
                _record_skip(rel_path, f"too large ({size / 1024:.1f} KB > {max_bytes / 1024:.0f} KB)")
                continue

            # Try to read as text
            is_known = _is_text_extension(filepath)
            ok, content_or_reason, _ = _try_read_text(filepath, max_bytes)
            if ok:
                included.append((rel_path, content_or_reason))
            elif is_known:
                _record_skip(rel_path, content_or_reason)
            else:
                _record_skip(rel_path, f"unknown extension + {content_or_reason}")

    else:
        # --- Full mode: walk entire repo ---
        included, skipped, skip_reason_counts = collect_files(
            root=root,
            max_file_size_bytes=max_bytes,
            include_data_metadata=args.include_data_metadata,
            include_runs_debug=args.include_runs_debug,
            print_skipped=args.print_skipped,
            skip_resolved_paths=skip_resolved,
        )
        missing_files = []

    # Sort by path for stable output
    included.sort(key=lambda x: str(x[0]))
    skipped.sort(key=lambda x: str(x[0]))

    snapshot_text = build_snapshot(
        included, skipped, skip_reason_counts, root, args.max_file_size_kb,
        snapshot_mode=snapshot_mode,
        snapshot_title=args.snapshot_title,
        missing_files=missing_files,
        include_sources=include_sources,
    )
    write_snapshot(out_path, snapshot_text)

    total_chars = sum(len(c) for _, c in included)
    print(f"\nDone.", file=sys.stderr)
    print(f"  Mode:            {snapshot_mode}", file=sys.stderr)
    print(f"  Included files:  {len(included)}", file=sys.stderr)
    print(f"  Skipped files:   {len(skipped)}", file=sys.stderr)
    print(f"  Missing files:   {len(missing_files)}", file=sys.stderr)
    print(f"  Total chars:     {total_chars:,}", file=sys.stderr)
    print(f"  Output:          {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
