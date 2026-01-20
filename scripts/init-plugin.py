#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_warn(message: str) -> None:
    print(f"[WARN] {message}")


def normalize_plugin_name(plugin_raw: str) -> tuple[str, str, str]:
    """Return (original, underscore_variant, dash_variant)."""
    original = plugin_raw.strip()
    if not original:
        raise ValueError("Invalid name, aborting.")

    normalized = " ".join(original.split())
    # Replace '-' and '_' with spaces BEFORE converting to variants.
    cleaned = normalized.replace("-", " ").replace("_", " ")
    cleaned = " ".join(cleaned.split()).lower()

    underscore = cleaned.replace(" ", "_")
    dash = cleaned.replace(" ", "-")
    return original, underscore, dash


def iter_root_files(repo_root: Path) -> list[Path]:
    return [p for p in repo_root.iterdir() if p.is_file()]


def iter_github_files(repo_root: Path) -> list[Path]:
    github_dir = repo_root / ".github"
    if not github_dir.exists():
        return []
    return [p for p in github_dir.rglob("*") if p.is_file()]


def replace_in_file(path: Path, *, plugin_snake: str, plugin_dash: str) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        log_warn(f"Skipping non-UTF8 file: {path}")
        return False

    replaced = text.replace("rag2f_plugin_template", plugin_snake)
    replaced = replaced.replace("rag2f-plugin-template", plugin_dash)
    if replaced == text:
        return False

    path.write_text(replaced, encoding="utf-8")
    return True


def rename_src_dir(repo_root: Path, plugin_snake: str) -> None:
    old_src = repo_root / "src" / "rag2f_plugin_template"
    new_src = repo_root / "src" / plugin_snake

    if not old_src.exists():
        log_info(f"Directory {old_src} not found, skipping rename")
        return

    if new_src.exists() and old_src.resolve() != new_src.resolve():
        raise FileExistsError(f"Destination {new_src} already exists.")

    old_src.rename(new_src)
    log_info(f"Renamed src directory to {new_src}")


def update_plugin_json(repo_root: Path, plugin_original: str) -> None:
    plugin_json_path = repo_root / "plugin.json"
    data = json.loads(plugin_json_path.read_text(encoding="utf-8"))
    data["name"] = plugin_original
    plugin_json_path.write_text(
        json.dumps(data, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log_info(f"Updated plugin.json with name {plugin_original}")


def remove_pre_commit_hook(repo_root: Path) -> None:
    hook_path = repo_root / ".git" / "hooks" / "pre-commit"
    if hook_path.exists():
        hook_path.unlink(missing_ok=True)
        log_info("Removed pre-commit hook")
    else:
        log_info("No pre-commit hook present")


def get_repo_root() -> Path:
    # scripts/init-plugin.py -> repo root is parent of scripts/
    return Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize this repository as a new RAG2F plugin.")
    parser.add_argument(
        "--name",
        dest="plugin_name",
        help="Plugin display name (original value stored in plugin.json name).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = get_repo_root()
    log_info(f"Repository root: {repo_root}")

    plugin_input = args.plugin_name
    if not plugin_input:
        plugin_input = input("Enter plugin name: ")

    try:
        plugin_original, plugin_snake, plugin_dash = normalize_plugin_name(plugin_input)
    except ValueError as exc:
        print(str(exc))
        return 1

    os.environ["PLUGIN_SNAKE"] = plugin_snake
    os.environ["PLUGIN_DASH"] = plugin_dash
    os.environ["PLUGIN_ORIGINAL"] = plugin_original

    log_info("Running replacements in root files")
    for file_path in iter_root_files(repo_root):
        replace_in_file(file_path, plugin_snake=plugin_snake, plugin_dash=plugin_dash)

    log_info("Running replacements in .github")
    for file_path in iter_github_files(repo_root):
        replace_in_file(file_path, plugin_snake=plugin_snake, plugin_dash=plugin_dash)

    rename_src_dir(repo_root, plugin_snake)
    update_plugin_json(repo_root, plugin_original)
    remove_pre_commit_hook(repo_root)

    print(
        f"Plugin prepared: {plugin_original} (underscore: {plugin_snake}, dash: {plugin_dash})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
