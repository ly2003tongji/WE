#!/usr/bin/env python3
"""从官方单体 scenario pickle 和 asset shard 构造小场景闭环子集。"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import tarfile
from pathlib import Path, PurePosixPath


ARCHIVE_PREFIX = PurePosixPath("navtest_failures/assets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-pkl", type=Path, required=True)
    parser.add_argument("--asset-tar", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene", action="append", required=True)
    return parser.parse_args()


def write_scenarios(source: Path, destination: Path, scene_names: list[str]) -> None:
    with source.open("rb") as handle:
        scenarios = pickle.load(handle)
    if not isinstance(scenarios, dict):
        raise TypeError(f"scenario pickle 顶层必须是 dict，实际为 {type(scenarios)!r}")

    missing = sorted(set(scene_names) - scenarios.keys())
    if missing:
        raise KeyError(f"scenario pickle 缺少场景：{missing}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(
            {name: scenarios[name] for name in scene_names},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def safe_destination(root: Path, member_name: str) -> Path | None:
    archive_path = PurePosixPath(member_name)
    try:
        relative = archive_path.relative_to(ARCHIVE_PREFIX)
    except ValueError:
        return None
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        return None
    destination = root.joinpath(*relative.parts)
    if os.path.commonpath((root.resolve(), destination.resolve())) != str(root.resolve()):
        raise ValueError(f"拒绝路径穿越：{member_name}")
    return destination


def extract_assets(
    archive: Path,
    destination_root: Path,
    scene_names: list[str],
) -> tuple[int, int]:
    wanted = set(scene_names)
    found: set[str] = set()
    member_count = 0
    extracted_bytes = 0
    destination_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar:
            destination = safe_destination(destination_root, member.name)
            if destination is None:
                continue
            relative = PurePosixPath(member.name).relative_to(ARCHIVE_PREFIX)
            scene_name = relative.parts[0]
            if scene_name not in wanted:
                continue
            found.add(scene_name)
            member_count += 1

            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"拒绝非普通文件成员：{member.name}")

            destination.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise OSError(f"无法读取 archive 成员：{member.name}")
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
            extracted_bytes += member.size

    missing = sorted(wanted - found)
    if missing:
        raise KeyError(f"asset shard 缺少场景：{missing}")
    return member_count, extracted_bytes


def main() -> None:
    args = parse_args()
    scene_names = list(dict.fromkeys(args.scene))
    scenario_output = (
        args.output_root
        / "scenarios"
        / "original"
        / "navtest_failures"
        / "all_scenarios.pkl"
    )
    asset_output = args.output_root / "assets" / "navtest_failures" / "assets"

    write_scenarios(args.scenario_pkl, scenario_output, scene_names)
    members, extracted_bytes = extract_assets(args.asset_tar, asset_output, scene_names)
    print(f"scenes={len(scene_names)}")
    print(f"scenario_pkl={scenario_output}")
    print(f"asset_root={asset_output}")
    print(f"archive_members={members}")
    print(f"asset_bytes={extracted_bytes}")


if __name__ == "__main__":
    main()
