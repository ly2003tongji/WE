#!/usr/bin/env python3
"""只读列出并比较 WorldEngine 的 Hugging Face/ModelScope 文件清单。"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass


HF_API = "https://huggingface.co/api/datasets/OpenDriveLab/WorldEngine/tree/main"
MS_INFO_API = "https://www.modelscope.cn/api/v1/datasets/OpenDriveLab/WorldEngine"
MS_TREE_API = "https://www.modelscope.cn/api/v1/datasets/{dataset_id}/repo/tree"


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int


def get_json(url: str, params: dict[str, object] | None = None) -> tuple[object, object]:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "worldengine-h20-audit/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response), response.headers


def huggingface_files() -> tuple[list[RemoteFile], str]:
    info, _ = get_json("https://huggingface.co/api/datasets/OpenDriveLab/WorldEngine")
    assert isinstance(info, dict)
    revision = str(info["sha"])

    url = f"{HF_API}?recursive=true&expand=true"
    files: list[RemoteFile] = []
    while url:
        items, headers = get_json(url)
        assert isinstance(items, list)
        files.extend(
            RemoteFile(str(item["path"]), int(item.get("size", 0)))
            for item in items
            if item["type"] == "file"
        )
        next_url = ""
        for link in str(headers.get("Link", "")).split(","):
            if 'rel="next"' in link:
                next_url = link.split(";", 1)[0].strip().strip("<>")
                break
        url = next_url
    return sorted(files, key=lambda item: item.path), revision


def modelscope_files() -> tuple[list[RemoteFile], int]:
    info, _ = get_json(MS_INFO_API)
    assert isinstance(info, dict)
    dataset_id = int(info["Data"]["Id"])

    files: list[RemoteFile] = []
    page_number = 1
    while True:
        payload, _ = get_json(
            MS_TREE_API.format(dataset_id=dataset_id),
            {
                "Revision": "master",
                "Root": "/",
                "Recursive": "True",
                "PageNumber": page_number,
                "PageSize": 100,
            },
        )
        assert isinstance(payload, dict)
        entries = payload["Data"]["Files"]
        files.extend(
            RemoteFile(str(item["Path"]), int(item.get("Size", 0)))
            for item in entries
            if item["Type"] != "tree"
        )
        if len(entries) < 100:
            break
        page_number += 1
    return sorted(files, key=lambda item: item.path), dataset_id


def print_tsv(source: str, files: list[RemoteFile]) -> None:
    for item in files:
        print(f"{source}\t{item.path}\t{item.size}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=("huggingface", "modelscope", "both"),
        default="both",
    )
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    hf_files: list[RemoteFile] = []
    ms_files: list[RemoteFile] = []
    if args.source in {"huggingface", "both"}:
        hf_files, hf_revision = huggingface_files()
        if args.summary:
            print(
                f"huggingface\trevision={hf_revision}\t"
                f"files={len(hf_files)}\tbytes={sum(item.size for item in hf_files)}"
            )
        else:
            print_tsv("huggingface", hf_files)

    if args.source in {"modelscope", "both"}:
        ms_files, ms_dataset_id = modelscope_files()
        if args.summary:
            print(
                f"modelscope\tdataset_id={ms_dataset_id}\t"
                f"files={len(ms_files)}\tbytes={sum(item.size for item in ms_files)}"
            )
        else:
            print_tsv("modelscope", ms_files)

    if args.source == "both":
        hf = {item.path: item.size for item in hf_files}
        ms = {item.path: item.size for item in ms_files}
        print(f"compare\tonly_huggingface={len(hf.keys() - ms.keys())}")
        print(f"compare\tonly_modelscope={len(ms.keys() - hf.keys())}")
        print(
            "compare\tsize_mismatches="
            f"{sum(hf[path] != ms[path] for path in hf.keys() & ms.keys())}"
        )


if __name__ == "__main__":
    main()
