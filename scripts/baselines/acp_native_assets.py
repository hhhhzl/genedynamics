"""Inventory or fetch actual ACP author resources with ZIP CRC and SHA256 evidence.

HTTP byte ranges let checkpoint/episode qualification fetch selected archive members
without claiming that a selected episode is the full released training dataset.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import fnmatch
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

URLS = {
    "checkpoints": "https://real.stanford.edu/adaptive-compliance/checkpoints.zip",
    "flip_up": "https://real.stanford.edu/adaptive-compliance/data/flip_up_230.zip",
    "vase_wiping": "https://real.stanford.edu/adaptive-compliance/data/vase_wiping_200.zip",
}


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
            self.headers = dict(response.headers)
            self.length = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")
        self.position = 0
        self.bytes_fetched = 0
        self.block_size = 8 * 1024**2
        self.cache = OrderedDict()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset if whence == io.SEEK_SET else self.position + offset if whence == io.SEEK_CUR else self.length + offset
        if position < 0:
            raise ValueError("negative HTTP archive position")
        self.position = position
        return position

    def read(self, size=-1):
        stop = self.length if size < 0 else min(self.length, self.position + size)
        if stop <= self.position:
            return b""
        result = []
        while self.position < stop:
            index = self.position // self.block_size
            if index not in self.cache:
                start = index * self.block_size
                self.cache[index] = self._fetch(start, min(self.length, start+self.block_size))
                if len(self.cache) > 4:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(index)
            block = self.cache[index]
            offset = self.position - index*self.block_size
            count = min(stop-self.position, len(block)-offset)
            result.append(block[offset:offset+count])
            self.position += count
        return b"".join(result)

    def _fetch(self, start, stop):
        headers = {"Range": f"bytes={start}-{stop-1}"}
        if self.etag:
            headers["If-Range"] = self.etag
        request = urllib.request.Request(self.url, headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{stop-1}/{self.length}":
                raise RuntimeError("author archive changed or server did not honor the exact byte range")
            data = response.read()
        if len(data) != stop - start:
            raise EOFError("incomplete author archive range")
        self.bytes_fetched += len(data)
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "fetch"))
    parser.add_argument("--asset", choices=tuple(URLS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--member", action="append", default=[])
    parser.add_argument("--max-extracted-gib", type=float, default=15)
    args = parser.parse_args()
    root = args.output.resolve()
    if "latex" in root.parts:
        parser.error("asset evidence belongs outside the paper draft")
    root.mkdir(parents=True, exist_ok=True)
    remote = HTTPRangeFile(URLS[args.asset])
    with zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
        inventory = {"asset": args.asset, "url": remote.url, "archive_bytes": remote.length,
                     "headers": remote.headers,
                     "members": [{"path": info.filename, "bytes": info.file_size,
                                  "compressed_bytes": info.compress_size, "zip_crc32": f"{info.CRC:08x}"}
                                 for info in infos]}
        (root / f"{args.asset}.inventory.json").write_text(json.dumps(inventory, indent=2)+"\n")
        if args.action == "inventory":
            print(json.dumps({"asset": args.asset, "archive_bytes": remote.length,
                              "members": len(infos), "inventory": str(root / f"{args.asset}.inventory.json")}, indent=2))
            return
        if not args.member:
            parser.error("fetch requires one or more explicit --member patterns")
        selected = [info for info in infos if not info.is_dir()
                    and any(fnmatch.fnmatchcase(info.filename, pattern) for pattern in args.member)]
        if not selected:
            raise SystemExit("no author archive members matched")
        total = sum(info.file_size for info in selected)
        if total > args.max_extracted_gib * 1024**3:
            raise SystemExit(f"selected members exceed extraction budget: {total} bytes")
        manifest = {"asset": args.asset, "url": remote.url, "archive_etag": remote.etag,
                    "selection": args.member, "complete_archive": len(selected) == sum(not i.is_dir() for i in infos),
                    "members": []}
        manifest_path = root / f"{args.asset}.manifest.json"
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        reusable = {item["path"]: item for item in previous.get("members", [])}
        if previous and (previous.get("url") != remote.url or previous.get("archive_etag") != remote.etag):
            raise ValueError("existing asset evidence refers to a different author archive version")
        for i, info in enumerate(selected):
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts or ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("unsafe author archive member")
            destination = root.joinpath(*relative.parts)
            if destination.is_symlink() or not destination.resolve().is_relative_to(root):
                raise ValueError("asset destination escapes its extraction directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            entry = reusable.get(str(destination.relative_to(root)))
            if entry is not None and destination.is_file() and destination.stat().st_size == info.file_size:
                if hashlib.sha256(destination.read_bytes()).hexdigest() == entry["sha256"]:
                    manifest["members"].append(entry)
                    continue
            temporary = destination.with_name(destination.name+".partial")
            digest = hashlib.sha256()
            with archive.open(info) as source, temporary.open("wb") as output:
                while data := source.read(8 * 1024**2):
                    output.write(data)
                    digest.update(data)
            if temporary.stat().st_size != info.file_size:
                raise EOFError("incomplete extracted author resource")
            temporary.replace(destination)
            manifest["members"].append({"path": str(destination.relative_to(root)),
                                        "bytes": info.file_size, "sha256": digest.hexdigest(),
                                        "zip_crc32": f"{info.CRC:08x}"})
            if i % 100 == 0 or i == len(selected)-1:
                print(json.dumps({"completed": i+1, "total": len(selected), "last": info.filename}), flush=True)
            (root / f"{args.asset}.manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        print(json.dumps({"asset": args.asset, "selected_bytes": total,
                          "downloaded_bytes": remote.bytes_fetched, "members": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
