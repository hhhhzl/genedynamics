"""Gated assets remain honest and archive members stay inside an owned tree."""
import json
from pathlib import Path
import stat
import zipfile

import pytest

from scripts.baselines.human2sim2robot_native_assets import (
    archive_specs, native_manifest, safe_extract,
)


def test_expected_author_routes_include_complete_processed_and_optional_raw_data():
    specs = archive_specs(include_raw=True)
    assert len(specs) == 6
    assert {row["route"] for row in specs} == {
        "assets.zip", "pretrained_models.zip",
        "data/human_demo_processed_data/snackbox_push.zip",
        "data/human_demo_processed_data/plate_pivotliftrack.zip",
        "data/human_demo_raw_data/snackbox_push.zip",
        "data/human_demo_raw_data/plate_pivotliftrack.zip",
    }


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink", "oversize", "wrong_top"])
def test_native_zip_members_are_validated_before_any_extraction(tmp_path, kind):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        if kind == "symlink":
            entry = zipfile.ZipInfo("assets/link")
            entry.create_system = 3
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            stream.writestr(entry, "../../../escape")
        else:
            name = {"traversal": "assets/../../escape", "absolute": "/tmp/escape",
                    "oversize": "assets/data", "wrong_top": "pretrained_models/data"}[kind]
            stream.writestr(name, "payload")
    destination = tmp_path / "unpacked"
    with pytest.raises(ValueError):
        safe_extract(archive, destination, top="assets", max_unpacked_bytes=1 if kind == "oversize" else 1024)
    assert not destination.exists()


def test_valid_archive_receives_file_hashes_and_preserves_expected_tree(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("assets/robot/model.urdf", "<robot />")
    destination = tmp_path / "unpacked"
    files = safe_extract(archive, destination, top="assets", max_unpacked_bytes=1024)
    assert files["assets/robot/model.urdf"]["bytes"] == 9
    assert len(files["assets/robot/model.urdf"]["sha256"]) == 64
    assert (destination / "assets/robot/model.urdf").read_text() == "<robot />"


def test_pending_native_manifest_never_publishes_private_url(monkeypatch):
    from scripts.baselines.human2sim2robot_native_assets import ROOT
    monkeypatch.setenv("HUMAN2SIM2ROBOT_ASSET_URL", "https://example.invalid/private-access-key")
    result = native_manifest(ROOT / ".runtime/test_native_manifest_missing_assets")
    assert result["status"] == "assets_pending"
    assert result["private_url_provided"]
    assert not result["native_method_qualified"]
    assert "private-access-key" not in json.dumps(result)
    assert result["native_environment"]["python"].startswith("3.8")
