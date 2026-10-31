#!/usr/bin/env python3
"""Read-only checks for the staged/public v0.1.1 distribution and v0.1 rollback.

Published downloads deliberately use urllib without GitHub credentials. Draft
downloads use gh because GitHub hides draft assets from anonymous clients.
Nothing in this program publishes, merges, updates an installation, or uses a GPU.
"""

import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "receipts" / "v0.1.1"
REPO = "seanyourhighness/L0xRE-BeeLLama-Low"
TAG = "l0xre-v0.1.1"
API = "https://api.github.com/repos/" + REPO
RAW = "https://raw.githubusercontent.com/" + REPO + "/main/"
DOWNLOAD = "https://github.com/" + REPO + "/releases/download/"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def anonymous_open(url):
    request = urllib.request.Request(url, headers={"User-Agent": "l0xre-public-verifier/1", "Accept": "application/vnd.github+json" if url.startswith(API) else "*/*"})
    return urllib.request.urlopen(request, timeout=60)


def anonymous_json(url):
    with anonymous_open(url) as response:
        return json.load(response)


def gh_json(endpoint):
    return json.loads(subprocess.check_output(["gh", "api", endpoint], text=True))


def check_stream(stream, record):
    sha = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):
        size += len(chunk)
        sha.update(chunk)
    actual = sha.hexdigest()
    require(size == record["bytes"] and actual == record["sha256"], "Downloaded bytes/hash differ: " + record["name"])
    return {"name": record["name"], "bytes": size, "sha256": actual, "passed": True}


def download(record, asset=None):
    if asset is None:
        url = DOWNLOAD + record.get("tag", TAG) + "/" + urllib.parse.quote(record["name"])
        with anonymous_open(url) as stream:
            result = check_stream(stream, record)
        result["authentication"] = "none"
    else:
        command = ["gh", "api", "repos/" + REPO + "/releases/assets/" + str(asset["id"]), "-H", "Accept: application/octet-stream"]
        with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
            result = check_stream(process.stdout, record)
            require(process.wait() == 0, "gh asset download failed: " + record["name"])
        result["authentication"] = "draft asset via gh"
    print("verified " + record["name"], file=sys.stderr, flush=True)
    return result


def validate_release(release, expected, draft):
    require(release["tag_name"] == TAG, "Wrong release tag")
    require(release["draft"] is draft, "Unexpected draft/published state")
    require((release["published_at"] is None) is draft, "Unexpected publication timestamp")
    require(digest(release["body"].encode()) == expected["notes_sha256"], "Release notes changed")
    require("L0xRE-BeeLLama-Low-source" not in release["body"], "Private repository link in notes")
    require(len(release["assets"]) == 11, "Expected exactly eleven release asset records")
    actual = {asset["name"]: asset for asset in release["assets"]}
    require(len(actual) == 11 and set(actual) == {record["name"] for record in expected["assets"]}, "Expected exactly the frozen eleven release assets")
    for record in expected["assets"]:
        asset = actual[record["name"]]
        require(asset["state"] == "uploaded", "Incomplete upload: " + record["name"])
        require(asset["size"] == record["bytes"] and asset["digest"] == "sha256:" + record["sha256"], "GitHub asset identity changed: " + record["name"])
    return actual


def validate_catalog(catalog, expected):
    require(catalog["schema_version"] == 1, "Invalid catalog schema")
    require("private_draft" not in catalog and "publication_allowed" not in catalog, "Private draft workflow fields remain in the public catalog")
    records = {record["name"]: record for record in expected["assets"]}
    require(set(catalog["packages"]) == {"linux", "windows"}, "Wrong platforms")
    for platform in ("linux", "windows"):
        require(set(catalog["packages"][platform]) == {"sm86", "sm89", "sm120"}, "Wrong architecture set")
        for arch, package in catalog["packages"][platform].items():
            record = records[package["filename"]]
            require(package["url"] == DOWNLOAD + TAG + "/" + record["name"], "Package URL does not target the public release")
            require(package["sha256"] == record["sha256"] and package["bytes"] == record["bytes"], "Catalog artifact identity differs")
            require("private_draft_only" not in package, "Public catalog still marked private-only")
            require(package["memory_profiles"] == (["12gb"] if arch == "sm86" else ["12gb", "16gb"]), "Wrong memory profiles")
            if platform == "windows" and arch != "sm120":
                require(package["status"] == "build-verified" and not package["certified_gpu_names"], "Unsupported native Windows hardware claim")


def validate_catalog_projection(frozen, public):
    projected = json.loads(json.dumps(frozen))
    projected.pop("private_draft", None)
    projected.pop("publication_allowed", None)
    for packages in projected["packages"].values():
        for package in packages.values():
            package["url"] = DOWNLOAD + TAG + "/" + package["filename"]
            package.pop("private_draft_only", None)
    require(projected == public, "Public catalog changed something other than distribution URLs/private-draft markers")


def validate_release_checksums(data, expected):
    records = {record["name"]: record["sha256"] for record in expected["assets"] if record["name"] != "SHA256SUMS"}
    found = {}
    for line in data.decode("ascii").splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  ([^/\\]+)", line)
        require(match is not None, "Invalid release SHA256SUMS line")
        checksum, name = match.groups()
        require(name not in found, "Duplicate release checksum: " + name)
        found[name] = checksum
    require(found == records, "Release SHA256SUMS does not match the current asset hashes")


def release_asset_bytes(asset, draft):
    if draft:
        return subprocess.check_output(["gh", "api", "repos/" + REPO + "/releases/assets/" + str(asset["id"]), "-H", "Accept: application/octet-stream"])
    with anonymous_open(DOWNLOAD + TAG + "/" + urllib.parse.quote(asset["name"])) as stream:
        return stream.read()


def main_file_records():
    records = read_json(RECEIPTS / "public-main-files.json")["files"]
    require(isinstance(records, list) and records, "Empty public main file manifest")
    paths = [record["path"] for record in records]
    require(len(paths) == len(set(paths)), "Duplicate path in public main file manifest")
    required = {
        "README.md", "MANIFEST.json", "THIRD-PARTY-NOTICES.md",
        "install/catalog.json", "install/README.md", "docs/README.md",
        "docs/V011-RELEASE.md", "docs/V011-PROVENANCE.md", "docs/V011-ROLLBACK.md",
        "tools/verify-public.py", "receipts/v0.1.1/expected-assets.json",
        "receipts/v0.1.1/rollback-v0.1-assets.json",
        "receipts/v0.1.1/public-draft-upload-hashes.json",
        "receipts/v0.1.1/public-catalog-projection.json",
        "receipts/v0.1.1/REVIEW.md", "receipts/v0.1.1/catalog-asset-swap.json",
        "receipts/v0.1.1/SHA256SUMS.public-candidate",
        "receipts/v0.1.1/gates-core.json", "receipts/v0.1.1/source-extract-build.json",
    }
    require(required.issubset(paths), "Public main file manifest omits critical files: " + ", ".join(sorted(required - set(paths))))
    for record in records:
        path = (ROOT / record["path"]).resolve()
        require(path.is_relative_to(ROOT) and path.relative_to(ROOT).as_posix() == record["path"], "Invalid path in public file manifest")
        require(re.fullmatch(r"[a-f0-9]{64}", record["sha256"]) is not None, "Invalid SHA-256 in public file manifest")
    return records


def check_local():
    expected = read_json(RECEIPTS / "expected-assets.json")
    require(expected["repository"] == REPO and expected["tag"] == TAG, "Wrong expected repository/tag")
    require(len(expected["assets"]) == 11, "Wrong expected asset count")
    catalog = read_json(ROOT / "install" / "catalog.json")
    validate_catalog(catalog, expected)
    manifest = read_json(ROOT / "MANIFEST.json")
    require(manifest["release"] == TAG, "Public manifest names the wrong release")
    manifest_assets = {record["filename"]: record for record in manifest["assets"]}
    require(set(manifest_assets) == {record["name"] for record in expected["assets"]}, "Public manifest must name the same eleven assets")
    for record in expected["assets"]:
        declared = manifest_assets[record["name"]]
        require(declared["sha256"] == record["sha256"] and declared["bytes"] == record["bytes"], "Public manifest artifact identity differs")
        require(declared["url"] == DOWNLOAD + TAG + "/" + record["name"], "Public manifest asset URL is not public")
    require(manifest["models"] == catalog["models"], "Public manifest model pins differ from installer catalog")
    notes = ROOT / "docs" / "V011-RELEASE.md"
    require(digest(notes.read_bytes()) == expected["notes_sha256"], "Public notes differ from frozen release notes")
    for record in main_file_records():
        path = (ROOT / record["path"]).resolve()
        require(path.is_relative_to(ROOT), "Invalid path in public file manifest")
        require(digest(path.read_bytes()) == record["sha256"], "Prepared public file changed: " + record["path"])
    # Only the current release documents are in scope. Historical documentation
    # remains an unchanged record of earlier packages and protocols.
    paths = [ROOT / "README.md", ROOT / "install" / "README.md", ROOT / "docs" / "README.md"]
    paths += sorted((ROOT / "docs").glob("V011-*.md"))
    links = []
    for path in paths:
        text = path.read_text()
        require("L0xRE-BeeLLama-Low-source" not in text, "Private repo reference in " + str(path))
        for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", text):
            target = target.strip("<>")
            if target.startswith(("https://", "http://", "mailto:")):
                continue
            relative = urllib.parse.unquote(target.split("#")[0])
            resolved = (path.parent / relative).resolve() if relative else path
            require(resolved.is_relative_to(ROOT) and resolved.exists(), "Dead repository path in " + str(path.relative_to(ROOT)) + ": " + target)
            if "#" in target and resolved.suffix.lower() == ".md":
                fragment = urllib.parse.unquote(target.split("#", 1)[1])
                headings = re.findall(r"^#+\s+(.+)$", resolved.read_text(), re.MULTILINE)
                anchors = {re.sub(r"[^\w\s-]", "", heading.lower()).replace(" ", "-") for heading in headings}
                require(not fragment or fragment in anchors, "Dead heading link in " + str(path.relative_to(ROOT)) + ": " + target)
            links.append({"document": str(path.relative_to(ROOT)), "target": target})
    return expected, catalog, links


def check_rollback():
    baseline = read_json(RECEIPTS / "rollback-v0.1-assets.json")
    require(baseline["repository"] == REPO and baseline["tag"] == "l0xre-v0.1", "Wrong rollback repository or release")
    expected_names = {"L0xRE-v0.1-" + platform + "-" + arch + extension
                      for platform, extension in (("linux", ".tar.zst"), ("windows", ".zip"))
                      for arch in ("sm86", "sm89", "sm120")}
    require(len(baseline["assets"]) == 6 and {record["name"] for record in baseline["assets"]} == expected_names, "Rollback requires all six published v0.1 runtime archives")
    release = anonymous_json(API + "/releases/tags/l0xre-v0.1")
    require(not release["draft"] and release["id"] == baseline["release_id"], "Previous published release is unavailable or replaced")
    actual = {asset["name"]: asset for asset in release["assets"]}
    jobs = []
    for record in baseline["assets"]:
        asset = actual[record["name"]]
        require(asset["id"] == record["asset_id"] and asset["size"] == record["bytes"] and asset["digest"] == "sha256:" + record["sha256"], "Rollback release asset changed")
        jobs.append(dict(record, tag="l0xre-v0.1"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        downloads = list(pool.map(download, jobs))
    completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "install", "-p", "test_install.py"], cwd=ROOT, capture_output=True, text=True)
    require(completed.returncode == 0, "Installer update/rollback checks failed:\n" + completed.stdout + completed.stderr)
    return {"passed": True, "previous_release": "l0xre-v0.1", "assets_unchanged": downloads, "installer_offline_suite": completed.stdout + completed.stderr, "scope": "Published archive preservation and offline installer rollback logic; no live service or GPU is changed. Actual GPU startup/rollback evidence remains in the certification receipts."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-local", action="store_true")
    mode.add_argument("--check-draft", action="store_true")
    mode.add_argument("--verify-public", action="store_true")
    parser.add_argument("--check-rollback", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"repository": REPO, "tag": TAG, "checked_at": datetime.now(timezone.utc).isoformat(), "passed": False}
    try:
        expected, catalog, links = check_local()
        result["repository_paths_checked"] = links
        if not args.check_local:
            repository = gh_json("repos/" + REPO) if args.check_draft else anonymous_json(API)
            require(repository["private"] is False, "Distribution repository must be public")
            if args.check_draft:
                releases = gh_json("repos/" + REPO + "/releases?per_page=100")
                matching = [release for release in releases if release["tag_name"] == TAG]
                require(len(matching) == 1, "Expected exactly one matching draft release")
                release = matching[0]
            else:
                release = anonymous_json(API + "/releases/tags/" + TAG)
                require(anonymous_json(API + "/releases/latest")["tag_name"] == TAG, "v0.1.1 is not marked as the latest public release")
                with anonymous_open(RAW + "install/catalog.json") as stream:
                    require(stream.read() == (ROOT / "install/catalog.json").read_bytes(), "Public main installer catalog differs from the prepared PR")
                with anonymous_open(RAW + "receipts/v0.1.1/expected-assets.json") as stream:
                    require(stream.read() == (RECEIPTS / "expected-assets.json").read_bytes(), "Public main asset manifest differs from the prepared PR")
                main_files = main_file_records()
                def check_public_file(record):
                    with anonymous_open(RAW + urllib.parse.quote(record["path"], safe="/")) as stream:
                        require(digest(stream.read()) == record["sha256"], "Public main file differs from prepared PR: " + record["path"])
                    return record["path"]
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    result["public_main_files_verified"] = list(pool.map(check_public_file, main_files))
            assets = validate_release(release, expected, args.check_draft)
            catalog_bytes = release_asset_bytes(assets["catalog-v0.1.1.json"], args.check_draft)
            require(catalog_bytes == (ROOT / "install/catalog.json").read_bytes(), "Uploaded catalog must equal the public installer catalog byte for byte")
            validate_catalog_projection(json.loads(catalog_bytes), catalog)
            validate_release_checksums(release_asset_bytes(assets["SHA256SUMS"], args.check_draft), expected)
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(download, record, assets[record["name"]] if args.check_draft else None) for record in expected["assets"]]
                result["downloads"] = [future.result() for future in futures]
            result.update(release_id=release["id"], draft=release["draft"], release_url=release["html_url"], notes_sha256=expected["notes_sha256"], public_catalog_matches_installer=True, release_checksums_match_assets=True)
        if args.check_rollback:
            result["rollback"] = check_rollback()
        result["passed"] = True
    except Exception as error:
        result["error"] = str(error)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
