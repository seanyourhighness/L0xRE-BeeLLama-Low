"""Offline installer selection and clean-extract integration tests."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("install.py")
spec = importlib.util.spec_from_file_location("l0xre_install", SOURCE)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
CATALOG = installer.load_catalog()


def gpu(arch="sm86", name="NVIDIA GeForce RTX 3060", memory=12288):
    return {"index": 0, "uuid": "GPU-fixture", "name": name, "arch": arch, "memory_mib": memory}


def record(path):
    return {"filename": path.name, "sha256": installer.sha256(path), "bytes": path.stat().st_size,
            "url": "https://example.invalid/" + path.name}


def archive_at(path, files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for name, data in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(data)
            item.mode = 0o755
            tar.addfile(item, io.BytesIO(data))
    path.write_bytes(subprocess.check_output(["zstd", "--quiet", "--compress", "--stdout"], input=stream.getvalue()))


class SelectionTests(unittest.TestCase):
    def test_certified_route_and_pending_release(self):
        self.assertEqual(installer.make_plan(CATALOG, gpu())["package"]["status"], "certified")
        blackwell = gpu("sm120", "NVIDIA GeForce RTX 5090", 32607)
        with self.assertRaisesRegex(ValueError, "not published"):
            installer.make_plan(CATALOG, blackwell)
        self.assertEqual(installer.make_plan(CATALOG, blackwell, runtime_archive="sealed.tar.zst")["package"]["status"], "certified-local")

    def test_candidate_needs_opt_in(self):
        ada = gpu("sm89", "NVIDIA GeForce RTX 4090", 24564)
        with self.assertRaisesRegex(ValueError, "allow-candidate"):
            installer.make_plan(CATALOG, ada)
        self.assertEqual(installer.make_plan(CATALOG, ada, True)["package"]["status"], "candidate")

    def test_card_scope_and_vram(self):
        for device in [gpu(memory=8192), gpu("sm75"), gpu("sm120", "RTX 5080", 32768), gpu("sm89", "RTX 4080", 24576)]:
            with self.assertRaises(ValueError):
                installer.make_plan(CATALOG, device, True, "sealed.tar.zst")

    def test_visible_gpu_selection(self):
        line = "1, GPU-second, NVIDIA GeForce RTX 3060, 8.6, 12288\n"
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-second,GPU-first"}), patch.object(installer.subprocess, "check_output", return_value=line) as call:
            self.assertEqual(installer.detect_gpu()["uuid"], "GPU-second")
            self.assertIn("--id=GPU-second", call.call_args.args[0])
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            with self.assertRaises(ValueError):
                installer.detect_gpu()

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(installer, "detect_gpu", return_value=gpu()), patch.object(installer, "get_artifact") as download:
            destination = Path(temp) / "not-created"
            self.assertEqual(installer.main(["--dry-run", "--dir", str(destination)]), 0)
            self.assertFalse(destination.exists())
            download.assert_not_called()


class ArtifactTests(unittest.TestCase):
    def test_artifact_filename_escape(self):
        for filename in ("../outside", "..", "folder\\outside"):
            with self.assertRaises(ValueError):
                installer.get_artifact({"filename": filename}, Path("unused"))

    def test_existing_and_complete_partial_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "model.gguf"; source.write_bytes(b"model fixture")
            expected = record(source)
            with patch.object(installer.subprocess, "run") as network:
                self.assertEqual(installer.get_artifact(expected, root), source)
                source.rename(root / "model.gguf.partial")
                self.assertEqual(installer.get_artifact(expected, root), source)
                network.assert_not_called()
            source.write_bytes(b"wrong model")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                installer.get_artifact(expected, root)
            self.assertEqual(source.read_bytes(), b"wrong model")

    def test_download_and_bad_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);data=b"download fixture"
            expected={"filename":"model.gguf","bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"url":"https://example.invalid/model.gguf"}
            def fetch(command, **kwargs):
                self.assertIn("--continue-at", command)
                Path(command[command.index("--output")+1]).write_bytes(data)
            with patch.object(installer.subprocess, "run", side_effect=fetch):
                self.assertEqual(installer.get_artifact(expected, root).read_bytes(), data)
            (root/"model.gguf").unlink();expected["sha256"]="0"*64
            with patch.object(installer.subprocess, "run", side_effect=fetch), self.assertRaises(ValueError):
                installer.get_artifact(expected, root)
            self.assertFalse((root/"model.gguf").exists())
            self.assertFalse((root/"model.gguf.partial").exists())

    def test_manifest_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"package";root.mkdir()
            outside=root.parent/"outside";outside.write_bytes(b"outside")
            (root/"SHA256SUMS").write_text(installer.sha256(outside)+"  ../outside\n")
            with self.assertRaises(ValueError):installer.verify_package(root)

    def test_archive_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/"bad.tar.zst"
            archive_at(archive,{"../escaped":b"bad"})
            with self.assertRaises(tarfile.TarError):
                installer.unpack(archive,root/"runtime",CATALOG["packages"]["linux"]["sm86"])
            self.assertFalse((root/"escaped").exists())
            self.assertFalse((root/"runtime").exists())


class IntegrationTests(unittest.TestCase):
    def test_install_reuse_launcher_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory(prefix="l0xre fixtures '") as temp:
            root=Path(temp);install=root/"installation";models=root/"models with spaces '";models.mkdir()
            catalog=copy.deepcopy(CATALOG)
            for key in ("target","draft","vision"):
                path=models/catalog["models"][key]["filename"];path.write_bytes(key.encode())
                catalog["models"][key]=record(path)
            stub=b'import json,os,sys;print(json.dumps({"argv":sys.argv[1:],"gpu":os.environ["CUDA_VISIBLE_DEVICES"]}))\n'
            checksum=hashlib.sha256(stub).hexdigest()+"  tools/universal/r6-launch.py\n"
            archive=root/"fixture.tar.zst"
            archive_at(archive,{"package/tools/universal/r6-launch.py":stub,"package/SHA256SUMS":checksum.encode()})
            package=catalog["packages"]["linux"]["sm86"];package.update(record(archive));package["install_reserve_bytes"]=1000
            config=root/"catalog.json";config.write_text(json.dumps(catalog))
            args=["--catalog",str(config),"--dir",str(install),"--models-dir",str(models),"--runtime-archive",str(archive),"--yes","--vision","--port","8190"]
            with patch.object(installer,"detect_gpu",return_value=gpu()),patch.object(installer.shutil,"which",return_value="fixture-dependency"),patch.object(installer.os,"sched_getaffinity",return_value=set(range(8))):
                self.assertEqual(installer.main(args),0)
                self.assertEqual(installer.main(args),0)
                result=json.loads(subprocess.check_output([str(install/"start.sh"),"--dry-run"],text=True))
                self.assertEqual(result["gpu"],"GPU-fixture")
                self.assertIn(str(models/"L0xRE-27b-Low.gguf"),result["argv"])
                self.assertIn("--mmproj",result["argv"])
                self.assertIn("8190",result["argv"])
                self.assertNotIn("--qualification-probe",result["argv"])
                receipt=json.loads((install/"INSTALLATION.json").read_text())
                payload=Path(receipt["runtime"])/"tools/universal/r6-launch.py";payload.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError,"checksum mismatch"):
                    installer.main(args)


if __name__ == "__main__":
    unittest.main()
