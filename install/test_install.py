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
        self.assertTrue(installer.make_plan(CATALOG, blackwell)["hardware_qualified_for_gpu"])
        pending = copy.deepcopy(CATALOG)
        pending["packages"]["linux"]["sm120"]["url"] = None
        with self.assertRaisesRegex(ValueError, "not published"):
            installer.make_plan(pending, blackwell)
        self.assertEqual(installer.make_plan(pending, blackwell, runtime_archive="sealed.tar.zst")["package"]["status"], "certified")

    def test_candidate_needs_opt_in(self):
        ada = gpu("sm89", "NVIDIA GeForce RTX 4090", 24564)
        with self.assertRaisesRegex(ValueError, "allow-candidate"):
            installer.make_plan(CATALOG, ada)
        self.assertEqual(installer.make_plan(CATALOG, ada, True)["package"]["status"], "candidate")

    def test_card_scope_and_vram(self):
        for device in [gpu(memory=8192), gpu("sm75"), gpu("sm120", "RTX 4080", 32768), gpu("sm89", "RTX 4060", 8192)]:
            with self.assertRaises(ValueError):
                installer.make_plan(CATALOG, device, True, "sealed.tar.zst")
        for arch, name, memory in [("sm86", "RTX 3090", 24576), ("sm89", "RTX 4070 Ti", 12288), ("sm89", "RTX 4060 Ti", 16384), ("sm120", "RTX 5070", 12288), ("sm120", "RTX 5060 Ti", 16384)]:
            plan = installer.make_plan(CATALOG, gpu(arch, name, memory), True)
            self.assertFalse(plan["hardware_qualified_for_gpu"])
            if arch == "sm120":
                self.assertIn("12gb", plan["package"]["filename"])

    def test_dual_retains_profile_and_fast_routes(self):
        cards = [gpu(), {**gpu(), "index": 1, "uuid": "GPU-second"}]
        with patch.object(installer, "detect_gpu", side_effect=cards):
            plan = installer.make_dual_plan(CATALOG, "0,1", True)
        self.assertFalse(plan["hardware_qualified_for_gpu"])
        for devices in [[cards[0], cards[0]], [cards[0], {**cards[1], "memory_mib": 8192}], [cards[0], {**cards[1], "name": "RTX 3090"}]]:
            with patch.object(installer, "detect_gpu", side_effect=devices), self.assertRaises(ValueError):
                installer.make_dual_plan(CATALOG, "0,1", True)
        profile = {"argv": ["@ARCH_ROOT@/bin/llama-server", "-c", "81920", "--spec-draft-n-max", "7", "--port", "8080"],
                   "env": {"ESCHA_OFFICIAL_RAW_BRIDGE": "1", "L0XRE_INT8_PREFILL": "1", "LD_PRELOAD": "@ARCH_ROOT@/bin/libqk16-context-gate.so:@ARCH_ROOT@/bin/libbridge-packed.so"}}
        spec = importlib.util.spec_from_file_location("dual", SOURCE.with_name("dual-launch.py"))
        dual = importlib.util.module_from_spec(spec); spec.loader.exec_module(dual)
        receipt = {**plan, "runtime": "/runtime", "dual_addon_root": "/addon", "port": 8190, "models_dir": "/models", "vision": False}
        with patch.dict(os.environ, {"ESCHA_UNKNOWN": "1", "LD_PRELOAD": "bad.so"}):
            args, env = dual.command_for(receipt, profile)
        self.assertEqual(args[args.index("-c")+1], "81920")
        self.assertEqual(args[args.index("--spec-draft-n-max")+1], "7")
        self.assertEqual(args[args.index("--tensor-split")+1], "1,1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-fixture,GPU-second")
        self.assertEqual(env["ESCHA_OFFICIAL_RAW_BRIDGE"], "1")
        self.assertEqual(env["L0XRE_INT8_PREFILL"], "1")
        self.assertIn("/addon/sm86/libbridge-dual.so", env["LD_PRELOAD"])
        self.assertNotIn("libbridge-packed.so", env["LD_PRELOAD"])
        self.assertNotIn("ESCHA_UNKNOWN", env)

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
    def test_update_preferences_and_verified_rollback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old, new = root / "old", root / "new"
            for runtime in (old, new):
                runtime.mkdir()
                (runtime / "payload").write_text(runtime.name)
                (runtime / "SHA256SUMS").write_text(installer.sha256(runtime / "payload") + "  payload\n")
            models = root / "models elsewhere"
            plan = installer.make_plan(CATALOG, gpu())
            installer.write_launcher(root, old, models, plan, True, 8191)
            installer.write_launcher(root, new, models, plan, True, 8191)
            self.assertTrue((root / "update.sh").is_file())
            with patch.object(installer, "load_catalog", return_value=CATALOG), patch.object(installer, "detect_gpu", return_value=gpu()) as detect:
                self.assertEqual(installer.main(["--dir", str(root), "--update", "--catalog", "fixture", "--dry-run"]), 0)
                detect.assert_called_once_with("GPU-fixture")
            self.assertEqual(installer.main(["--dir", str(root), "--rollback", "--dry-run"]), 0)
            self.assertEqual(json.loads((root / "INSTALLATION.json").read_text())["runtime"], str(new))
            self.assertEqual(installer.main(["--dir", str(root), "--rollback"]), 0)
            receipt = json.loads((root / "INSTALLATION.json").read_text())
            self.assertEqual((receipt["runtime"], receipt["models_dir"], receipt["port"], receipt["vision"]), (str(old), str(models), 8191, True))
            (old / "payload").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "integrity|checksum|SHA|Checksum"):
                installer.main(["--dir", str(root), "--rollback"])

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
    def test_dual_install_addon_launcher_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory(prefix="l0xre dual '") as temp:
            root = Path(temp); install = root / "installation"; models = root / "models"; models.mkdir()
            catalog = copy.deepcopy(CATALOG)
            for key in ("target", "draft"):
                path = models / catalog["models"][key]["filename"]
                path.write_bytes(key.encode()); catalog["models"][key] = record(path)
            profile = {"argv": ["@ARCH_ROOT@/bin/llama-server", "-c", "81920", "--spec-draft-n-max", "7", "--port", "8080"],
                       "env": {"L0XRE_INT8_PREFILL": "1", "ESCHA_OFFICIAL_RAW_BRIDGE": "1", "LD_PRELOAD": "@ARCH_ROOT@/bin/libbridge-packed.so"},
                       "model_sha256": {k: catalog["models"][k]["sha256"] for k in ("target", "draft")}}
            files = {"tools/universal/r6-launch.py": b"# fixture\n",
                     "tools/universal/r6-profile.json": json.dumps(profile).encode(),
                     "architectures/sm86/R6-MANIFEST.json": b'{"loader_links":{}}'}
            files["SHA256SUMS"] = "".join(hashlib.sha256(data).hexdigest()+"  "+name+"\n" for name, data in files.items()).encode()
            archive = root / "runtime.tar.zst"; archive_at(archive, files)
            package = catalog["packages"]["linux"]["sm86"]; package.update(record(archive)); package["install_reserve_bytes"] = 1000
            downloads = install / "downloads"; downloads.mkdir(parents=True)
            addon = downloads / "dual.tar.zst"; marker = b'{"hardware_qualified":false}'
            archive_at(addon, {"DUAL-EXPERIMENT.json": marker, "SHA256SUMS": (hashlib.sha256(marker).hexdigest()+"  DUAL-EXPERIMENT.json\n").encode()})
            catalog["dual_addons"]["linux"].update(record(addon))
            config = root / "catalog.json"; config.write_text(json.dumps(catalog))
            cards = [gpu(), {**gpu(), "index": 1, "uuid": "GPU-second"}]
            args = ["--catalog", str(config), "--dir", str(install), "--models-dir", str(models), "--runtime-archive", str(archive),
                    "--gpus", "0,1", "--allow-candidate", "--yes"]
            with patch.object(installer, "detect_gpu", side_effect=cards * 2), patch.object(installer.shutil, "which", return_value="fixture-dependency"), patch.object(installer.os, "sched_getaffinity", return_value=set(range(8))):
                self.assertEqual(installer.main(args), 0)
                self.assertEqual(installer.main(args), 0)
            fake = root / "nvidia-smi"
            fake.write_text("#!/usr/bin/env python3\nimport sys\nu=next(a[5:] for a in sys.argv if a.startswith('--id='))\nprint(u+', NVIDIA GeForce RTX 3060, 8.6, 12288')\n")
            fake.chmod(0o755)
            env = {**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"]}
            result = json.loads(subprocess.check_output([str(install / "start.sh"), "--dry-run"], env=env, text=True))
            self.assertFalse(result["hardware_qualified"])
            self.assertTrue(result["external_fast_bridges"])
            self.assertEqual(result["env"]["CUDA_VISIBLE_DEVICES"], "GPU-fixture,GPU-second")
            self.assertEqual(result["argv"][result["argv"].index("--spec-draft-n-max")+1], "7")
            receipt = json.loads((install / "INSTALLATION.json").read_text())
            (Path(receipt["dual_addon_root"]) / "DUAL-EXPERIMENT.json").write_text("tampered")
            failed = subprocess.run([str(install / "start.sh"), "--dry-run"], env=env, text=True, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("integrity mismatch", failed.stderr)

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
