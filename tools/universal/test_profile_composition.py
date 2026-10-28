"""Prevent the missing16GB correctness setting from entering a staged package."""
from pathlib import Path
import hashlib,json,subprocess,sys,tempfile,unittest

class ProfileComposition(unittest.TestCase):
    def test_generated_bytecode_never_enters_the_integrity_manifest(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);base=root/'base';arch=base/'architectures/sm86'
            arch.mkdir(parents=True);cache=base/'tools/universal/__pycache__';cache.mkdir(parents=True)
            (cache/'v01-launch.cpython-314.pyc').write_bytes(b'changed generated cache')
            (arch/'V0.1-MANIFEST.json').write_text(json.dumps(dict(
                runtime_sha256={'tools/universal/__pycache__/v01-launch.cpython-314.pyc':'stale-generated-hash'},loader_links={})))
            profile=root/'12.json';profile.write_text(json.dumps(dict(argv=['server','-c','131072'],env={},model_sha256={'target':'same-model'})))
            output=root/'staged'
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('compose-memory-profiles.py')),
                '--base',str(base),'--arch','sm86','--profile-12gb',str(profile),'--output',str(output)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            manifest=json.loads((output/'architectures/sm86/V0.1-MANIFEST.json').read_text())
            self.assertFalse((output/'tools/universal/__pycache__').exists())
            self.assertFalse(any('__pycache__' in p or p.endswith('.pyc') for p in manifest['runtime_sha256']))
    def test_sm86_can_stage_only_its_validated_12gb_preset(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);base=root/'base';arch=base/'architectures/sm86'
            arch.mkdir(parents=True);(base/'tools/universal').mkdir(parents=True)
            (arch/'V0.1-MANIFEST.json').write_text(json.dumps(dict(runtime_sha256={},loader_links={})))
            profile=root/'12.json';profile.write_text(json.dumps(dict(argv=['server','-c','131072'],env={},model_sha256={'target':'same-model'})))
            out=root/'staged'
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('compose-memory-profiles.py')),
                '--base',str(base),'--arch','sm86','--profile-12gb',str(profile),'--output',str(out)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            manifest=json.loads((out/'architectures/sm86/V0.1-MANIFEST.json').read_text())
            self.assertEqual(set(manifest['memory_profiles']),{'12gb'})
            self.assertFalse(manifest['hardware_qualified'])

    def test_smaller_context_requires_an_explicit_fallback_declaration(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);base=root/'base';arch=base/'architectures/sm86'
            arch.mkdir(parents=True);(base/'tools/universal').mkdir(parents=True)
            (arch/'V0.1-MANIFEST.json').write_text(json.dumps(dict(runtime_sha256={},loader_links={})))
            profile=root/'12.json';profile.write_text(json.dumps(dict(argv=['server','-c','65536'],env={},model_sha256={'target':'same-model'})))
            args=[sys.executable,str(Path(__file__).with_name('compose-memory-profiles.py')),
                '--base',str(base),'--arch','sm86','--profile-12gb',str(profile),'--output',str(root/'staged')]
            result=subprocess.run(args,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('wrong context capacity',result.stderr)
            self.assertFalse((root/'staged').exists())
            result=subprocess.run([*args,'--context-12gb','65536'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            config=json.loads((root/'staged/architectures/sm86/profiles/12gb.json').read_text())
            self.assertEqual(config['context_tokens'],65536)

    def test_required_head_setting_covers_both_presets_before_staging(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);base=root/'base';arch=base/'architectures/sm120'
            arch.mkdir(parents=True);(base/'tools/universal').mkdir(parents=True)
            manifest={'runtime_sha256':{},'loader_links':{},
                      'required_profile_env':{'L0XRE_HEAD_BATCH_INVARIANT':'1'}}
            (arch/'V0.1-MANIFEST.json').write_text(json.dumps(manifest))
            p12=root/'12.json';p16=root/'16.json'
            common={'model_sha256':{'target':'same-model'},'env':{'L0XRE_HEAD_BATCH_INVARIANT':'1'}}
            p12.write_text(json.dumps(dict(common,argv=['server','-c','131072'])))
            p16.write_text(json.dumps(dict(common,argv=['server','-c','262144'],env={})))
            before=hashlib.sha256(p16.read_bytes()).hexdigest();out=root/'staged'
            args=[sys.executable,str(Path(__file__).with_name('compose-memory-profiles.py')),
                  '--base',str(base),'--arch','sm120','--profile-12gb',str(p12),
                  '--profile-16gb',str(p16),'--output',str(out)]
            failed=subprocess.run(args,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertIn('16gb must set L0XRE_HEAD_BATCH_INVARIANT=1',failed.stderr)
            self.assertFalse(out.exists())
            self.assertEqual(hashlib.sha256(p16.read_bytes()).hexdigest(),before)
            p16.write_text(json.dumps(dict(common,argv=['server','-c','262144'])))
            done=subprocess.run(args,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            staged=json.loads((out/'architectures/sm120/V0.1-MANIFEST.json').read_text())
            self.assertEqual(staged['required_profile_env'],manifest['required_profile_env'])
            self.assertFalse(staged['promoted'])
            for entry in staged['memory_profiles'].values():
                path=out/entry['path']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),entry['sha256'])

if __name__=='__main__':unittest.main()
