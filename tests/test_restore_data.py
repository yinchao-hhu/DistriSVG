import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

spec=importlib.util.spec_from_file_location('restore_data',Path(__file__).resolve().parents[1]/'scripts/restore_data.py')
restore=importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)

class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.base=Path(self.temp.name)
        self.root=self.base/'repo';self.root.mkdir()
        self.downloads=self.base/'downloads';self.downloads.mkdir()
        self.old_root=restore.ROOT;restore.ROOT=self.root

    def tearDown(self):
        restore.ROOT=self.old_root
        self.temp.cleanup()

    def make_archive(self,name='data/value.csv'):
        value=b'x,y\n1,2\n';path=self.downloads/'pdac.zip'
        text=io.StringIO();writer=csv.DictWriter(text,fieldnames=['path','bytes','sha256']);writer.writeheader()
        writer.writerow({'path':name,'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()})
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr(name,value);z.writestr('_release/FILES.csv',text.getvalue())
        (self.root/'provenance').mkdir()
        (self.root/'provenance/DATA_PACKAGES.json').write_text(json.dumps([{'archive':'pdac.zip','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]))
        return path,value

    def split(self,path):
        raw=path.read_bytes();cut=len(raw)//2
        path.with_name(path.name+'.part-001').write_bytes(raw[:cut])
        path.with_name(path.name+'.part-002').write_bytes(raw[cut:])
        path.unlink()
        return raw

    def test_join_and_restore(self):
        path,value=self.make_archive();raw=self.split(path)
        restore.join_parts(self.downloads)
        self.assertEqual(path.read_bytes(),raw)
        restore.restore(path)
        self.assertEqual((self.root/'data/value.csv').read_bytes(),value)

    def test_truncated_parts_fail(self):
        path,_=self.make_archive();self.split(path)
        path.with_name(path.name+'.part-002').unlink()
        with self.assertRaises(ValueError):restore.join_parts(self.downloads)
        self.assertFalse(path.exists())
        self.assertFalse(path.with_suffix('.joining').exists())

    def test_gap_in_parts_fails(self):
        path,_=self.make_archive();self.split(path)
        path.with_name(path.name+'.part-002').rename(path.with_name(path.name+'.part-003'))
        with self.assertRaises(ValueError):restore.join_parts(self.downloads)

    def test_existing_different_file_is_not_overwritten(self):
        path,_=self.make_archive();target=self.root/'data/value.csv';target.parent.mkdir();target.write_text('private')
        with self.assertRaises(FileExistsError):restore.restore(path)
        self.assertEqual(target.read_text(),'private')

    def test_parent_traversal_rejected(self):
        path,_=self.make_archive('../outside.csv')
        with self.assertRaises(ValueError):restore.restore(path)
        self.assertFalse((self.base/'outside.csv').exists())

    def test_symlink_escape_rejected(self):
        path,_=self.make_archive();outside=self.base/'outside';outside.mkdir()
        (self.root/'data').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):restore.restore(path)
        self.assertFalse((outside/'value.csv').exists())

    def adapter_source(self):
        source=self.root/'analysis/adapters/competitors/run_method.R'
        source.parent.mkdir(parents=True)
        source.write_text('x <- 1\n')
        return source

    def test_compatibility_is_idempotent_and_copies_adapters(self):
        source=self.adapter_source()
        restore.setup_compatibility()
        restore.setup_compatibility()
        alias=self.root/'04_数值实验/overleaf'
        self.assertEqual(alias.resolve(),(self.root/'05_论文草稿/latex').resolve())
        target=self.root/'04_数值实验/results/distrisvg_completion_2026-09-25/official_competitors/adapters/run_method.R'
        self.assertFalse(target.is_symlink())
        self.assertEqual(target.read_bytes(),source.read_bytes())
        target.write_text('x <- 2\n')
        self.assertEqual(source.read_text(),'x <- 1\n')
        with self.assertRaises(FileExistsError):restore.setup_compatibility()

    def test_wrong_figure_alias_rejected(self):
        parent=self.root/'04_数值实验';parent.mkdir()
        outside=self.base/'outside';outside.mkdir()
        (parent/'overleaf').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):restore.setup_compatibility()

    def test_adapter_symlink_escape_rejected(self):
        self.adapter_source()
        outside=self.base/'outside';outside.mkdir()
        parent=self.root/'04_数值实验/results';parent.parent.mkdir(parents=True)
        parent.symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):restore.setup_compatibility()
        self.assertEqual(list(outside.iterdir()),[])

    def test_figure_target_symlink_escape_rejected(self):
        outside=self.base/'outside';outside.mkdir()
        (self.root/'05_论文草稿').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):restore.setup_compatibility()
        self.assertEqual(list(outside.iterdir()),[])

if __name__=='__main__':unittest.main(verbosity=2)
