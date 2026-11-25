import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('ghrb_sources',ROOT/'scripts/datasets/prepare_ghrb_sources.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


class SourceAcquisitionTests(unittest.TestCase):
    def test_verified_rename_keeps_archive_and_materializes_only_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);commit='a'*40;folder=root/'org/old'/commit;folder.mkdir(parents=True)
            with zipfile.ZipFile(folder/'snapshot.zip','w') as z:
                z.writestr('new-'+commit+'/src/Main.java','class Main {}')
                z.writestr('new-'+commit+'/pom.xml','<project/>')
            result=M.fetch('org/old',commit,'org/new',root)
            self.assertEqual(result['java_files'],1)
            self.assertFalse((folder/'source/pom.xml').exists())
            self.assertTrue((folder/'snapshot.zip').exists())
            self.assertEqual(M.fetch('org/old',commit,'org/new',root),result)
            (folder/'source/src/Main.java').write_text('changed')
            with self.assertRaises(ValueError):M.fetch('org/old',commit,'org/new',root)

    def test_archive_traversal_is_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);commit='b'*40;folder=root/'org/repo'/commit;folder.mkdir(parents=True)
            with zipfile.ZipFile(folder/'snapshot.zip','w') as z:z.writestr('repo-'+commit+'/../Bad.java','class Bad {}')
            with self.assertRaises(ValueError):M.fetch('org/repo',commit,'org/repo',root)
            self.assertFalse((folder/'source').exists())


if __name__=='__main__':unittest.main()
