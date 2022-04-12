import unittest
import tempfile
import os

import commonlib.filelist as filelist


class FileListTests(unittest.TestCase):

    def test_1_write_and_read_file_list(self):
        tmp_file_handle, tmp_file_path = tempfile.mkstemp()
        os.close(tmp_file_handle)

        files_list = [
            './tests_data/commonlib/test_files/file_1.txt',
            './tests_data/commonlib/test_files/file_2.txt',
            ]
        filelist.write_file_list(
            tmp_file_path,
            files_list)

        os.chdir('./tests_data/')

        try:
            file_list = filelist.read_file_list(
                tmp_file_path,
                relative_paths=True
                )

            for file_path in file_list:
                self.assertTrue(
                    os.path.exists(file_path),
                    msg=f'file path {file_path} is not valid',
                    )
        finally:
            # test clean up
            os.chdir('../')
            os.remove(tmp_file_path)


    def test_2_read_file_list(self):
        file_list = filelist.read_file_list(
            'tests_data/commonlib/ref_files.list'
            )
        for file_path in file_list:
            self.assertTrue(
                os.path.exists(file_path),
                msg=f'file path {file_path} is not valid',
                )

    def test_3_compare_file_lists(self):
        sources = ['a.ext', 'b.ext', 'c.ext', 'd.ext', 'c.ext']
        targets = ['./d.ext', 'b.ext']
        inds_ref = [3, 1]
        inds = filelist.compare_file_lists(sources, targets)
        self.assertEqual(inds_ref, inds)

        targets = ['c.ext']
        inds_ref = [2]
        inds = filelist.compare_file_lists(sources, targets)
        self.assertEqual(inds_ref, inds)

        targets = ['folder/d.ext']
        inds_ref = [3]
        inds = filelist.compare_file_lists(sources, targets, only_file_name=True)
        self.assertEqual(inds_ref, inds)

        targets = ['g.ext']
        inds_ref = [None]
        inds = filelist.compare_file_lists(sources, targets, only_file_name=True)
        self.assertEqual(inds_ref, inds)
if __name__ == '__main__':
    unittest.main()
