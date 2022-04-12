import unittest
import numpy as np

import commonlib.cephdataloaders as cdl


class CephDataLoaderTests(unittest.TestCase):
    pass

class CephDataWrapperTests(unittest.TestCase):
    pass

class DataResizeLoaderTests(unittest.TestCase):
    def test_invert_pts_modification(self):
        pts = [[(9, 0), (0, 9), (6, 7.2)]]
        pts_inv_1_ref = [[(18, 0), (0, 18), (12, 14.4)]]
        pts_inv_2_ref = [[(27, 0), (0, 27), (18, 21.6)]]
        target_size = (30, 30)
        class BaseLoader:
            def __init__(self):
                self.imgs = [np.zeros((1, 40, 60, 1)), np.zeros((1, 90, 80, 1))]
                self.pts = [[(0,0), (0,0), (0,0)], [(0,0), (0,0), (0,0)]]

            def __len__(self):
                return 2

            def __getitem__(self, ind):
                return self.imgs[ind], [self.pts[ind]]

            def invert_pts_modification(self, ind, pts):
                return pts

        base_loader = BaseLoader()

        resize_loader = cdl.DataResizeLoader(base_loader, target_size)

        pts_inv_1 = resize_loader.invert_pts_modification(0, pts)

        self.assertEqual(pts_inv_1, pts_inv_1_ref)

        pts_inv_2 = resize_loader.invert_pts_modification(1, pts)
        self.assertEqual(pts_inv_2, pts_inv_2_ref)


class KeypointMapDataLoaderTests(unittest.TestCase):
    pass

class KeypointWeightsDataLoaderTests(unittest.TestCase):
    pass

class DataBatchLoaderTests(unittest.TestCase):
    def test_invert_pts_modification(self):

        pts = [
                [(0, 0)],
                [(0, 0)],
              ]
                
        pts_ref = [
                [(0,9), (0,1), (2,0)],
                [(1,8), (1,0), (9,1)],
                [(2,7), (0,1), (3,8)],
                [(3,6), (1,0), (7,4)],
                [(4,5), (0,1), (5,6)],
                ]

        class BaseLoader:
            def __len__(self):
                return 5

            def __getitem__(self, ind):
                return np.zeros((1, 30, 30, 1)), [pts_ref[ind]]

            def invert_pts_modification(self, ind, pts):
                return [pts_ref[ind]]

        base_loader = BaseLoader()

        batch_loader = cdl.DataBatchLoader(base_loader, 2)

        pts_inv_1 = batch_loader.invert_pts_modification(0, pts)
        self.assertEqual(pts_inv_1, pts_ref[:2])

        pts_inv_2 = batch_loader.invert_pts_modification(1, pts)
        self.assertEqual(pts_inv_2, pts_ref[2:4])


class PointSelectionLoaderTests(unittest.TestCase):
    def test_invert_pts_modification(self):
        pts_1 = [[(1, 2), (3, 4)]]
        pts_2 = [[(5, 6), (7, 8)]]
                
        pts_ref_1 = [[(0, 0), (3, 4), (0, 0), (1, 2)]]
        pts_ref_2 = [[(0, 0), (7, 8), (0, 0), (5, 6)]]

        class BaseLoader:
            def __len__(self):
                return 2

            def __getitem__(self, ind):
                return np.zeros((1, 30, 30, 1)), [[(0, 0), (0, 0), (0, 0), (0, 0)]]

            def invert_pts_modification(self, ind, pts):
                return pts

        base_loader = BaseLoader()

        point_sel_loader = cdl.PointSelectionLoader(base_loader, (3, 1))

        pts_inv_1 = point_sel_loader.invert_pts_modification(0, pts_1)
        self.assertEqual(pts_inv_1, pts_ref_1)

        pts_inv_2 = point_sel_loader.invert_pts_modification(1, pts_2)
        self.assertEqual(pts_inv_2, pts_ref_2)

