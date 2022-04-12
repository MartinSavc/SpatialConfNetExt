import unittest
import numpy as np

import commonlib.cephdataloaders as cdl
import commonlib.cephdatamodifiers as cdm

class RandFunGeneratorTests(unittest.TestCase):
    pass

class NormalizeIntensityModifierTests(unittest.TestCase):
    def test_normalize_data(self):
        class BaseLoader:
            def __init__(self):
                self.imgs = [
                        np.array((1, 10, 9, 8)),
                        np.array((1.4, 0.2, 4.7)),
                        ]
                self.pts = [
                        [(1,0), (0,1), (0,0)],
                        [(0,1), (1,0), (0,0)],
                        ]

            def __len__(self):
                return 2

            def __getitem__(self, ind):
                return self.imgs[ind], [self.pts[ind]]

        base_loader = BaseLoader()
        norm_int_modifier = cdm.NormalizeIntensityModifier(base_loader, 0.5, -1)

        d1, _ = norm_int_modifier[0]
        d2, _ = norm_int_modifier[1]

        np.testing.assert_array_almost_equal(d1, np.array((-0.5, 4, 3.5, 3)))
        np.testing.assert_array_almost_equal(d2, np.array((-0.3, -0.9, 1.35)))
        

class CephDataBlackLevelModTests(unittest.TestCase):
    pass

class CephDataIntensityModTests(unittest.TestCase):
    pass

class CephDataGammaModTests(unittest.TestCase):
    pass

class CephDataRotateScaleModTests(unittest.TestCase):
    pass
    #def test_invert_pts_modification(self):
    #    raise Exception('no test implemented yet')
    #    class BaseLoader:
    #        def __len__(self):
    #            return 5

    #        def __getitem__(self, ind):
    #            return np.zeros((1, 30, 30, 1)), [pts_ref[ind]]

    #        def invert_pts_modification(self, ind, pts):
    #            return pts

    #    base_loader = BaseLoader()

    #    rot_scale_mod = cdm.CephDataRotateScaleMod(
    #            base_loader,
    #            <scale_rand>,
    #            <angle_rand>,
    #            )

    #    rot_scale_mod.invert_pts_modification(0, 


class CephDataTranslateModTests(unittest.TestCase):
    pass
    #def test_invert_pts_modification(self):
    #    raise Exception('no test implemented yet')

class PointResampleDataModifierTests(unittest.TestCase):
    pass
    #def test_invert_pts_modification(self):
    #    raise Exception('no test implemented yet')

class SubselectDataModifierTests(unittest.TestCase):
    pass
    #def test_invert_pts_modification(self):
    #    raise Exception('no test implemented yet')

class CacheDataModifierTests(unittest.TestCase):
    def test_cache_data(self):
        class BaseLoader:
            def __init__(self):
                self.imgs = [
                        np.zeros((1, 40, 60, 1)),
                        np.ones((1, 90, 80, 1)),
                        ]
                self.pts = [
                        [(1,0), (0,1), (0,0)],
                        [(0,1), (1,0), (0,0)],
                        ]

                self.access_el = [False, False]

            def was_el_accessed(self, ind):
                r = self.access_el[ind]
                self.access_el = [False, False]
                return r

            def __len__(self):
                return 4

            def __getitem__(self, ind):
                self.access_el[ind] = True
                return self.imgs[ind], [self.pts[ind]]


        base_loader = BaseLoader()
        cache_modifier = cdm.CacheDataModifier(base_loader)

        data_0_ref  = base_loader[0]
        data_1_ref  = base_loader[1]
        base_loader.was_el_accessed(0)

        data_cache = cache_modifier[0]
        self.assertTrue(base_loader.was_el_accessed(0))
        np.testing.assert_array_equal(data_0_ref[0], data_cache[0])
        self.assertEqual(data_0_ref[1], data_cache[1])

        data_cache = cache_modifier[1]
        self.assertTrue(base_loader.was_el_accessed(1))
        np.testing.assert_array_equal(data_1_ref[0], data_cache[0])
        self.assertEqual(data_1_ref[1], data_cache[1])

        data_cache = cache_modifier[1]
        self.assertFalse(base_loader.was_el_accessed(1))
        np.testing.assert_array_equal(data_1_ref[0], data_cache[0])
        self.assertEqual(data_1_ref[1], data_cache[1])

        data_cache = cache_modifier[0]
        self.assertFalse(base_loader.was_el_accessed(0))
        np.testing.assert_array_equal(data_0_ref[0], data_cache[0])
        self.assertEqual(data_0_ref[1], data_cache[1])

        data_cache[0][0,0] = 10
        data_cache[1][0][1] = None 
        self.assertNotEqual(data_0_ref[1], data_cache[1])


class ShuffleDataModifierTests(unittest.TestCase):
    def test_shuffle_data(self):
        class BaseLoader:


            def __len__(self):
                return 10

            def __getitem__(self, ind):
                return ind


        base_loader = BaseLoader()
        shuffle_modifier = cdm.ShuffleDataModifier(base_loader)

        ref_ind = [i for i in range(10)]

        ret_ind_1 = [shuffle_modifier[i] for i in range(10)]
        shuffle_modifier.generate_rand_values()
        ret_ind_2 = [shuffle_modifier[i] for i in range(10)]

        self.assertNotEqual(ref_ind, ret_ind_1)
        self.assertNotEqual(ret_ind_1, ret_ind_2)
