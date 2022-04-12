import numpy as np
import unittest
import os

from commonlib.resize_and_labels_fun import *

class RotateAndScaleTest(unittest.TestCase):

    def test_1_rotate_and_scale(self):
        ''' test the rotate_and_scale function using hand written test data
        '''
        # 5x5 px image, vertical line
        imTest = np.array(
                [[0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0]],
                dtype=np.float32)

        # 3 points. fist at center
        # point(y,x) or (row_ind, col_ind)
        # points are originally row vectors (function converts it to row vector)
        pointsTest = np.array(
                [[2, 2],
                 [2, 1],
                 [1, 1]],
                dtype=np.float64)
        pointsTest = pointsTest.transpose()
        d, f = rotate_and_scale_im_points(imTest, pointsTest, 90, 1)

        # predvideni rezultati
        d_r = np.array(
           [[ 0.,  0.,  0.,  0.,  0.],
            [ 0., -0., -0., -0., -0.],
            [ 1.,  1.,  1.,  1.,  1.],
            [ 0., -0., -0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.]],
           dtype=np.float32)
        f_r = np.array(
            [[2., 3., 3.],
             [2., 2., 1.]])
        # če je vse ok, ni izpisa...
        np.testing.assert_array_almost_equal(d, d_r)
        np.testing.assert_array_almost_equal(f, f_r)

        # primer 4x4 matrike, podobno kot prej
        imTest2 = np.array(
                [[0, 0, 1, 0],
                 [0, 0, 1, 0],
                 [0, 0, 1, 0],
                 [0, 0, 1, 0]],
                dtype=np.float32)
        # 3 points. fist at center
        # point(y,x) or (row_ind, col_ind)
        # points are originally row vectors (function converts it to row vector)
        pointsTest = np.array(
                [[2, 2],
                 [2, 1],
                 [1, 1]],
                dtype=np.float64)
        pointsTest = pointsTest.transpose()
        g, h = rotate_and_scale_im_points(imTest2, pointsTest, 90, 1)

        g_r = np.array(
           [[ 0., -0., -0.,  0.],
            [ 1.,  1.,  1.,  1.],
            [ 0., -0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.]],
           dtype=np.float32)
        h_r = np.array(
           [[1., 2., 2.],
            [2., 2., 1.]])
        np.testing.assert_array_almost_equal(g, g_r)
        np.testing.assert_array_almost_equal(h, h_r)


class getPointsFileFromImgTest(unittest.TestCase):
    def test_1_png(self):
        # rabimo curr dir in 2 višje
        # ./podatki-hdd/gsedej/Delo/CephBot/KPnet-CephBot/21_DS06/KPnet
        # ./podatki-hdd/gsedej/Delo/CephBot/KPnet-CephBot/21_DS06/2_labels/slika.png-points.txt

        fnameImageOrig = '1_images/slika.png'
        expected = os.path.join(os.getcwd(), "2_labels/slika.png-points.txt")
        result = getPointsFileFromImg(fnameImageOrig)
        #print(f'result: {result}')
        self.assertEqual(result, expected)

    def test_2(self):
        fnameImageOrig = 'slika.png'
        expected = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "2_labels/slika.png-points.txt"))
        result = getPointsFileFromImg(fnameImageOrig)
        #print(f'result: {result}')
        self.assertEqual(result, expected)

    def test_3_jpg(self):
        fnameImageOrig = '1_images/slika.jpg'
        expected = os.path.join(os.getcwd(), "2_labels/slika.jpg-points.txt")
        result = getPointsFileFromImg(fnameImageOrig)
        self.assertEqual(result, expected)

    def test_4_extension(self):
        fnameImageOrig = '1_images/slika.jpg'
        extension = '-pnts.txt'
        expected = os.path.join(os.getcwd(), "2_labels/slika-pnts.txt")
        result = getPointsFileFromImg(fnameImageOrig, extension=extension)
        self.assertEqual(result, expected)

    def test_5_txtDirName(self):
        fnameImageOrig = '1_images/slika.png'
        txtDirName = 'labele'
        expected = os.path.join(os.getcwd(), "labele/slika.png-points.txt")
        result = getPointsFileFromImg(fnameImageOrig, txtDirName=txtDirName)
        self.assertEqual(result, expected)


class readPointsInTxtTest(unittest.TestCase):
    fnameTxtFull = './tests_data/commonlib/image0001.jpg-points.txt'
    fnameTxtShortSpace = './tests_data/commonlib/im00.jpg-points-space.txt'
    fnameTxtShortNoHead = './tests_data/commonlib/im00.jpg-points-noHead.txt'
    byListFname = './tests_data/commonlib/landmarks1-important.txt'

    expectedFull = np.array(
        [[2391., 3207., 3697., 1678., 1326., 2636., 3232., 2570., 3022.,
        2551., 2950., 3328., 3693., 1935., 3344., 3518., 3159., 2519.,
        2533., 2922., 1426., 3540., 1928., 3507., 2395., 2037., 2820.,
        2566., 2512., 3342., 3120., 2310., 2687., 2900., 3276., 2095.,
        3009., 3972., 1075., 2962., 3000., 2964., 3871., 3124., 1579.,
        2855., 2912., 2940., 2794., 2582., 3328., 3825., 1604., 2988.,
        2449., 2745., 2061., 2170., 1160., 1631., 2844., 1522., 1544.,
        1644., 1678., 1780., 1545., 3734., 1858., 1056., 2838., 1533.],
        [2397., 2456., 4024., 2700., 4138., 4149., 4041., 2274., 4168.,
        4040., 4091., 3958., 4089., 3912., 2937., 4135., 2516., 3208.,
        4299., 3524., 4249., 4402., 3035., 4018., 2538., 2216., 4549.,
        4477., 4576., 4054., 4472., 4787., 4423., 4199., 4245., 2345.,
        4104., 3693., 4275., 4369., 3411., 4369., 4052., 4094., 3746.,
        4198., 3544., 3536., 2465., 3627., 3473., 4207., 2702., 4128.,
        2878., 3099., 2431., 2289., 4030., 4273., 3438., 4042., 3268.,
        2809., 2599., 2692., 2851., 2980., 4542., 3760., 3120., 3567.]])

    expectedByList_important = np.array(
        [[1678., 1935., 1326., 2037., 1928., 2570., 2391., 2519., 2533.,
        2636., 3232., 3518., 3693., 3697., 3207., 3159., 3022., 2551.,
        2950., 3328., 2912., 2582., 2940., 3328., 2922., 2988., 1075.,
        1426., 2310., 2566., 2687., 2820., 2962., 2964., 3120., 3276.,
        3540., 3871.],
        [2700., 3912., 4138., 2216., 3035., 2274., 2397., 3208., 4299.,
        4149., 4041., 4135., 4089., 4024., 2456., 2516., 4168., 4040.,
        4091., 3958., 3544., 3627., 3536., 3473., 3524., 4128., 4275.,
        4249., 4787., 4477., 4423., 4549., 4369., 4369., 4472., 4245.,
        4402., 4052.]])

    expectedShort = np.array([
        [2391., 3207., 3697.],
        [2397., 2456., 4024.]])

    def test_0_test_files_exits(self):
        self.assertTrue(os.path.exists(self.fnameTxtFull), msg='cant find data file')
        self.assertTrue(os.path.exists(self.fnameTxtShortSpace), msg='cant find data file')
        self.assertTrue(os.path.exists(self.fnameTxtShortNoHead), msg='cant find data file')
        self.assertTrue(os.path.exists(self.byListFname), msg='cant find data file')
        

    def test_1_normal(self):
        result = readPointsInTxt(self.fnameTxtFull, csvChar=',', pointCount=None)
        np.testing.assert_array_equal(result, self.expectedFull)
        
    def test_2_noHead(self):
        result = readPointsInTxt(self.fnameTxtShortNoHead, csvChar=',', pointCount=3, firstLineMetadata=False)
        np.testing.assert_array_equal(result, self.expectedShort)
    
    def test_3_space(self):
        result = readPointsInTxt(self.fnameTxtShortSpace, csvChar=' ', pointCount=3)
        np.testing.assert_array_equal(result, self.expectedShort)
        
    def test_4_keypointList(self):
        result = readPointsInTxt(self.fnameTxtFull, csvChar=',', pointCount=None, byListFname=self.byListFname)
        np.testing.assert_array_equal(result, self.expectedByList_important)
