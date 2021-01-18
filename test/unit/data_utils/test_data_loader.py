import itertools
import unittest
import tempfile
from shutil import rmtree
import os
import six
from keras_segmentation.data_utils import data_loader
import random
import cv2
from imgaug import augmenters as iaa
import shutil
import numpy as np

class TestGetPairsFromPaths(unittest.TestCase):
    """ Test data loader facilities """

    def _setup_images_and_segs(self, images, segs):
        for file_name in images:
            open(os.path.join(self.img_path, file_name), 'a').close()
        for file_name in segs:
            open(os.path.join(self.seg_path, file_name), 'a').close()

    @classmethod
    def _cleanup_folder(cls, folder_path):
        return rmtree(folder_path)

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.img_path = os.path.join(self.tmp_dir, "images")
        self.seg_path = os.path.join(self.tmp_dir, "segs")
        os.mkdir(self.img_path)
        os.mkdir(self.seg_path)

    def tearDown(self):
        rmtree(self.tmp_dir)

    def test_get_pairs_from_paths_1(self):
        """ Normal execution """
        images = ["A.jpg", "B.jpg", "C.jpeg", "D.png"]
        segs = ["A.png", "B.png", "C.png", "D.png"]
        self._setup_images_and_segs(images, segs)

        expected = [("A.jpg", "A.png"),
                    ("B.jpg", "B.png"),
                    ("C.jpeg", "C.png"),
                    ("D.png", "D.png")]
        expected_values = []
        # Transform paths
        for (x, y) in expected:
            expected_values.append((os.path.join(self.img_path, x), os.path.join(self.seg_path, y)))
        self.assertEqual(expected_values, sorted(data_loader.get_pairs_from_paths(self.img_path, self.seg_path)))

    def test_get_pairs_from_paths_2(self):
        """ Normal execution with extra files """
        images = ["A.jpg", "B.jpg", "C.jpeg", "D.png", "E.txt"]
        segs = ["A.png", "B.png", "C.png", "D.png", "E.png"]
        self._setup_images_and_segs(images, segs)

        expected = [("A.jpg", "A.png"),
                    ("B.jpg", "B.png"),
                    ("C.jpeg", "C.png"),
                    ("D.png", "D.png")]
        expected_values = []
        # Transform paths
        for (x, y) in expected:
            expected_values.append((os.path.join(self.img_path, x), os.path.join(self.seg_path, y)))
        self.assertEqual(expected_values, sorted(data_loader.get_pairs_from_paths(self.img_path, self.seg_path)))


    def test_get_pairs_from_paths_3(self):
        """ Normal execution with multiple images pointing to one """
        images = ["A.jpg", "B.jpg", "C.jpeg", "D.png", "D.jpg"]
        segs = ["A.png", "B.png", "C.png", "D.png"]
        self._setup_images_and_segs(images, segs)

        expected = [("A.jpg", "A.png"),
                    ("B.jpg", "B.png"),
                    ("C.jpeg", "C.png"),
                    ("D.jpg", "D.png"),
                    ("D.png", "D.png"),]
        expected_values = []
        # Transform paths
        for (x, y) in expected:
            expected_values.append((os.path.join(self.img_path, x), os.path.join(self.seg_path, y)))
        self.assertEqual(expected_values, sorted(data_loader.get_pairs_from_paths(self.img_path, self.seg_path)))

    def test_get_pairs_from_paths_with_invalid_segs(self):
        images = ["A.jpg", "B.jpg", "C.jpeg", "D.png"]
        segs = ["A.png", "B.png", "C.png", "D.png", "D.jpg"]
        self._setup_images_and_segs(images, segs)

        expected = [("A.jpg", "A.png"),
                    ("B.jpg", "B.png"),
                    ("C.jpeg", "C.png"),
                    ("D.png", "D.png"),]
        expected_values = []
        # Transform paths
        for (x, y) in expected:
            expected_values.append((os.path.join(self.img_path, x), os.path.join(self.seg_path, y)))
        self.assertEqual(expected_values, sorted(data_loader.get_pairs_from_paths(self.img_path, self.seg_path)))

    def test_get_pairs_from_paths_with_no_matching_segs(self):
        images = ["A.jpg", "B.jpg", "C.jpeg", "D.png"]
        segs = ["A.png", "B.png", "C.png"]
        self._setup_images_and_segs(images, segs)

        expected = [("A.jpg", "A.png"),
                    ("B.jpg", "B.png"),
                    ("C.jpeg", "C.png")]
        expected_values = []
        # Transform paths
        for (x, y) in expected:
            expected_values.append((os.path.join(self.img_path, x), os.path.join(self.seg_path, y)))
        six.assertRaisesRegex(self, data_loader.DataLoaderError, "No corresponding segmentation found for image", data_loader.get_pairs_from_paths, self.img_path, self.seg_path)

    def test_get_pairs_from_paths_with_no_matching_segs_with_escape(self):
        images = ["A.jpg", "B.jpg", "C.jpeg", "D.png"]
        segs = ["A.png", "B.png", "C.png"]
        self._setup_images_and_segs(images, segs)

        expected = [("A.jpg", "A.png"),
                    ("B.jpg", "B.png"),
                    ("C.jpeg", "C.png")]
        expected_values = []
        # Transform paths
        for (x, y) in expected:
            expected_values.append((os.path.join(self.img_path, x), os.path.join(self.seg_path, y)))
        self.assertEqual(expected_values, sorted(data_loader.get_pairs_from_paths(self.img_path, self.seg_path, ignore_non_matching=True)))

class TestGetImageArray(unittest.TestCase):
    def test_get_image_array_normal(self):
        """ Stub test
        TODO(divamgupta): Fill with actual test
        """
        pass

class TestGetSegmentationArray(unittest.TestCase):
    def test_get_segmentation_array_normal(self):
        """ Stub test
        TODO(divamgupta): Fill with actual test
        """
        pass

class TestVerifySegmentationDataset(unittest.TestCase):
    def test_verify_segmentation_dataset(self):
        """ Stub test
        TODO(divamgupta): Fill with actual test
        """
        pass

class TestImageSegmentationGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.train_temp_dir = tempfile.mkdtemp()
        self.test_temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.train_temp_dir)
        shutil.rmtree(self.test_temp_dir)

    def custom_aug(self):
        return iaa.Sequential(
            [
                iaa.Fliplr(1),  # horizontally flip 100% of all images
            ])

    def test_image_segmentation_generator_custom_augmentation(self):
        image_size = 4

        train_image = np.arange(image_size * image_size)
        train_image = train_image.reshape((image_size, image_size))

        train_file = os.path.join(self.train_temp_dir, "train.png")
        test_file = os.path.join(self.test_temp_dir, "train.png")

        print(self.train_temp_dir)

        cv2.imwrite(train_file, train_image)
        cv2.imwrite(test_file, train_image)

        self.assertTrue(np.equal(
            cv2.imread(train_file)[:, :, 0],
            train_image
        ).all(), "Error saving train file")
        self.assertTrue(np.equal(
            cv2.imread(test_file)[:, :, 0],
            train_image
        ).all(), "Error saving test file")

        image_seg_pairs = img_seg_pairs = data_loader.get_pairs_from_paths(self.train_temp_dir, self.test_temp_dir)

        random.seed(0)
        random.shuffle(image_seg_pairs)
        zipped = itertools.cycle(img_seg_pairs)

        random.seed(0)

        generator = data_loader.image_segmentation_generator(
            self.train_temp_dir, self.test_temp_dir, 1,
            image_size * image_size, image_size, image_size, image_size, image_size,
            do_augment=True, custom_augmentation=self.custom_aug
        )

        i = 0
        loop = 1
        for (aug_im, aug_an), (expt_im_f, expt_an_f) in zip(generator, zipped):
            if i >= loop:
                break

            expt_im = data_loader.get_image_array(expt_im_f, image_size, image_size, ordering='channel_last')

            expt_im = cv2.flip(expt_im, flipCode=1)
            self.assertTrue(np.equal(expt_im, aug_im).all())

            i += 1

        os.remove(train_file)
        os.remove(test_file)

    def test_image_segmentation_generator_preprocessing(self):
        pass

