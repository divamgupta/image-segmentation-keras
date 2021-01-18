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
    def setUp(self):
        self.train_temp_dir = tempfile.mkdtemp()
        self.test_temp_dir = tempfile.mkdtemp()
        self.other_temp_dir = tempfile.mkdtemp()
        self.other_temp_dir_2 = tempfile.mkdtemp()

        self.image_size = 4

        # Training
        train_image = np.arange(self.image_size * self.image_size)
        train_image = train_image.reshape((self.image_size, self.image_size))

        train_file = os.path.join(self.train_temp_dir, "train.png")
        test_file = os.path.join(self.test_temp_dir, "train.png")

        cv2.imwrite(train_file, train_image)
        cv2.imwrite(test_file, train_image)

        # Testing
        train_image = np.arange(start=self.image_size * self.image_size,
                                stop=self.image_size * self.image_size * 2)
        train_image = train_image.reshape((self.image_size,self.image_size))

        train_file = os.path.join(self.train_temp_dir, "train2.png")
        test_file = os.path.join(self.test_temp_dir, "train2.png")

        cv2.imwrite(train_file, train_image)
        cv2.imwrite(test_file, train_image)

        # Extra one

        i = 0
        for dir in [self.other_temp_dir, self.other_temp_dir_2]:
            extra_image = np.arange(start=self.image_size * self.image_size * (2 + i),
                                    stop=self.image_size * self.image_size * (2 + i + 1))
            extra_image = extra_image.reshape((self.image_size, self.image_size))

            extra_file = os.path.join(dir, "train.png")
            cv2.imwrite(extra_file, extra_image)
            i += 1

            extra_image = np.arange(start=self.image_size * self.image_size * (2 + i),
                                    stop=self.image_size * self.image_size * (2 + i + 1))
            extra_image = extra_image.reshape((self.image_size, self.image_size))

            extra_file = os.path.join(dir, "train2.png")
            cv2.imwrite(extra_file, extra_image)
            i += 1

    def tearDown(self):
        shutil.rmtree(self.train_temp_dir)
        shutil.rmtree(self.test_temp_dir)
        shutil.rmtree(self.other_temp_dir)
        shutil.rmtree(self.other_temp_dir_2)

    def custom_aug(self):
        return iaa.Sequential(
            [
                iaa.Fliplr(1),  # horizontally flip 100% of all images
            ])

    def test_image_segmentation_generator_custom_augmentation(self):
        random.seed(0)
        image_seg_pairs = data_loader.get_pairs_from_paths(self.train_temp_dir, self.test_temp_dir)

        random.seed(0)
        random.shuffle(image_seg_pairs)

        random.seed(0)

        generator = data_loader.image_segmentation_generator(
            self.train_temp_dir, self.test_temp_dir, 1,
            self.image_size * self.image_size, self.image_size, self.image_size, self.image_size, self.image_size,
            do_augment=True, custom_augmentation=self.custom_aug
        )

        i = 0
        for (aug_im, aug_an), (expt_im_f, expt_an_f) in zip(generator, image_seg_pairs):
            if i >= len(image_seg_pairs):
                break

            expt_im = data_loader.get_image_array(expt_im_f, self.image_size, self.image_size, ordering='channel_last')

            expt_im = cv2.flip(expt_im, flipCode=1)
            self.assertTrue(np.equal(expt_im, aug_im).all())

            i += 1

    def test_image_segmentation_generator_custom_augmentation_with_other_inputs(self):
        other_paths = [
                self.other_temp_dir, self.other_temp_dir_2
            ]
        random.seed(0)
        image_seg_pairs = data_loader.get_pairs_from_paths(self.train_temp_dir,
                                                           self.test_temp_dir,
                                                           other_inputs_paths=other_paths)

        random.seed(0)
        random.shuffle(image_seg_pairs)

        random.seed(0)
        generator = data_loader.image_segmentation_generator(
            self.train_temp_dir, self.test_temp_dir, 1,
            self.image_size * self.image_size, self.image_size, self.image_size, self.image_size,
            self.image_size,
            do_augment=True, custom_augmentation=self.custom_aug, other_inputs_paths=other_paths
        )

        i = 0
        for (aug_im, aug_an), (expt_im_f, expt_an_f, expt_oth) in zip(generator, image_seg_pairs):
            if i >= len(image_seg_pairs):
                break

            ims = [expt_im_f]
            ims.extend(expt_oth)

            for i in range(aug_im.shape[1]):
                expt_im = data_loader.get_image_array(ims[i], self.image_size, self.image_size,
                                                      ordering='channel_last')

                expt_im = cv2.flip(expt_im, flipCode=1)

                self.assertTrue(np.equal(expt_im, aug_im[0, i, :, :]).all())

            i += 1

    def test_image_segmentation_generator_with_other_inputs(self):
        other_paths = [
                self.other_temp_dir, self.other_temp_dir_2
            ]
        random.seed(0)
        image_seg_pairs = data_loader.get_pairs_from_paths(self.train_temp_dir,
                                                           self.test_temp_dir,
                                                           other_inputs_paths=other_paths)

        random.seed(0)
        random.shuffle(image_seg_pairs)

        random.seed(0)
        generator = data_loader.image_segmentation_generator(
            self.train_temp_dir, self.test_temp_dir, 1,
            self.image_size * self.image_size, self.image_size, self.image_size, self.image_size,
            self.image_size,
            other_inputs_paths=other_paths
        )

        i = 0
        for (aug_im, aug_an), (expt_im_f, expt_an_f, expt_oth) in zip(generator, image_seg_pairs):
            if i >= len(image_seg_pairs):
                break

            ims = [expt_im_f]
            ims.extend(expt_oth)

            for i in range(aug_im.shape[1]):
                expt_im = data_loader.get_image_array(ims[i], self.image_size, self.image_size,
                                                      ordering='channel_last')
                self.assertTrue(np.equal(expt_im, aug_im[0, i, :, :]).all())

            i += 1

    def test_image_segmentation_generator_preprocessing(self):
        image_seg_pairs = data_loader.get_pairs_from_paths(self.train_temp_dir, self.test_temp_dir)

        random.seed(0)
        random.shuffle(image_seg_pairs)

        random.seed(0)

        generator = data_loader.image_segmentation_generator(
            self.train_temp_dir, self.test_temp_dir, 1,
            self.image_size * self.image_size, self.image_size, self.image_size, self.image_size,
            self.image_size,
            preprocessing=lambda x: x + 1
        )

        i = 0
        for (aug_im, aug_an), (expt_im_f, expt_an_f) in zip(generator, image_seg_pairs):
            if i >= len(image_seg_pairs):
                break

            expt_im = data_loader.get_image_array(expt_im_f, self.image_size, self.image_size,
                                                  ordering='channel_last')

            expt_im += 1
            self.assertTrue(np.equal(expt_im, aug_im[0, :, :]).all())

            i += 1

    def test_single_image_segmentation_generator_preprocessing_with_other_inputs(self):
        other_paths = [
                self.train_temp_dir, self.test_temp_dir
            ]
        random.seed(0)
        image_seg_pairs = data_loader.get_pairs_from_paths(self.train_temp_dir,
                                                                           self.test_temp_dir,
                                                                          other_inputs_paths=other_paths)

        random.seed(0)
        random.shuffle(image_seg_pairs)

        random.seed(0)
        generator = data_loader.image_segmentation_generator(
            self.train_temp_dir, self.test_temp_dir, 1,
            self.image_size * self.image_size, self.image_size, self.image_size, self.image_size,
            self.image_size,
            preprocessing=lambda x: x+1, other_inputs_paths=other_paths
        )

        i = 0
        for (aug_im, aug_an), (expt_im_f, expt_an_f, expt_oth) in zip(generator, image_seg_pairs):
            if i >= len(image_seg_pairs):
                break

            ims = [expt_im_f]
            ims.extend(expt_oth)

            for i in range(aug_im.shape[1]):
                expt_im = data_loader.get_image_array(ims[i], self.image_size, self.image_size,
                                                      ordering='channel_last')

                self.assertTrue(np.equal(expt_im + 1, aug_im[0, i, :, :]).all())

            i += 1

    def test_multi_image_segmentation_generator_preprocessing_with_other_inputs(self):
        other_paths = [
                self.other_temp_dir, self.other_temp_dir_2
            ]
        random.seed(0)
        image_seg_pairs = data_loader.get_pairs_from_paths(self.train_temp_dir,
                                                                           self.test_temp_dir,
                                                                          other_inputs_paths=other_paths)

        random.seed(0)
        random.shuffle(image_seg_pairs)

        random.seed(0)
        generator = data_loader.image_segmentation_generator(
            self.train_temp_dir, self.test_temp_dir, 1,
            self.image_size * self.image_size, self.image_size, self.image_size, self.image_size,
            self.image_size,
            preprocessing=[lambda x: x+1, lambda x: x+2, lambda x: x+3], other_inputs_paths=other_paths
        )

        i = 0
        for (aug_im, aug_an), (expt_im_f, expt_an_f, expt_oth) in zip(generator, image_seg_pairs):
            if i >= len(image_seg_pairs):
                break

            ims = [expt_im_f]
            ims.extend(expt_oth)

            for i in range(aug_im.shape[1]):
                expt_im = data_loader.get_image_array(ims[i], self.image_size, self.image_size,
                                                      ordering='channel_last')

                self.assertTrue(np.equal(expt_im + (i + 1), aug_im[0, i, :, :]).all())

            i += 1



