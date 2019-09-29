import unittest
import tempfile
from shutil import rmtree
import os
import six
from keras_segmentation import train

class TestTrainInternalFunctions(unittest.TestCase):
    """ Test internal functions of the module """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        rmtree(self.tmp_dir)

    def test_find_latest_checkpoint(self):
        # Populate a folder of images and try checkpoint
        checkpoints_path = os.path.join(self.tmp_dir, "test1")
        # Create files
        self.assertEqual(None, train.find_latest_checkpoint(checkpoints_path))
        # When fail_safe is turned off, throw an exception when no checkpoint is found.
        six.assertRaisesRegex(self, ValueError, "Checkpoint path", train.find_latest_checkpoint, checkpoints_path, False)
        for suffix in ["0", "2", "4", "12", "_config.json", "ABC"]:
            open(checkpoints_path + '.' + suffix, 'a').close()
        self.assertEqual(checkpoints_path + ".12", train.find_latest_checkpoint(checkpoints_path))