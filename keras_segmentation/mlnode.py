"""Actual mlnode class of seguitls
inheriting seg/mlnode"""

from typing import Optional
import tensorflow as tf
from workflow.seg_node.mlnode import MlNodeSeg

import os
import numpy as np
from pathlib import Path


class MlNodeKerasSeg(MlNodeSeg):
    """mlnode of segutils"""
    def __init__(self, projectname: str, node_name: str, workspace: Path, project_config_path: Optional[Path] = None, load_model_flag: bool = True, dump_results_flag: bool = False, show_flag: bool = False, git_flag: bool = True, gpu_flag: bool = True, db_stash_flag: bool = False, db_pull_flag: bool = True, wt_stash_flag: bool = False, wt_pull_flag: bool = True, print_flag: bool = True, run_mode: str = "infer", debug_level: int = 0, full_db_flag: bool = False, dump_parent_crops_flag: bool = False, all_nodes=None, remove_fp: bool = False, framework: str = "tensorflow") -> None:
        super().__init__(projectname, node_name, workspace, project_config_path, load_model_flag, dump_results_flag, show_flag, git_flag, gpu_flag, db_stash_flag, db_pull_flag, wt_stash_flag, wt_pull_flag, print_flag, run_mode, debug_level, full_db_flag, dump_parent_crops_flag, all_nodes, remove_fp, framework)

    def load_model(self, gpu_flag):
        """load model,
        override this
        eg:
        model = tf.keras.models.load_model('/tmp/model')
        """
        
        # self.model = tf.keras.models.load_model(self.weights_path)
        breakpoint()
        from keras_segmentation.models.unet import vgg_unet
        self.model = vgg_unet(n_classes=2 ,  input_height=416, input_width=608  )
        self.weights_path = os.path.join(self.weight_dir,'vgg_unet.h5')
        if os.path.isfile(self.weights_path):
            print("Existing trained model loading")
            self.model.load_weights(self.weights_path)
        else:
            print("weight not found")

    def predict(self, img_batch):
        """image batch for prediciton
        eg:
        pred_batch = self.model.predict(img_batch)
        """
        breakpoint()
        pred_batch = self.model.predict_multiple(inps = img_batch, out_dir = "Results")
        return pred_batch


if __name__ == "__main__":
    pass
