from tqdm import tqdm
from PIL import Image
from glob import glob
from time import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segmentation.keras_segmentation.pretrained import pspnet_50_ADE_20K

model = pspnet_50_ADE_20K()  # load the pretrained model trained on ADE20k dataset


def output_prediction(input_folder: str):
    '''Performs object segmentation of the input's image'''
    files = glob(f"{input_folder}/*")
    t1 = []

    for iteration, file in tqdm(enumerate(files), total=len(files)):
        img_file = cv2.imread(file)
        img_file = cv2.resize(img_file, (640, 480))
        t = time()
        out = model.predict_segmentation(inp=img_file)
        t1.append(time() - t)
        out = np.array(out, dtype='uint8')
        out1 = cv2.resize(out, (640, 480))
        #plt.imshow(img_file)
        #plt.imshow(out1, interpolation='none', alpha=0.6)

    # Creating a file with the time taken per frame
    record = open("Record.txt", "w")
    record.write("time in seconds for each input file : \n")
    for iteration, x in enumerate(t1):
        record.write(str(iteration) + " " + str(x) + "\n")

    return out1


if __name__ == "__main__":
    output_prediction("sidewalk")
