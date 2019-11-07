import numpy as np
import cv2
from keras.utils import Sequence


class TCCGenerator(Sequence):

    def __init__(self, image_filenames, masks_filenames, batch_size):
        self.image_filenames, self.masks_filenames = image_filenames, masks_filenames
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.image_filenames) / float(self.batch_size)).astype(int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.masks_filenames[idx * self.batch_size:(idx+1) * self.batch_size]

        masks = np.array([cv2.imread(filename,0) for filename in batch_y])
        images = np.array([cv2.imread(filename) for filename in batch_x])
        images = images/255
        masks = masks/255

        return images, masks