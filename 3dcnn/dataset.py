"""
Copyright(c) 2019-2022 Deep Safety GmbH.

All rights not expressly granted by the Licensor remain reserved.

This unpublished material is proprietary to Deep Safety GmbH.

Proprietary software is computer software licensed under exclusive legal right
of the copyright holder. The receipt or possession of this source code and /
or related information does not convey or imply any rights to use, reproduce,
disclose or distribute its contents, or to manufacture, use, or sell anything
that it may describe, in whole or in part unless prior written permission is
obtained from Deep Safety GmbH.

The methods and techniques described herein are considered trade secrets and /
or confidential. You shall not disclose such Confidential Information and
shall use it only in accordance with the terms of the license agreement you
entered into with Deep Safety GmbH.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES.
"""

import os
import numpy as np
from numpy import load
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """
    Directory of the folder which expects to be inb the structure of ->[train/val]->[class labels]->[videos].
    Class initializes with the list of file names and array of labels

    :param config: YAML file containing the configuration such as directory, clip_len, crop_size
    :param mode: Chooses whether [train/val] directory, default to [train]
    """

    def __init__(self, config, mode='train'):

        folder = os.path.join(config["directory"], mode)
        self.config = config
        self.clip_len = config["clip_len"]
        self.crop_size = config["crop_size"]
        self.fnames, labels = [], []

        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        # {'dancing': 0, 'deadlifting': 1, 'excercing': 2, 'jogging': 3, 'looking_at_phone': 4, 'moving_furniture': 5,
        #  'packing': 6, 'pull_ups': 7, 'push_up': 8, 'putting_on_shoes': 9, 'reading': 10, 'shaking_head': 11,
        #  'talking_on_cell_phone': 12}

    def __getitem__(self, index):

        color_directory = self.name_convertor(self.fnames[index])
        buffer = self.resnet_numpy_convertor(self.fnames[index])
        color_buffer = self.color_numpy_convertor(color_directory)
        buffer, time_index, height_index, width_index = self.resnet_crop(buffer, self.clip_len, self.crop_size)
        color_buffer = self.color_crop(color_buffer, self.clip_len, self.crop_size, time_index, height_index,
                                       width_index)
        buffer = self.resnet_normalize(buffer)
        color_buffer = self.color_normalize(color_buffer)

        return buffer, color_buffer, self.label_array[index]

    def name_convertor(self, name):
        """
        Takes the video file and changes the directory with changing the [train/val] to [train_color/val_color] and
        changes the video file extension from xyz.mp4.npz to xyz.avi.npz

        :param name: Video file from [train/val] directory with the extension '.mp4.npz'
        :return color_directory: Video file from [train_color/val_color] directory with the extension '.avi.npz'
        """
        # print("name:", name)
        head_file = os.path.split(name)
        file_name = head_file[1]
        file = file_name.split(".")
        file_name_change = file[0] + ".avi.npz"
        class_split = os.path.split(head_file[0])
        class_name = class_split[1]
        split_up = os.path.split(class_split[0])
        split_name = split_up[1]
        directory = split_up[0]

        if split_name == "val":
            color_directory = os.path.join(directory, "val_color", class_name, file_name_change)
            # print("color_directory:", color_directory)

        if split_name == "train":
            color_directory = os.path.join(directory, "train_color", class_name, file_name_change)
            # print("color_directory:", color_directory)

        return color_directory

    def resnet_numpy_convertor(self, fname):
        """
        Takes the video numpy zip file and unzips

        :param fname: Zipped video file from [train/val] split
        :return buffer: Numpy array
        """

        # print(fname)
        load_buffer = load(fname)
        buffer = load_buffer['arr_0']

        return buffer

    def color_numpy_convertor(self, fname):
        """
        Takes the video numpy zip file and unzips

        :param fname: Zipped video file from [train_color/val_color] split
        :return color_buffer: Numpy array
        """

        load_buffer = load(fname)
        color_buffer = load_buffer['arr_0']

        return color_buffer

    def resnet_crop(self, buffer, clip_len, crop_size):
        """
        Generates the random time frame, height and width, and crops randomly based on the given size by the user in
        the numpy array

        :param buffer: Numpy array of the video [C,T,H,W]
        :param clip_len: Number of frames to be used from the user
        :param crop_size: Cropping size of the video from the user

        :return buffer: Numpy array of original video
        :return time_index: Generated random integer of starting frame
        :return height_index: Generated random integer of frame height
        :return width_index: Generated random integer of frame width
        """

        time_index = np.random.randint(0, (buffer.shape[1] - (clip_len * self.config["frame_offset"])))

        height_index = np.random.randint(buffer.shape[2] - crop_size)
        width_index = np.random.randint(buffer.shape[3] - crop_size)

        buffer = buffer[:, time_index:time_index + clip_len:self.config["frame_offset"],
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size]

        return buffer, time_index, height_index, width_index

    def color_crop(self, color_buffer, clip_len, crop_size, time_index, height_index, width_index):
        """
        Random numbers generated of time frame, height and width from in the resnet_crop() crops randomly based on
        the given size by the user in the numpy array of the color buffer

        :param color_buffer: Numpy array of the optical-flow video [C,T,H,W]
        :param clip_len: Number of frames to be used from the user
        :param crop_size: Cropping size of the video from the user
        :param time_index: Generated random integer of starting frame from resnet_crop()
        :param height_index: Generated random integer of frame height from resnet_crop()
        :param width_index: Generated random integer of frame width from resnet_crop()

        :return buffer: Numpy array of optical-flow video
        """

        color_buffer = color_buffer[:, time_index:time_index + (clip_len - 1):self.config["frame_offset"],
                       height_index: height_index + crop_size,
                       width_index: width_index + crop_size]

        return color_buffer

    def resnet_normalize(self, buffer):
        """
        Normalizes the array of color video

        :param buffer:Numpy array of the video [C,T,H,W]
        :return buffer Normalized numpy array

        """
        buffer = (buffer - 128) / 128
        return buffer

    def color_normalize(self, color_buffer):
        """
        Normalizes the array of optical-flow video

        :param color_buffer:Numpy array of the optical-flow video [C,T,H,W]
        :return color_buffer: Normalized numpy array

        """
        color_buffer = (color_buffer - 128) / 128
        return color_buffer

    def __len__(self):
        return len(self.fnames)
