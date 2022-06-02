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
import cv2
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """
    Directory of the folder which expects to be inb the structure of ->[train/val/test]->[class labels]->[videos].
    Class initializes with the list of file names and array of labels

    :Param directory [str]: Path of the directory contain the dataset
    :Param mode [str, optional]: Determines which folder of the directory the dataset will read from. Default to train.
    :Param clip_length [int, optional]: Determines how many frames are there in each clip. Defaults to 20.

    :Return Buffer: normalized numpy array of videos in the form of [C,D,W,H]
    """

    def __init__(self, config, mode='train'):

        folder = os.path.join(config["directory"], mode)
        self.clip_len = config["clip_len"]
        self.resize_height = config["resize_height"]
        self.resize_width = config["resize_width"]
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

    def resnet_loadvideo(self, fname):
        """
        Checks the video resolution and resizes the vide according to the initialized resolution.

        :Param fname: directory address of the single video
        :Return Buffer: Resized videos in the numpy array in the form of [C,D,W,H] based on pytorch usage
        """

        capture = cv2.VideoCapture(fname)
        video_file = fname
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))

        count = 0
        retaining = True

        while count < frame_count and retaining:
            retaining, frame = capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            buffer[count] = frame
            count += 1

        capture.release()
        buffer = buffer.transpose((3, 0, 1, 2))

        return buffer, video_file

    def video_name(self, video_file):
        """
        Changes the file directory from train to

        :param video_file: video file from the train or val

        :return color_fname:
        """
        # print(video_file)
        head_original_file = os.path.split(video_file)
        file_name = head_original_file[1]
        head_class = os.path.split(head_original_file[0])
        class_name = head_class[1]
        split_up = os.path.split(head_class[0])
        split_name = split_up[1]
        directory = split_up[0]

        if split_name == "train":
            color_fname = os.path.join(directory, "train_color", class_name, file_name)

        else:
            color_fname = os.path.join(directory, "val_color", class_name, file_name)

        return color_fname

    def color_loadvideo(self, color_fname):
        """
        Checks the video resolution and resizes the vide according to the initialized resolution.

        :Param fname: directory address of the single video
        :Return Buffer: Resized videos in the numpy array in the form of [C,D,W,H] based on pytorch usage
        """

        capture = cv2.VideoCapture(color_fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        color_buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))

        count = 0
        retaining = True

        while count < frame_count and retaining:
            retaining, frame = capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            color_buffer[count] = frame
            count += 1

        capture.release()
        color_buffer = color_buffer.transpose((3, 0, 1, 2))

        return color_buffer

    def resnet_crop(self, buffer, clip_len, crop_size):

        """
        Takes the resized video in the form of numpy array from method load_video(), clip_length and crop_size which are
        initialized. Randomize the crop in the video

        :Param buffer [numpy array]: resized video in the form of numpy array from method load_video()
        :Param clip_len [int]: No frames are there in each clip
        :Param crop_size [int]: resolution for the video to be cropped

        :Return buffer [numpy array]: cropped video both in resolution and depth
        """
        time_index = np.random.randint(0, (buffer.shape[1] - clip_len))
        height_index = np.random.randint(buffer.shape[2] - crop_size)
        width_index = np.random.randint(buffer.shape[3] - crop_size)

        buffer = buffer[:, time_index:time_index + clip_len, height_index:height_index + crop_size,
                 width_index:width_index + crop_size]
        return buffer, time_index, height_index, width_index

    def resnet_normalize(self, buffer):
        """
        Normalizes the buffer. Default values of RGB images normalization are used, as precomputed
        mean and std_dev values (akin to ImageNet) were unavailable for Kinetics. Feel
        free to push to and edit this section to replace them if found.

        :Param buffer [numpy array]: cropped video in the form of numpy array from method crop()
        :Return buffer [numpy array]: normalized numpy array
        """

        buffer = (buffer - 128) / 128
        return buffer

    def color_crop(self, color_buffer, time_index, height_index, width_index):

        color_buffer = color_buffer[:, time_index:time_index + (self.clip_len - 1),
                       height_index: height_index + self.crop_size,
                       width_index: width_index + self.crop_size]

        return color_buffer

    def color_normalize(self, color_buffer):

        color_buffer = (color_buffer - 128) / 128

        return color_buffer

    def __getitem__(self, index):

        buffer, video_file = self.resnet_loadvideo(self.fnames[index])
        color_fname = self.video_name(video_file)
        buffer, time_index, height_index, width_index = self.resnet_crop(buffer, self.clip_len, self.crop_size)
        buffer = self.resnet_normalize(buffer)

        color_buffer = self.color_loadvideo(color_fname)
        color_buffer = self.color_crop(color_buffer, time_index, height_index, width_index)
        color_buffer = self.color_normalize(color_buffer)

        return buffer, self.label_array[index], color_buffer

    def __len__(self):
        return len(self.fnames)
