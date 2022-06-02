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
import random

import numpy as np
from numpy import load
import pandas as pd
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

    def __getitem__(self, index):
        optical_flow_video_file = self.get_optical_flow_file(self.fnames[index])
        csv_file, csv_folder_name = self.get_csv_file(self.fnames[index])
        frame_number = self.get_csv_info(csv_file, csv_folder_name)
        buffer = self.resnet_numpy_convertor(self.fnames[index])
        color_buffer = self.color_numpy_convertor(optical_flow_video_file)
        buffer, height_index, width_index = self.resnet_crop(buffer, frame_number, self.clip_len, self.crop_size)
        color_buffer = self.color_crop(color_buffer, frame_number, self.clip_len, height_index, width_index,
                                       self.crop_size)
        buffer = self.resnet_normalize(buffer)
        color_buffer = self.color_normalize(color_buffer)

        return buffer, color_buffer, self.label_array[index]

    def __len__(self):
        return len(self.fnames)

    def get_optical_flow_file(self, fname):
        """
        Takes the video file and changes the directory with changing the [train/val] to [train_color/val_color] and
        changes the video file extension from xyz.mp4.npz to xyz.avi.npz

        :param fname: Video file from [train/val] directory with the extension '.mp4.npz'
        :return color_directory: Video file from [train_color/val_color] directory with the extension '.avi.npz'
        """

        head_file = os.path.split(fname)
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

        elif split_name == "train":
            color_directory = os.path.join(directory, "train_color", class_name, file_name_change)

        return color_directory

    def get_csv_file(self, fname):
        """
        csv file finder

        :param fname: Video file from [train/val] directory with the extension '.mp4.npz'
        :return exact_file: corresponding csv file for the video file
        : return folder_name: Class name corresponding to the video file
        """

        file_name_split = fname.split("/")
        short_file_name = file_name_split[-1].split(".")
        file_name = short_file_name[0] + ".mp4.csv"
        folder_name = file_name_split[-2]
        split_name = file_name_split[-3]
        exact_file = "/home/navaneethan/dataset/csv" + "/" + split_name + "/" + folder_name + "/" + file_name

        return exact_file, folder_name

    def get_csv_info(self, exact_file, folder_name):

        '''
        Calclulates the threshold value for the prediction from the csv file

        :param exact_file: corresponding csv file for the video file
        :param folder_name: Class name corresponding to the video file

        :return value: Threshold value
        '''

        header = ['INDEX', 'checking_tires', 'deadlifting', 'excercing', 'jogging', 'looking_at_phone',
                  'moving_furniture', 'playing_badminton', 'playing_chess', 'playing_guitar', 'pull_ups',
                  'push_up', 'putting_on_shoes', 'shaking_head']

        info_data = pd.read_csv(exact_file, names=header)
        info = info_data[folder_name].tolist()

        last_frame = len(info)
        average = sum(info) / len(info)
        threshold_value = (round(average, 2)) - 0.2

        start = False
        listing = []

        for idx, val in enumerate(info):
            if start == False and val >= threshold_value:
                listing.append(idx)
                start = True
            elif start == True and val < threshold_value:
                listing.append(idx)
                start = False
            else:
                pass
        listing.append(last_frame)

        chunk_size = 2
        chunked_list = list()
        for i in range(0, len(listing), chunk_size):
            chunked_list.append(listing[i:i + chunk_size])

        emp_list = []
        for idx, list_elements in enumerate(chunked_list):
            start = list_elements[0]
            end = list_elements[-1]
            all_numbers_in_range = [*range(start, end + 1)]
            emp_list.extend(all_numbers_in_range)

        value = random.choice(emp_list)
        return value

    def resnet_numpy_convertor(self, fname):
        """
        Takes the video numpy zip file and unzips

        :param fname: Zipped video file from [train/val] split
        :return buffer: Numpy array
        """

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

    def resnet_crop(self, buffer, frame_number, clip_len, crop_size):
        '''
        Gets the numpy array and crops in randomly on temporal and spatial coordinates

        :param buffer: Numpy array of the video
        :param frame_number: Total number of frames in the video file
        :param clip_len: No of frames required for training
        :param crop_size: Final crop size in spatial dimension
        :return buffer: Cropped numpy array of the video file
        :return height_index: Random height index for optical flow video usage
        :return width_index: Random width index for optical flow video usage

        '''

        height_index = np.random.randint(buffer.shape[2] - crop_size)
        width_index = np.random.randint(buffer.shape[3] - crop_size)

        buffer = buffer[:, frame_number:frame_number + clip_len:self.config["frame_offset"],
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size]

        return buffer, height_index, width_index

    def color_crop(self, color_buffer, frame_number, clip_len, height_index, width_index, crop_size):

        '''
        Gets the numpy array and crops in randomly on temporal and spatial coordinates

        :param color_buffer: Numpy array of the optical flow video
        :param frame_number: Total number of frames in the video file
        :param clip_len: No of frames required for training
        :param height_index: Random height index from original video
        :return width_index: Random width index from original video
        :param crop_size: Final crop size in spatial dimension
        :return color_buffer: Cropped numpy array of the optical flow video file

        '''
        color_buffer = color_buffer[:, frame_number:frame_number + (clip_len - 1):self.config["frame_offset"],
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
