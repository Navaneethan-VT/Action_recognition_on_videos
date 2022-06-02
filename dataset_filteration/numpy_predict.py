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

import glob
from multiprocessing import Pool
from numpy import load
import numpy as np
from _csv import writer
import torch
import os

from train import Temp

# check_point = "/home/navaneethan/two_stream_architecture/tb_logs/new_model/version_3/checkpoints/epoch=608-step" \
#               "=168083.ckpt"
check_point = "/home/navaneethan/results/new_model/checkpoints/epoch=615-step=170015.ckpt"
device = torch.device("cuda")
model = Temp.load_from_checkpoint(check_point, map_location=device)
model.eval()


def load_numpy(file_name, color_file_name):
    """
    Uncompresses the Numpy array

    :param file_name: Name of the file to uncompress
    :param color_file_name: Name of the color video file to uncompress
    :return buffer: Numpy array of the video for first stream network
    :return color_buffer: Numpy array of the colored video for the second stream netwrok
    """
    load_buffer = load(file_name)
    buffer = load_buffer['arr_0']

    load_color_buffer = load(color_file_name)
    color_buffer = load_color_buffer['arr_0']

    return buffer, color_buffer


def index_frame(frame_index):
    """
    It makes a list of index for using in the csv file

    :param frame_index: Total no of frames
    :retrun index_list: frame numbers in list
    """
    index_list = list(range(0, frame_index))

    return index_list


def trim_frame(buffer, color_buffer, i):
    """
    With the frame index, it trims and reshapes to the shape [1,3,16,112,112] and [1,3,15,112,112] for first stream
    and second stream respectively

    :param buffer: Numpy array of the actual video frame for first stream network
    :param color_buffer: Numpy array of the colored video frame for second stream network
    :return buffer_1: Numpy array for the first stream network
    :return color_buffer_1: Numpy array for the second stream network
    """

    print("buffer:", buffer.shape)
    print("c_buffer:", color_buffer.shape)
    buffering = buffer[:, i:i + 16, :112, :112]
    print("buffer1:", buffering.shape)
    dash = torch.from_numpy(buffering)
    print("dash:", dash.shape)
    buffer_1 = dash.reshape((1, 3, 16, 112, 112))
    print("buffer_1:", buffer_1.shape)

    color_buffering = color_buffer[:, i:i + 15, :112, :112]
    print("c_buffer1:", color_buffering.shape)
    beta = torch.from_numpy(color_buffering)
    print("beta:", beta.shape)
    color_buffer_1 = beta.reshape((1, 3, 15, 112, 112))
    print("color_buffer_1:", color_buffer_1.shape)
    print("1")
    return buffer_1, color_buffer_1


def load_checkpoint(buffer, color_buffer):
    """
    Using the GPU and predicts the model by using the check_point

    :param check_point: check_point of the trained model
    :param buffer: Reshaped numpy array for the first stream network
    :param color_buffer: Reshaped numpy array of the colored video for the second stream network

    :return prediction: The prediction from the model in the form of numpy array with 13 classes
    """

    # print("printing load_checkpoint")

    with torch.no_grad():
        out = model(buffer, color_buffer)
        out = (out.numpy().flatten())
        prediction = np.round_(out, decimals=3)
        print(prediction)

    return prediction


def write_csv(index, csv_file_name, data):
    """
    With the index number and the prediction as a list saves in the csv file

    :param index: Frame number
    :param csv_file_name: Name of the csv file to load the list
    :param  data: The predicted data from the model for the corresponding 16 frames
    """

    # print("write_csv")
    index_with_prediction = [index]
    index_with_prediction.extend(data)
    print(index_with_prediction)
    with open(csv_file_name, 'a', newline='') as file:
        writer_object = writer(file)
        writer_object.writerow(index_with_prediction)
        file.close()

    # pd.read_csv(file, header=None).T.to_csv(file, header=False, index=False)


def sub_main(name):
    """
    Sub main folder is as the main function, to have a multiprocessing converted to sub main
    ◉ It converts the corresponding file name for calling the optical flow video
    ◉ Creates a folder for csv if not exists

    :param name: Video file for predicting

    """

    head_file = os.path.split(name)
    file_name = head_file[1]
    file = file_name.split(".")
    file_name_change = file[0] + ".avi.npz"
    csv_file_name = file_name + ".csv"
    class_split = os.path.split(head_file[0])
    class_name = class_split[1]
    split_up = os.path.split(class_split[0])
    split_name = split_up[1]
    directory = split_up[0]

    if split_name == "val":
        color_name = os.path.join(directory, "val_color", class_name, file_name_change)
        # print("color_directory:", color_directory)

    if split_name == "train":
        color_name = os.path.join(directory, "train_color", class_name, file_name_change)

    csv_folder = os.path.join("/home/navaneethan/dataset/csv/", split_name, class_name)

    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)

    csv_file_name = os.path.join(csv_folder, csv_file_name)

    buffer, color_buffer = load_numpy(name, color_name)
    frame_index = buffer.shape
    index_list = index_frame(int(frame_index[1] - 16))
    # print(index_list)

    for i in index_list:
        print(i)
        b_numpy, c_numpy = trim_frame(buffer, color_buffer, i)
        print("2...")
        prediction = load_checkpoint(b_numpy, c_numpy)
        print("2....")
        write_csv(i, csv_file_name, prediction)


def main():
    directory = glob.glob("/home/navaneethan/testing_folder/val/*/*")
    # directory = glob.glob("/home/navaneethan/working/dataset/val/*/*")
    p = Pool(1)
    result = p.map(func=sub_main, iterable=directory)
    p.close()
    p.join()


if __name__ == '__main__':
    main()
