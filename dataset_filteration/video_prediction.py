import time
from _csv import writer

import cv2
import glob
import numpy as np
import torch
import os
from train import Temp

#
check_point = "/home/navaneethan/two_stream_architecture/tb_logs/new_model/version_3/checkpoints/epoch=608-step" \
              "=168083.ckpt"
# check_point = "/home/navaneethan/results/new_model/checkpoints/epoch=615-step=170015.ckpt"
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:1")
model = Temp.load_from_checkpoint(check_point, map_location=device)
model.eval()


def file_split(name):
    """
    From the file name takes the file for the second stream network and creates the csv file for the given video file
    if that doesn't exist

    :param name: File for using to extract name for finding the optical video file

    :return color_name: File for using to extract name for finding the optical video file
    :return csv_file_name: Csv file address for storing the prediction


    """
    head_file = os.path.split(name)
    file_name = head_file[1]
    file = file_name.split(".")
    file_name_change = file[0] + ".avi"
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

    return color_name, csv_file_name


def convert_numpy(file, color_file):
    cap = cv2.VideoCapture(file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    resize_height = 112
    resize_width = 112

    buffer = np.empty((frame_count, resize_height, resize_width, 3), np.dtype('float32'))
    count = 0
    ret = True
    while count < frame_count and ret:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if (frame_height != resize_height) or (frame_width != resize_width):
            frame = cv2.resize(frame, (resize_width, resize_height))
        buffer[count] = frame
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    buffer = buffer.transpose((3, 0, 1, 2))
    buffer = (buffer - 128) / 128

    cap = cv2.VideoCapture(color_file)
    color_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    color_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    color_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    resize_height = 112
    resize_width = 112
    color_buffer = np.empty((color_count, resize_height, resize_width, 3), np.dtype('float32'))
    count = 0
    ret = True
    while count < color_count and ret:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if (color_height != resize_height) or (color_width != resize_width):
            frame = cv2.resize(frame, (resize_width, resize_height))
        color_buffer[count] = frame
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    color_buffer = color_buffer.transpose((3, 0, 1, 2))
    color_buffer = (color_buffer - 128) / 128

    buffering = torch.from_numpy(buffer)
    color_buffering = torch.from_numpy(color_buffer)

    return buffering, color_buffering, frame_count


def trim_frame(buffer, color_buffer, frame):
    """
    With the frame index, it trims and reshapes to the shape [1,3,16,112,112] and [1,3,15,112,112] for first stream
    and second stream respectively

    :param buffer: Numpy array of the actual video frame for first stream network
    :param color_buffer: Numpy array of the colored video frame for second stream network
    :return buffer_1: Numpy array for the first stream network
    :return color_buffer_1: Numpy array for the second stream network
    """
    # print(f"1: {buffer.shape}")
    # print(f"frame:{frame}")
    buffering = buffer[:, frame:frame + 16, :112, :112]
    # print(f"1.1: {buffering.shape}")

    dash_1 = buffering.reshape((1, 3, 16, 112, 112))
    # print(f"1.3: {dash_1.shape}")
    # print(f"2: {color_buffer.shape}")
    color_buffering = color_buffer[:, frame:frame + 15, :112, :112]
    # print(f"2.1: {color_buffering.shape}")

    dash_2 = color_buffering.reshape((1, 3, 15, 112, 112))
    # print(f"2.3: {dash_2.shape}")

    return dash_1, dash_2


def predict(buffer, color_buffer):
    """
    Using the GPU and predicts the model by using the check_point

    :param buffer: Reshaped numpy array for the first stream network
    :param color_buffer: Reshaped numpy array of the colored video for the second stream network

    :return prediction: The prediction from the model in the form of numpy array with 13 classes
    """

    with torch.no_grad():
        out = model(buffer, color_buffer)
        out = (out.numpy().flatten())
        prediction = np.round_(out, decimals=3)
    # print(prediction)
    return prediction


def write_csv(frame, csv_file_name, prediction):
    """
    With the index number and the prediction as a list saves in the csv file

    :param index: Frame number
    :param csv_file_name: Name of the csv file to load the list
    :param  data: The predicted data from the model for the corresponding 16 frames
    """

    index_with_prediction = [frame]
    index_with_prediction.extend(prediction)
    with open(csv_file_name, 'a', newline='') as file:
        writer_object = writer(file)
        writer_object.writerow(index_with_prediction)
        file.close()


def main():
    temp_direct = len(glob.glob("/home/navaneethan/dataset/train/*/*"))
    folder_list = ["checking_tires", "looking_at_phone", "playing_guitar", "shaking_head", "deadlifting",
                   "putting_on_shoes",
                   "moving_furniture", "pull_ups", "excercing", "playing_badminton", "push_up", "jogging",
                   "playing_chess"]
    foldering = ["deadlifting","putting_on_shoes","playing_chess"]

    folder_iterate = "/home/navaneethan/dataset/train/"
    directory_list = glob.glob("/home/navaneethan/dataset/train/*/*")
    no_of_file = len(directory_list)
    count = 1
    for folder in foldering:

        new_directory = os.path.join(folder_iterate, folder, "*")
        lets_iterate = glob.glob(new_directory)
        for directory in lets_iterate:
            print(directory)
            start = time.time()
            color_buffer_file, csv_file_name = file_split(directory)
            buffer, color_buffer, frame = convert_numpy(directory, color_buffer_file)
            frame_count = buffer.shape[1] - 16
            for frame in range(frame_count):
                trim_buffer, trim_color_buffer = trim_frame(buffer, color_buffer, frame)
                prediction = predict(trim_buffer, trim_color_buffer)
                # print(prediction)
                write_csv(frame, csv_file_name, prediction)
            end = time.time()
            print(f"[{temp_direct}]({count}/{no_of_file})Total time consumption for prediction is {end - start})")
            count += 1


if __name__ == '__main__':
    main()
