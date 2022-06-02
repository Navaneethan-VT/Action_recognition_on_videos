import glob
import os
from multiprocessing import Pool
from numpy import asarray
from numpy import savez_compressed
import cv2
import numpy as np


def numpy(files_list):
    print(files_list)
    filing = files_list.split("/")

    directory = filing[0] + "/" + filing[1] + "/" + filing[2] + "/" + filing[3] + "/compress/" + filing[4]
    print(directory)
    new_file = os.path.join(directory, filing[-1])
    print(new_file)
    # head_original_file = os.path.split(files_list)
    # file_name = head_original_file[1]
    # classes = os.path.split(head_original_file[0])
    # class_name = classes[1]
    # split_up = os.path.split(classes[0])
    # split_name = split_up[1]
    # directory = split_up[0]
    # new_directory = os.path.join(directory, "numpy_compressed", split_name, class_name)
    # if not os.path.exists(new_directory):
    #     os.makedirs(new_directory)
    # new_file = os.path.join(new_directory, file_name)
    resize_height = 128
    resize_width = 172
    capture = cv2.VideoCapture(files_list)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buffer = np.empty((frame_count, resize_height, resize_width, 3), np.dtype('float32'))

    count = 0
    retaining = True

    while count < frame_count and retaining:
        retaining, frame = capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if (frame_height != resize_height) or (frame_width != resize_width):
            frame = cv2.resize(frame, (resize_width, resize_height))
        buffer[count] = frame
        count += 1

    capture.release()
    buffer = buffer.transpose((3, 0, 1, 2))
    buffer = asarray(buffer)

    savez_compressed(new_file, buffer)


def main():
    files_list = glob.glob("/home/navaneethan/dataset/putting_on_shoes_color/*")
    # color_name = "/home/navaneethan/dataset/dataset/train_color"
    # for files in files_list:
    #     splits = files.split("/")
    #     name_split = splits[-1].split(".")
    #     new_name = name_split[0] + ".avi"
    #     dude = color_name + "/" + splits[-2] + "/" + new_name
    #     print(dude)
    # if not os.path.exists(dude):
    #     color(files)
    p = Pool(60)
    result = p.map(func=numpy, iterable=files_list)
    p.close()
    p.join()
    # for file in files_list:
    #     numpy(file)


if __name__ == '__main__':
    main()
