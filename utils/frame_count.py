import os
import cv2
import glob


def check_video(file_name):
    """
    check all the frames and counts frames
    :param file_name: directory of the video_file
    :return frame_count: int value of total frame number in the video
    :return file_name: directory of the video_file
    """
    capture = cv2.VideoCapture(file_name)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    return frame_count, file_name


def delete_video(frame_count, file_name):
    """
    Deletes the video which has frames less than 100

    :param frame_count: int value of total frame number in the video
    :param file_name: directory of the video_file

    """
    if frame_count < 100:
        print(frame_count, file_name)
        os.remove(file_name)


def main():
    file = glob.glob("~/dataset/train/*/*.mp4")
    for files in file:
        frame_count, file_name = check_video(files)
        delete_video(frame_count, file_name)


if __name__ == '__main__':
    main()
