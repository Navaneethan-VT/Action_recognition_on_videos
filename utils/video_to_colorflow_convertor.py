import glob
import os
from multiprocessing import Pool

import cv2
import numpy as np


def color(files_list):

    filing = files_list.split("/")
    directory = filing[0] + "/" + filing[1] + "/" + filing[2] + "/" + filing[3] + "/" + filing[4]+"_color"

    name = filing[-1].split(".")
    new_name = name[0] + ".avi"
    new_file = os.path.join(directory, new_name)

    capture = cv2.VideoCapture(files_list)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(new_file, fourcc, fps, (frame_width, frame_height))

    suc, prev = capture.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        suc, img = capture.read()
        if not suc:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray

        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]

        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx * fx + fy * fy)

        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 1] = 255
        hsv[..., 2] = np.minimum(v * 4, 255)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        out.write(bgr)

    capture.release()
    out.release()


def main():

    files_list = glob.glob("/home/navaneethan/dataset/putting_on_shoes/*.mp4")
    p = Pool(60)
    result = p.map(func=color, iterable=files_list)
    p.close()
    p.join()


if __name__ == '__main__':
    main()
