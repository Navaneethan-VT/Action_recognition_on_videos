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
import os
from multiprocessing import Pool

import cv2
import numpy as np
from csv import writer

out = cv2.VideoWriter('face_human_test.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (568, 320))


def detect(file_name):
    """
    Detects human body and human face using haar cascade
    :param file_name: File address for detection
    :return: list of frame numbers with starting and ending position
    """
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(file_name)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    main_list = [file_name]
    lists = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pos_frame = video_capture.get(cv2.CAP_PROP_POS_FRAMES)

        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        people = False
        peoples = 0
        faces = 0
        face = False

        for (xA, yA, xB, yB) in boxes:
            # display the detected boxes in the colour picture
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            people = True
            peoples += 1

        bound = face_cascade.detectMultiScale(gray,
                                              scaleFactor=1.1,
                                              minNeighbors=5,
                                              minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in bound:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            face = True
            faces += 1

        if people or face:
            lists.append(int(pos_frame))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out.write(frame)

        cv2.imshow('Video', frame)
    maxi = max(lists)
    mini = min(lists)

    if not lists:
        print("empty")
        minimum = 0
        maximum = frame_count

    else:
        if (maxi - mini) < 40:
            minimum = 0
            maximum = frame_count
        else:
            maximum = max(lists)
            minimum = min(lists)

    main_list.append(minimum)
    main_list.append(maximum)
    video_capture.release()
    cv2.destroyAllWindows()
    out.release()

    return main_list,


def write_csv(data):
    """
    create a csv file and load the starting frame and ending frame
    :param data: list of starting frame and ending frame number
    """

    with open('trim.csv', 'a', newline='') as file:
        writer_object = writer(file)
        writer_object.writerow(data)
        file.close()


def combine(directory):
    """
    Passes file address and calls the methods
    :param directory: file address

    """
    main_list = detect(directory)
    write_csv(main_list)


def main():
    file = "/home/navaneethan/working/playing_badminton_train.mp4"
    combine(file)
    directory = "/home/navaneethan/dataset"
    folders = ["train", "val"]
    for fold in folders:
        video_directory = os.path.join(directory, fold, "*/*")
        files = glob.glob(video_directory)
        p = Pool(60)
        result = p.map(func=combine, iterable=files)
        p.close()
        p.join()


if __name__ == "__main__":
    main()
