import sys
import dlib
from skimage import io
import cv2
import openface
import tflearn
import sklearn
import face_detection, finding_face_landmark, projecting_faces


def main():
    file_name="http://all4desktop.com/data_images/original/4153786-barbara-palvin-8.jpg"
    face_detection(file_name)
    finding_face_landmark(file_name)
    projecting_faces(file_name)