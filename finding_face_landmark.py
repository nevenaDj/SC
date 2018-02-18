import sys
import dlib
from skimage import io
#import openface
import cv2
import math


def euclidean_distance(point1, point2):
    distance = pow((point1.x - point2.x), 2) + pow((point1.y - point2.y), 2)
    return math.sqrt(distance)


def finding_face_landmark(file_name):
    # You can download the required pre-trained face detection model here:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    predictor_model = "shape_predictor_68_face_landmarks.dat"


    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    #face_aligner = openface.AlignDlib(predictor_model)

    win = dlib.image_window()

    # Load the image
    image = io.imread(file_name)

    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)

    print("Found {} faces in the image file {}".format(len(detected_faces), image))
    if (len(detected_faces) != 1):
        print("On the photo, there are more faces. Please try out with different photo. ")
        return []
      #  exit(0)

    # Show the desktop window with the image
    win.set_image(image)

    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):
        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                                 face_rect.right(), face_rect.bottom()))

        # Draw a box around each face we found
        win.add_overlay(face_rect)

        # Get the the face's pose
        pose_landmarks = face_pose_predictor(image, face_rect)

        leftEye1 = pose_landmarks.part(42)
        rightEye1 = pose_landmarks.part(39)
        nose = pose_landmarks.part(30)
        noseTip = pose_landmarks.part(27)
        mouth = pose_landmarks.part(62)
        noseLeft = pose_landmarks.part(31)
        noseRight = pose_landmarks.part(35)

        right1 = pose_landmarks.part(1)
        left1 = pose_landmarks.part(15)
        right2 = pose_landmarks.part(4)
        left2 = pose_landmarks.part(12)
        right3 = pose_landmarks.part(6)
        left3 = pose_landmarks.part(10)

        leftEye2 = pose_landmarks.part(45)
        rightEye2 = pose_landmarks.part(36)

        #for j in range(1, 68):
        #   pos = pose_landmarks.part(j)
        #   cv2.circle(image, (pos.x, pos.y), 1, (0, 0, 255), -1)

        #cv2.circle(image, (leftEye1.x, leftEye1.y), 1, (255, 255, 0), -1)
        #cv2.circle(image, (rightEye1.x, rightEye1.y), 1, (255, 255, 0), -1)
        #cv2.circle(image, (nose.x, nose.y), 1, (255, 255, 0), -1)
        #cv2.circle(image, (noseTip.x, noseTip.y), 1, (255, 255, 0), -1)
        #cv2.circle(image, (mouth.x, mouth.y), 1, (255, 255, 0), -1)
        #cv2.circle(image, (noseLeft.x, noseLeft.y), 1, (255, 255, 0), -1)
        #cv2.circle(image, (noseRight.x, noseRight.y), 1, (255, 255, 0), -1)
        #cv2.circle(image, (right1.x, right1.y), 1, (255, 255, 0), -1)
        #cv2.circle(image, (right2.x, right2.y), 1, (255, 255, 0), -1)
        #cv2.circle(image, (right3.x, right3.y), 1, (255, 255, 0), -1)
        #cv2.circle(image, (left1.x, left1.y), 1, (255, 255, 0), -1)
        #cv2.circle(image, (left2.x, left2.y), 1, (255, 255, 0), -1)
        #cv2.circle(image, (left3.x, left3.y), 1, (255, 255, 0), -1)
        #cv2.circle(image, (leftEye2.x, leftEye2.y), 1, (255, 255, 0), -1)
        #cv2.circle(image, (rightEye2.x, rightEye2.y), 1, (255, 255, 0), -1)

        #cv2.line(image, (leftEye1.x, leftEye1.y), (rightEye1.x, rightEye1.y), (255, 255, 0), 1)
        #cv2.line(image, (leftEye1.x, leftEye1.y), (mouth.x, mouth.y), (255, 255, 0), 1)
        #cv2.line(image, (rightEye1.x, rightEye1.y), (mouth.x, mouth.y), (255, 255, 0), 1)
        #cv2.line(image, (leftEye1.x, leftEye1.y), (nose.x, nose.y), (255, 255, 0), 1)
        #cv2.line(image, (rightEye1.x, rightEye1.y), (nose.x, nose.y), (255, 255, 0), 1)
        #cv2.line(image, (mouth.x, mouth.y), (nose.x, nose.y), (255, 255, 0), 1)
        #cv2.line(image, (noseTip.x, noseTip.y), (nose.x, nose.y), (255, 255, 0), 1)
        #cv2.line(image, (noseLeft.x, noseLeft.y), (noseRight.x, noseRight.y), (255, 255, 0), 1)
        #cv2.line(image, (left1.x, left1.y), (right1.x, right1.y), (255, 255, 0), 1)
        #cv2.line(image, (left2.x, left2.y), (right2.x, right2.y), (255, 255, 0), 1)
        #cv2.line(image, (left3.x, left3.y), (right3.x, right3.y), (255, 255, 0), 1)
        #cv2.line(image, (left1.x, left1.y), (leftEye2.x, leftEye2.y), (255, 255, 0), 1)
        #cv2.line(image, (right1.x, right1.y), (rightEye2.x, rightEye2.y), (255, 255, 0), 1)
        #cv2.line(image, (leftEye1.x, leftEye1.y), (leftEye2.x, leftEye2.y), (255, 255, 0), 1)
        #cv2.line(image, (rightEye1.x, rightEye1.y), (rightEye2.x, rightEye2.y), (255, 255, 0), 1)

        d1 = euclidean_distance(leftEye1, rightEye1)    # distance between the eyes
        d2 = euclidean_distance(leftEye1, mouth)        # distance between middle of the left eyes and middle point of mouth
        d3 = euclidean_distance(rightEye1, mouth)       # distance between middle of the right eyes and middle point of mouth
        d4 = euclidean_distance(leftEye1, nose)         # distance between middle of the left eyes and middle point of nose
        d5 = euclidean_distance(rightEye1, nose)        # distance between middle of the rigth eyes and middle point of nose
        d6 = euclidean_distance(mouth, nose)            # distance between middle point of mouth and middle point of nose
        d7 = euclidean_distance(noseTip, nose)          # distance of middle point of d1 and middle of nose
        d8 = euclidean_distance(noseLeft, noseRight)    # width of nose
        d9 = euclidean_distance(left1, right1)          # width of face
        d10 = euclidean_distance(left2, right2)         # width of face
        d11 = euclidean_distance(left3, right3)         # width of face
        d12 = euclidean_distance(leftEye1, leftEye2)    # width od left eye
        d13 = euclidean_distance(rightEye1, rightEye2)  # width of right eye
        d14 = euclidean_distance(left1, leftEye2)
        d15 = euclidean_distance(right1, rightEye2)

        features = []

        features.append(d1)
        features.append(d2)
        features.append(d3)
        features.append(d4)
        features.append(d5)
        features.append(d6)
        features.append(d7)
        features.append(d8)
        features.append(d9)
        features.append(d10)
        features.append(d11)
        features.append(d12)
        features.append(d13)
        features.append(d14)
        features.append(d15)

        # Draw the face landmarks on the screen.
        win.add_overlay(pose_landmarks)

        #  Use openface to calculate and perform the face alignment
        #  alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        #  Save the aligned image to a file
        #  cv2.imwrite("aligned_face_{}.jpg".format(i), alignedFace)
        #  cv2.imshow("Output", image)
        #  cv2.waitKey(0)
        dlib.hit_enter_to_continue()
        return features