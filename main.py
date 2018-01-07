import face_detection, finding_face_landmark, projecting_faces



def main():
    file_name="face.jpg"
    crop_img=face_detection.face_detection(file_name)

    finding_face_landmark.finding_face_landmark(crop_img)
    #projecting_faces.projecting_faces(file_name)


main()