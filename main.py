import face_detection, finding_face_landmark, projecting_faces



def main():
    print("Keydet")
    file_name="face.jpg"
    face_detection.face_detection(file_name)

    finding_face_landmark.finding_face_landmark(file_name)
    print("veg")
    #projecting_faces(file_name)


main()