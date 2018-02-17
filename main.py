import face_detection, finding_face_landmark, projecting_faces
import utils as utils
import neural_network as n


def main():
    image = "C:\\ProjekatSoft\\wiki_crop\\00\\305500_1940-09-18_1963.jpg"
    features = finding_face_landmark.finding_face_landmark(image)
    print(features)

    data_file_name = "features.csv"
    X,Y,Q, x_min, x_max = utils.get_data(data_file_name)

    model_file_name = './my_test_model.ckpt'
    neural_network = n.Neural_Network(X,Y, model_file_name)
    #  neural_network.training()

    features = utils.normalize_features(x_min, x_max, features)
    print(features)

    predict = neural_network.predict([features])
    image_path = Q[predict][0]
    name = utils.get_name(image_path, 'C:\\ProjekatSoft\\wiki_crop\\wiki.mat')

    percent = utils.get_percent(features, X[predict:predict+1, :15][0])
    utils.show_image('C:\\ProjekatSoft\\wiki_crop\\' + image_path, name, percent)

    # fd = open('features2.csv', 'w')
    # utils.generate_file_with_features('C:\\ProjekatSoft\\wiki_crop', 'features2.csv')

    #crop_img=face_detection.face_detection(file_name)      - integrated in landmark
    #finding_face_landmark.finding_face_landmark(file_name) # with face_detection and projecting faces
    #projecting_faces.projecting_faces(file_name)


main()