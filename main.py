import finding_face_landmark
import utils as utils
import neural_network as n


def main():
    print("Image:")
    image = input("")

    features = finding_face_landmark.finding_face_landmark(image)
    print(features)
    if (len(features) == 0):
        exit(0)

    data_file_name = "features.csv"
    X,Y,Q = utils.get_data(data_file_name, 2000)

    x_min, x_max = utils.get_min_max(X)
    X = utils.normalize_features(x_min, x_max, X)

    test_file_name = "test.csv"
    T,P,L = utils.get_data_test(test_file_name, x_min, x_max, len(X), Q, Y)

    model_file_name = './my_test_model.ckpt'
    neural_network = n.Neural_Network(X,Y, model_file_name)
    # neural_network.training()
    # neural_network.test(T,P)

    features = utils.normalize_features(x_min, x_max, features)

    predict = neural_network.predict([features])
    image_path = Q[predict][0].strip()

    metadata = 'C:\\ProjekatSoft\\wiki_crop\\wiki.mat'
    name = utils.get_name(image_path, metadata)

    percent = utils.get_percent(features, X[predict:predict+1, :15][0])
    utils.show_image('C:\\ProjekatSoft\\wiki_crop\\' + image_path, name, percent)

    #  images_path = 'C:\\ProjekatSoft\\wiki_crop'
    #  file_name = 'features2.csv'
    #  fd = open(file_name, 'w')
    #  utils.generate_file_with_features(images_path, file_name)


main()