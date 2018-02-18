import numpy as np
import pandas as pd
import os
import cv2
import scipy.io as spio
import finding_face_landmark
from skimage import io


def generate_file_with_features(input_dir, output_file):
    for file in os.listdir(input_dir):
        if (file == "wiki.mat"):
            continue
        print(file)
        input_dir_new = input_dir + "\\" + file
        for image in os.listdir(input_dir_new):
            print(image)
            features = finding_face_landmark.finding_face_landmark(input_dir_new + "\\" + image)
            if len(features) == 0:
                continue
            row = str(features).strip("[]")
            row += ", " + file + "/" + image  + "\n"
            fd = open(output_file, 'a')
            fd.write(row)
            fd.close()


def get_data(file_name, n):
    fd = pd.read_csv(file_name, header=None)
    data = fd.as_matrix()

    X = data[:n, :15].astype(np.float32)           # get features
    Q = data[:n, 15:]                               # get labelse
    Y = np.identity(len(X))                        # create identy matrix

    return  X,Y,Q


def get_min_max(X):
    x_max = X.max()
    x_min = X.min()
    return  x_min, x_max


def normalize_features(x_min, x_max, features):
    return (features - x_min) / (x_max - x_min)      # convert to [0,1]


def get_percent(input, predict):
    percents = []
    sum_percent = 0

    for i in range(0, len(input)):
        percent = input[i] * 100 / predict[i]
        if (percent > 100):
            percent = 100- (percent - 100)
        percents.append(percent)

    for percent in percents:
       sum_percent += percent

    percent = float(sum_percent) / 15
    print(percent)
    return percent


def get_name(path, metadata_path):
    mat = spio.loadmat(metadata_path, squeeze_me=True)
    name = mat['wiki']['name'].tolist()
    full_path = mat['wiki']['full_path'].tolist()
    print(path)
    index = full_path.tolist().index(path.strip())
    print(index)
    print(name[index])
    return name[index]


def show_image(image_path, name, percent):
    image = io.imread(image_path)
    cv2.putText(image, name, (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    cv2.putText(image, "{0:.2f}".format(percent), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_data_test(test_file_name, x_min, x_max, n, Q, Y):
    T, P, L = get_data(test_file_name, 100)
    T = normalize_features(x_min, x_max, T)
    P = np.identity(n)

    for i in range(0, len(L)):
        item_index = np.where(Q == L[i])
        print(item_index)
        index = list(item_index)[0][0]
        print(index)
        P[i] = Y[index]

    P = P[:T.shape[0], :len(Y)]
    return T,P,L
