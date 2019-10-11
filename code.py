import h5py
import cv2
import numpy as np
import os
from tqdm import tqdm

TRAIN_DIR = ""
TEST_DIR = ""


def get_data(Dir):
    X = []
    y = []
    for nextDir in os.listdir(Dir):
        if not nextDir.startswith('.'):
            if nextDir in ['name1']:
                label = 0
            elif nextDir in ['name2']:
                label = 1
            else:
                label = 2

            temp = Dir + nextDir

            for file in tqdm(os.listdir(temp)):
                img = cv2.imread(temp + '/' + file)
                if img is not None:
                    img = np.asarray(img)
                    X.append(img)
                    y.append(label)

    x = np.asarray(X)
    y = np.asarray(y)
    return x, y

X_train, y_train = get_data(TRAIN_DIR)
X_test, y_test = get_data(TEST_DIR)

print(X_train.shape,'\n',X_test.shape)
print(y_train.shape,'\n',y_test.shape)

f = h5py.File("train.h5")
f['x'] = X_train
f['y'] = y_train
f.close()

f = h5py.File("test.h5")
f['x'] = X_test
f['y'] = y_test
f.close()