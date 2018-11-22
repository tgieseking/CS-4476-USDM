import glob
import os.path
import numpy as np
from getSift import saveSIFT
from vocabulary import createVocab, createHist
from splitting import split_train_test, split_seed_active, scale_train_test

imagedir = '../../coil-100/'

obj_nums = range(1, 101)
angles = range(0, 360, 5)
num_obj = 100
num_angle = 72

sample_images_per_class = 10
sample_features_per_image = 10
vocab_size = 1000

image_paths = np.asarray([[imagedir + "obj" + str(obj_num) + "__" + str(angle) + ".png" for angle in angles] for obj_num in obj_nums])

sift_paths = np.asarray([[imagedir + "obj" + str(obj_num) + "__" + str(angle) + ".npy" for angle in angles] for obj_num in obj_nums])

print("Starting SIFTing")

# for obj_num in range(obj_num):
#     for angle in range(num_angle):
#         saveSIFT(image_paths[obj_num, angle], sift_paths[obj_num, angle])

random_features = []
for obj_index in range(num_obj):
    sample = np.random.choice(num_angle, sample_images_per_class, replace=False)
    for angle_index in sample:
        sift = np.load(sift_paths[obj_index, angle_index])
        num_rows = sift.shape[0]
        if num_rows < sample_features_per_image:
            for feature_index in range(num_rows):
                random_features.append(sift[feature_index])
        else:
            feature_sample = np.random.choice(num_rows, sample_features_per_image, replace=False)
            for feature_index in feature_sample:
                random_features.append(sift[feature_index])
random_features = np.asarray(random_features)

print("Starting clustering")

vocab = createVocab(random_features, vocab_size)

print("Starting histogram creation")

hists = []
labels = []

for obj_index in range(obj_num):
    for angle_index in range(num_angle):
        sift = np.load(sift_paths[obj_index, angle_index])
        if sift.size > 1:
            hists.append(createHist(sift, vocab, vocab_size))
        else:
            hists.append(np.zeros(vocab_size))
        labels.append(obj_index)
hists = np.asarray(hists)
labels = np.asarray(labels)

print("Starting splitting")

X_train, X_test, y_train, y_test = split_train_test(hists, labels)
X_train_scaled, X_test_scaled = scale_train_test(X_train, X_test)
X_seed, X_active, y_seed, y_active = split_seed_active(X_train_scaled, y_train)

np.save("../data/COIL_X_seed", X_seed)
np.save("../data/COIL_X_active", X_active)
np.save("../data/COIL_X_test", X_test_scaled)
np.save("../data/COIL_y_seed", y_seed)
np.save("../data/COIL_y_active", y_active)
np.save("../data/COIL_y_test", y_test)

import pdb; pdb.set_trace()
