#!/usr/bin/env python
import os.path, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from opensfm import dataset
from opensfm import features
from opensfm import csfm


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        usage()

    data = dataset.DataSet(path)
    images = data.images()
    db = []
    with open('training_features.txt', 'a') as fout:
        for image in images:
            print image
            p, f = features.read_feature(data.feature_file(image))
            print f.shape
            db.append(f)
    db = np.concatenate(db)
    print db.shape
    k = 100000
    words, labels = csfm.approximate_k_means(db, np.zeros(0), k, max_iterations = 0)
    iterations = 50
    step = 1
    for i in range(0,iterations, step):
        words, labels = csfm.approximate_k_means(db, words, k, max_iterations = step)
        np.save('words.%04d'%i, words)

        frequencies = np.zeros(len(words), dtype=int)
        for l in labels:
            frequencies[l] += 1
        np.save('frequencies.%04d'%i, frequencies)

    np.save('words.npy', words)
    np.save('labels.npy', labels)
    np.save('frequencies.npy', frequencies)


