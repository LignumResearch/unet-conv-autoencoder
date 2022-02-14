import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier
from lib import metrics

def fit(latent, labels):
    latent_dim = latent.shape[1]
    
    seed = 199
    np.random.seed(seed)
    split = 0.1

    latent = np.load('latent.npy')
    labels = np.load('labels_train.npy')

    x_train, x_test, y_train, y_test = train_test_split(latent, labels, test_size = split, 
        random_state = seed, stratify = labels)

    clf = DummyClassifier(random_state=seed)

    clf.fit(x_train, y_train.argmax(1))

    pred = clf.predict(x_test)

    metrics.save(y_test.argmax(1), pred, "dummy", latent_dim)