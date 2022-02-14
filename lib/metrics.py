from sklearn.metrics import *
from csv import writer
from os import getcwd
from os.path import exists, basename

def save(true, pred, clf_name, latent_dim):
    
    # check if scores report exists, insert header
    if not exists('../scores.csv'):
        with open('../scores.csv', 'w') as f:
            w = writer(f)
            w.writerow([
                "dataset",
                "latent_dim",
                "classifier",
                "accuracy",
                "balanced_accuracy",
                # "top_k_accuracy",
                # "average_precision",
                # "f1",
                "f1_micro",
                "f1_macro",
                "f1_weighted",
                # "precision",
                "precision_micro",
                "precision_macro",
                "precision_weighted",
                # "recall",
                "recall_micro",
                "recall_macro",
                "recall_weighted",
                # "jaccard",
                "jaccard_micro",
                "jaccard_macro",
                "jaccard_weighted",
                # "roc_auc_micro_ovr",
                # "roc_auc_macro_ovr",
                # "roc_auc_weighted_ovr",
                # "roc_auc_micro_ovo",
                # "roc_auc_macro_ovo",
                # "roc_auc_weighted_ovo",
            ])

    # scores
    scores = [
        basename(getcwd()),                                 # dataset
        latent_dim,                                         # latent_dim
        clf_name,                                                   # classifier
        accuracy_score(true, pred),                         # accuracy
        balanced_accuracy_score(true, pred),                # balanced_accuracy
        # top_k_accuracy_score(true, pred),                   # top_k_accuracy
        # average_precision_score(true, pred),                # average_precision
        # f1_score(true, pred),                               # f1
        f1_score(true, pred, average='micro'),              # f1_micro
        f1_score(true, pred, average='macro'),              # f1_macro
        f1_score(true, pred, average='weighted'),           # f1_weighted
        # precision_score(true, pred),                        # precision
        precision_score(true, pred, average='micro', zero_division=0),       # precision_micro
        precision_score(true, pred, average='macro', zero_division=0),       # precision_macro
        precision_score(true, pred, average='weighted', zero_division=0),    # precision_weighted
        # recall_score(true, pred),                           # recall
        recall_score(true, pred, average='micro'),          # recall_micro
        recall_score(true, pred, average='macro'),          # recall_macro
        recall_score(true, pred, average='weighted'),       # recall_weighted
        # jaccard_score(true, pred),                          # jaccard
        jaccard_score(true, pred, average='micro'),         # jaccard_micro
        jaccard_score(true, pred, average='macro'),         # jaccard_macro
        jaccard_score(true, pred, average='weighted'),      # jaccard_weighted
        # roc_auc_score(true, pred, average='micro', multi_class='ovr'),         # roc_auc_micro ovr config
        # roc_auc_score(true, pred, average='macro', multi_class='ovr'),         # roc_auc_macro ovr config
        # roc_auc_score(true, pred, average='weighted', multi_class='ovr'),      # roc_auc_weighted ovr config
        # roc_auc_score(true, pred, average='micro', multi_class='ovo'),         # roc_auc_micro ovo config
        # roc_auc_score(true, pred, average='macro', multi_class='ovo'),         # roc_auc_macro ovo config
        # roc_auc_score(true, pred, average='weighted', multi_class='ovo'),      # roc_auc_weighted ovo config
    ]

    with open('../scores.csv', 'a') as f:
        w = writer(f)
        w.writerow(scores)
