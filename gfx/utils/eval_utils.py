import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from astropy.stats import jackknife
import matplotlib.pyplot as plt
import seaborn as sns

def save_cm(predict, label, label_name, save_path):
    cm = metrics.multilabel_confusion_matrix(label, predict)
    row = 5
    col = len(label_name) // row
    fig, ax = plt.subplots(row, col, figsize=(7, 8))
    for axes, cfs_matrix, label in zip(ax.flatten(), cm, label_name):
        print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    axes.set_ylabel('True')
    axes.set_xlabel('Predicted')
    axes.set_title(class_label)

def compute_accuracy_metrics(ground_truth, predicted, threshold=0.5):
    decisions = predicted > threshold
    binary_pred = decisions.astype(np.int16)
    return metrics.classification_report(ground_truth, binary_pred, output_dict=True)

def get_binary_decisions(ground_truth, predicted, best_f1=True):
    """https://github.com/MTG/mtg-jamendo-dataset/blob/31507d6e9a64da11471bb12168372db4f98d7783/scripts/mediaeval/calculate_decisions.py#L8"""
    thresholds = {}
    for idx in range(len(ground_truth[0])):
        precision, recall, threshold = metrics.precision_recall_curve(
            ground_truth[:, idx], predicted[:, idx])
        f_score = np.nan_to_num(
            (2 * precision * recall) / (precision + recall))
        thresholds[idx] = threshold[np.argmax(f_score)]

    if best_f1:
        decisions = predicted > np.array(list(thresholds.values()))
        decisions = decisions.astype("float32")
    else:
        decisions = predicted > 0.5
        decisions = decisions.astype("float32")

    sample_f1 = metrics.f1_score(ground_truth, decisions, average='samples')
    macro_f1 = metrics.f1_score(ground_truth, decisions, average='macro')
    return sample_f1, macro_f1, decisions, thresholds
