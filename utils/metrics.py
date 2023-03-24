import numpy as np
from sklearn import metrics


def pixel_level_metrics_per_image(output, mask):
    groundtruth = (mask > 0).astype(np.int).flatten()
    groundtruth_inv = np.logical_not(groundtruth)

    pixel_auc = metrics.roc_auc_score(groundtruth, output.flatten())
    
    threshold = 0.5
    predicition = (output > threshold).astype(np.int).flatten()
    predicition_inv = np.logical_not(predicition)

    true_pos = float(np.logical_and(predicition, groundtruth).sum())
    false_neg = float(np.logical_and(predicition_inv, groundtruth).sum())
    false_pos = float(np.logical_and(predicition, groundtruth_inv).sum())

    pixel_f1_th05 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    pixel_precision_th05 = true_pos / (true_pos + false_pos + 1e-6)
    pixel_recall_th05 = true_pos / (true_pos + false_neg + 1e-6)

    return pixel_auc, pixel_precision_th05, pixel_recall_th05, pixel_f1_th05
    

def pixel_level_metrics_per_image_inv(output, mask):
    groundtruth = (mask > 0).astype(np.int).flatten()
    groundtruth_inv = np.logical_not(groundtruth)

    try:
        pixel_auc = metrics.roc_auc_score(groundtruth, output.flatten())
    except:
        pixel_auc = 0.

    try:
        pixel_auc_inv = metrics.roc_auc_score(groundtruth_inv, output.flatten())
    except:
        pixel_auc_inv = 0.
    
    pixel_auc = max(pixel_auc, pixel_auc_inv)
    
    threshold = 0.5
    predicition = (output > threshold).astype(np.int).flatten()
    predicition_inv = np.logical_not(predicition)

    true_pos = float(np.logical_and(predicition, groundtruth).sum())
    false_neg = float(np.logical_and(predicition_inv, groundtruth).sum())
    false_pos = float(np.logical_and(predicition, groundtruth_inv).sum())

    pixel_f1_th05 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    pixel_precision_th05 = true_pos / (true_pos + false_pos + 1e-6)
    pixel_recall_th05 = true_pos / (true_pos + false_neg + 1e-6)

    true_pos_inv = float(np.logical_and(predicition, groundtruth_inv).sum())
    false_neg_inv = float(np.logical_and(predicition_inv, groundtruth_inv).sum())
    false_pos_inv = float(np.logical_and(predicition, groundtruth).sum())

    pixel_f1_th05_inv = 2 * true_pos_inv / (2 * true_pos_inv + false_pos_inv + false_neg_inv + 1e-6)
    pixel_precision_th05_inv = true_pos_inv / (true_pos_inv + false_pos_inv + 1e-6)
    pixel_recall_th05_inv = true_pos_inv / (true_pos_inv + false_neg_inv + 1e-6)
    
    if pixel_f1_th05 < pixel_f1_th05_inv:
        pixel_f1_th05 = pixel_f1_th05_inv
        pixel_precision_th05 = pixel_precision_th05_inv
        pixel_recall_th05 = pixel_recall_th05_inv

    return pixel_auc, pixel_precision_th05, pixel_recall_th05, pixel_f1_th05
    

    
def image_level_metrics(image_scores, image_labels, threshold=0.5):
    predicition = (np.array(image_scores) > threshold).astype(np.int)
    groundtruth = (np.array(image_labels) > 0).astype(np.int)
    predicition_inv, groundtruth_inv = np.logical_not(predicition), np.logical_not(groundtruth)
    
    true_pos = float(np.logical_and(predicition, groundtruth).sum())
    false_neg = float(np.logical_and(predicition_inv, groundtruth).sum())
    false_pos = float(np.logical_and(predicition, groundtruth_inv).sum())
    true_neg = float(np.logical_and(predicition_inv, groundtruth_inv).sum())
    
    image_acc = (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos + 1e-6)
    image_sensitivity = true_pos / (true_pos + false_neg + 1e-6)
    image_specificity = true_neg / (true_neg + false_pos + 1e-6)
    image_f1 = 2 * image_sensitivity * image_specificity / (image_sensitivity + image_specificity)

    try:
        image_auc = metrics.roc_auc_score(groundtruth, image_scores)
    except:
        image_auc = 0.
    
    return image_acc, image_auc, image_sensitivity, image_specificity, image_f1


def pixel_level_metrics(pixel_aucs, pixel_precision_th05s, pixel_recall_th05s, pixel_f1_th05s):
    pixel_auc = np.mean(np.array(pixel_aucs))
    pixel_precision_th05 = np.mean(np.array(pixel_precision_th05s))
    pixel_recall_th05 = np.mean(np.array(pixel_recall_th05s))
    pixel_f1_th05 = np.mean(np.array(pixel_f1_th05s))

    return pixel_auc, pixel_precision_th05, pixel_recall_th05, pixel_f1_th05
    