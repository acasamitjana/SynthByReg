import numpy as np
from scipy.spatial.distance import dice

def dice_coefficient(y_true, y_pred):
    """
    Computes the Sorensen-Dice metric
                    TP
        Dice = 2 -------
                  T + P
    Parameters
    ----------
    y_true : numpy.array
        Binary representation
    y_pred : keras.placeholder
        Binary representation
    Returns
    -------
    scalar
        Dice metric
    """
    initial_shape = y_pred.shape

    y_pred = y_pred > 0.5
    y_true = y_true > 0.5

    y_pred_flatten = y_pred.reshape(-1,1)
    y_true_flatten = y_true.reshape(-1,1)

    dice_score_negated = dice(y_true_flatten, y_pred_flatten)

    return 1 - dice_score_negated

def gdice_coefficient(y_true, y_pred):
    """
    Computes the Sorensen-Dice metric
                    TP
        Dice = 2 -------
                  T + P
    Parameters
    ----------
    y_true :
        Binary representation
    y_pred :
        Binary representation
    Returns
    -------
    scalar
        Dice metric
    """
    initial_shape = y_pred.shape

    y_pred = y_pred > 0.5
    y_true = y_true > 0.5

    num_classes = y_pred.shape[0]

    y_pred_flatten = y_pred.reshape(num_classes, -1)
    y_true_flatten = y_true.reshape(num_classes, -1)

    numerator = 0
    denominator = 0
    for it_c in range(num_classes):
        w = np.nan_to_num(1 / np.sum(y_true_flatten[it_c]),0)
        numerator += w * np.sum(y_pred_flatten[it_c]*y_true_flatten[it_c])
        denominator += w * (np.sum(y_pred_flatten[it_c]) + np.sum(y_true_flatten[it_c]))

    dice_score = 2 * numerator/denominator

    return dice_score
