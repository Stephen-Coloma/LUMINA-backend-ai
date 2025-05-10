# ==== Utility Imports ====
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute(all_targets, all_preds, running_loss, total_samples):
    """
    Performs model evaluation by calculating the average loss, accuracy,
    precision, recall, f1 score, and a confusion matrix from the model's
    prediction and actual target values.

    Args:
    :param all_targets: The real target values.
    :param all_preds: The predicted values from the model.
    :param running_loss: The sum of losses.
    :param total_samples: The total samples used.
    :return: None
    """
    avg_loss = running_loss / total_samples
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')
    conf_mtx = confusion_matrix(all_targets, all_preds)

    return avg_loss, accuracy, precision, recall, f1, conf_mtx

def log(results, logger, is_training=True):
    """
    Saves the results from a log file using a logger object.
    The logger will save various metric results obtained
    from the performance of the model.

    Args:
    :param results: A list containing the results from each metrics.
    :param logger: An object to facilitate logging.
    :param is_training: Flag indicating if the log is meant for training or validation.
    :return: None
    """
    avg_loss, accuracy, precision, recall, f1, conf_mtx = results

    header = 'TRAINING RESULTS:' if is_training else 'VALIDATION RESULTS:'
    logger.info(
        f'\n{header}\n'
        f'> Avg Loss: {avg_loss:.4f}\n'
        f'> Accuracy: {accuracy:.4f}\n'
        f'> Precision: {precision:.4f}\n'
        f'> Recall: {recall:.4f}\n'
        f'> F1 Score: {f1:.4f}\n'
        'Confusion Matrix:\n'
        f'{conf_mtx}'
    )