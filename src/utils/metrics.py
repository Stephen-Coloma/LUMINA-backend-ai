from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def compute(all_targets, all_preds, running_loss, data_len):
    avg_loss = running_loss / data_len
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')
    conf_mtx = confusion_matrix(all_targets, all_preds)

    return (avg_loss, accuracy, precision, recall, f1, conf_mtx)

def log(results, logger, is_training=True):
    avg_loss, accuracy, precision, recall, f1_score, conf_mtx = results

    header = 'Training Results' if is_training else 'Validation Results'
    logger.info(
        f'\n{header}\n'
        f'> Avg Loss: {avg_loss:.4f}\n'
        f'> Accuracy: {accuracy:.4f}\n'
        f'> Precision: {precision:.4f}\n'
        f'> Recall: {recall:.4f}\n'
        f'> F1 Score: {f1_score:.4f}\n'
        'Confusion Matrix:\n'
        f'{conf_mtx}'
    )