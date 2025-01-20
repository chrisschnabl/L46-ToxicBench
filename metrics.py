import numpy as np
import torch
from torchmetrics.classification import CalibrationError
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    matthews_corrcoef
)
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np


def compute_metrics(predictions, references):
    """
    Compute various classification metrics including ECE, MCE, MCC, and export raw data.
    
    predictions: 2D array of model outputs (logits or probabilities).
    references: 1D array of true labels (0 or 1).
    """
    # Convert predictions to probabilities and classes
    preds = np.argmax(predictions, axis=1)
    probabilities = predictions[:, 1]  # Assuming the second column corresponds to the positive class
    precision, recall, f1, _ = precision_recall_fscore_support(references, preds, average="binary")
    accuracy = accuracy_score(references, preds)
    
    tn, fp, fn, tp = confusion_matrix(references, preds).ravel()
    
    try:
        auc = roc_auc_score(references, probabilities)
    except ValueError:
        auc = None  # Handle cases where AUROC cannot be computed (e.g., single-class references)
    
    auprc = average_precision_score(references, probabilities)
    
    mcc = matthews_corrcoef(references, preds)
    
    ece_metric = CalibrationError(n_bins=10, norm='l1', task='binary')   # L1 norm for ECE
    mce_metric = CalibrationError(n_bins=10, norm='max', task='binary')  # Max norm for MCE
    probabilities_tensor = torch.tensor(probabilities)
    references_tensor = torch.tensor(references)
    
    ece = ece_metric(probabilities_tensor, references_tensor).item()
    mce = mce_metric(probabilities_tensor, references_tensor).item()
    
    # Return metrics and raw data
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "AUROC": auc,
        "AUPRC": auprc,
        "MCC": mcc,
        "ECE": ece,
        "MCE": mce,
        "raw_data": {
            "predictions": preds.tolist(),
            "probabilities": probabilities.tolist(),
            "references": references.tolist(),
        }
    }

def evaluate_model(model, dataset, collator, tokenizer):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collator)
    all_preds = []
    all_labels = []
    cnt = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = {k: v.to(model.device) for k, v in batch.items() if k in tokenizer.model_input_names}
            labels = batch["labels"].to(model.device)
            
            # Get model predictions
            outputs = model(**inputs)
            logits = outputs.logits
            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return compute_metrics(all_preds, all_labels)