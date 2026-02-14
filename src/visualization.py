import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, save_path='training_history.png'):
    """Plot training and validation history."""
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', linewidth=2)
    plt.plot(val_accuracies, label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, classes, save_path='confusion_matrix.png'):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_actual_vs_predicted(y_true, y_pred_encoded, label_encoder, save_path='actual_vs_predicted.png'):
    """Plot actual vs predicted comparison."""
    y_true_decoded = label_encoder.inverse_transform(y_true)
    y_pred_decoded = label_encoder.inverse_transform(y_pred_encoded)
    
    label_to_num = {'negative_shift': -1, 'stable': 0, 'positive_shift': 1}
    actual_num = [label_to_num.get(l, 0) for l in y_true_decoded]
    pred_num = [label_to_num.get(l, 0) for l in y_pred_decoded]
    
    indices = range(len(y_true))
    
    plt.figure(figsize=(15, 6))
    plt.plot(indices, actual_num, 'o-', label='Actual', alpha=0.7)
    plt.plot(indices, pred_num, 's--', label='Predicted', alpha=0.7)
    plt.yticks([-1, 0, 1], ['Negative', 'Stable', 'Positive'])
    plt.title('Actual vs Predicted Sentiment Shifts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
