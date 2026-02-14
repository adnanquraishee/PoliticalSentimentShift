import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import *
from src.utils import set_seed, setup_environment
from src.data import (create_sample_dataset, process_data, aggregate_time_series, 
                      create_shift_labels, create_sequences, prepare_dataloaders)
from src.preprocessing import TextPreprocessor, SentimentAnalyzer
from src.model import SentimentShiftPredictor, Trainer
from src.visualization import plot_training_history, plot_confusion_matrix, plot_actual_vs_predicted

def main():
    # Setup
    setup_environment()
    set_seed(RANDOM_STATE)
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    # 1. Data Loading & Preprocessing
    print("\n[Stage 1] Loading and Preprocessing Data...")
    df = create_sample_dataset()
    df = process_data(df)
    
    # Text Preprocessing & Sentiment Analysis
    preprocessor = TextPreprocessor()
    df['processed_content'] = df['content'].apply(preprocessor.preprocess)
    
    analyzer = SentimentAnalyzer()
    df = analyzer.analyze_dataframe(df)

    # 2. Time Series Creation
    print("\n[Stage 2] Creating Time Series Data...")
    time_series_df = aggregate_time_series(df)
    time_series_df = create_shift_labels(time_series_df)
    
    # 3. Sequence Creation for LSTM
    feature_columns = ['sentiment_mean', 'sentiment_std', 'sentiment_min', 
                       'sentiment_max', 'combined_sentiment_mean', 
                       'combined_sentiment_std', 'avg_content_length']
    
    X, y = create_sequences(time_series_df, feature_columns, 'shift_label', SEQUENCE_LENGTH)
    
    # 4. Prepare DataLoaders
    train_loader, test_loader, scaler, label_encoder, X_train, X_test, y_train, y_test = \
        prepare_dataloaders(X, y)
        
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 5. Model Initialization
    print("\n[Stage 3] Initializing Model...")
    input_size = X_train.shape[2]
    num_classes = len(label_encoder.classes_)
    
    model = SentimentShiftPredictor(input_size, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 6. Training Loop
    print("\n[Stage 4] Starting Training...")
    trainer = Trainer(model, criterion, optimizer, scheduler, device)
    
    # Create directory for saving models
    os.makedirs('model_artifacts', exist_ok=True)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    best_model_path = 'model_artifacts/best_model.pth'
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.validate(test_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

    # Save last model as backup
    torch.save(model.state_dict(), 'model_artifacts/last_model.pth')

    # 7. Evaluation & Visualization
    print("\n[Stage 5] Evaluating & Visualizing...")
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Final Evaluation - Load Best Model
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("Warning: Best model not found. Using current model state.")
    model.eval()
    
    all_preds = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            
    # Metrics
    cm = confusion_matrix(y_test, all_preds)
    plot_confusion_matrix(cm, label_encoder.classes_)
    plot_actual_vs_predicted(y_test, all_preds, label_encoder)
    
    print("\n" + "="*50)
    print("Project Execution Completed Successfully!")
    print(f"Final Accuracy: {accuracy_score(y_test, all_preds):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
