# 🏛️ Political Sentiment Shift Prediction using Neural Networks

## 📋 Project Overview

This comprehensive data science project implements a state-of-the-art **Political Sentiment Shift Prediction System** using:
- **Deep Learning**: LSTM Neural Networks for time-series forecasting
- **NLP**: DistilBERT transformer model for sentiment analysis
- **Deployment**: Interactive Gradio interface for real-time predictions

---

## 🎯 Project Objectives

1. Analyze political news articles and predict sentiment shifts
2. Build a time-series forecasting model using LSTM/GRU networks
3. Deploy an interactive web interface for real-time predictions
4. Achieve high accuracy in predicting positive, negative, or stable sentiment shifts

---

## 📊 Complete Pipeline (20 Steps)

### **Data Preparation**
1. ✅ Import all required libraries (PyTorch, Transformers, NLTK, Gradio, etc.)
2. ✅ Read political news dataset
3. ✅ Comprehensive data exploration (head, tail, shape, dtypes, describe)
4. ✅ Data quality checks (duplicates, missing values, outliers)
5. ✅ Text preprocessing (cleaning, tokenization, lemmatization)

### **Feature Engineering**
6. ✅ Sentiment analysis using DistilBERT transformer
7. ✅ Aggregate sentiments by date for time-series analysis
8. ✅ Create shift labels (positive_shift, stable, negative_shift)
9. ✅ Input-output split with sequence creation
10. ✅ Train-test split with stratification

### **Model Development**
11. ✅ Build LSTM neural network architecture
12. ✅ Hyperparameter configuration and tuning
13. ✅ Model training with early stopping
14. ✅ Predictions on test data
15. ✅ Comprehensive evaluation (Accuracy, F1, Confusion Matrix)

### **Visualization & Deployment**
16. ✅ Plot actual vs predicted trends
17. ✅ Shift probability distribution analysis
18. ✅ Save model and artifacts
19. ✅ Build Gradio interface
20. ✅ Deploy prediction system

---

## 🚀 How to Run in Google Colab

### **Step 1: Upload to Colab**

```python
# Open Google Colab: https://colab.research.google.com/
# Create a new notebook
# Upload the political_sentiment_prediction_colab.py file
```

### **Step 2: Run the Code**

```python
# Option A: Copy and paste the entire code into a cell and run

# Option B: Upload file and run
!python political_sentiment_prediction_colab.py
```

### **Step 3: Access the Gradio Interface**

After running, you'll see:
```
Running on public URL: https://xxxxxxxx.gradio.live
```

Click the URL to access the interactive interface!

---

## 🛠️ System Requirements

### **Required Libraries**
```python
# Core libraries
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0

# NLP libraries
nltk>=3.8.0
transformers>=4.30.0

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# Machine Learning
scikit-learn>=1.2.0

# Deployment
gradio>=3.35.0
```

### **Installation Command**
```bash
pip install pandas numpy matplotlib seaborn nltk transformers torch scikit-learn gradio
```

---

## 📁 Project Structure

```
political_sentiment_prediction/
│
├── political_sentiment_prediction_colab.py  # Main script (all 20 steps)
├── README.md                                # This file
│
├── model_artifacts/                         # Saved models
│   ├── sentiment_shift_model.pth           # Trained LSTM model
│   ├── scaler.pkl                          # Feature scaler
│   ├── label_encoder.pkl                   # Label encoder
│   └── metadata.json                       # Model metadata
│
└── visualizations/                          # Generated plots
    ├── content_length_analysis.png
    ├── sentiment_distribution.png
    ├── time_series_sentiment.png
    ├── shift_labels.png
    ├── training_history.png
    ├── confusion_matrix.png
    ├── actual_vs_predicted.png
    └── probability_distribution.png
```

---

## 🎨 Key Features

### **1. Advanced Text Preprocessing**
- Lowercasing and cleaning
- Stopword removal
- Lemmatization
- URL and special character removal

### **2. Transformer-Based Sentiment Analysis**
- Uses DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
- Analyzes both content and titles
- Generates normalized sentiment scores (-1 to +1)

### **3. Time-Series Engineering**
- Aggregates daily sentiment scores
- Calculates statistical features (mean, std, min, max)
- Creates sequence-based inputs for LSTM

### **4. LSTM Neural Network**
- 2-layer LSTM architecture
- Dropout regularization
- Fully connected classification head
- Early stopping to prevent overfitting

### **5. Interactive Gradio Interface**
- Select political topics
- Choose forecast horizon (1-7 days)
- Real-time predictions with visualizations
- Confidence scores and probability distributions

---

## 📊 Model Architecture

```
SentimentShiftPredictor(
  (lstm): LSTM(input_size=7, hidden_size=64, num_layers=2, batch_first=True, dropout=0.3)
  (dropout): Dropout(p=0.3)
  (fc1): Linear(in_features=64, out_features=32)
  (relu): ReLU()
  (fc2): Linear(in_features=32, out_features=3)
)

Total Parameters: ~50,000
```

---

## 📈 Expected Performance

Based on the sample dataset:

| Metric | Value |
|--------|-------|
| **Accuracy** | 65-75% |
| **F1-Score** | 0.60-0.70 |
| **Training Time** | 2-5 minutes |
| **Inference Time** | <1 second |

*Note: Performance improves with larger datasets*

---

## 🎯 Use Cases

1. **Political Campaign Analysis**: Monitor public sentiment during elections
2. **Policy Impact Assessment**: Predict public reaction to policy changes
3. **Media Monitoring**: Track sentiment shifts in news coverage
4. **Crisis Management**: Early warning system for negative sentiment trends
5. **Strategic Planning**: Inform communication strategies based on predicted shifts

---

## 📝 Sample Output

### **Prediction Summary**
```
📊 POLITICAL SENTIMENT SHIFT PREDICTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 Topic: General Politics
📅 Forecast Horizon: 3 days

🔮 PREDICTION RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Predicted Shift: POSITIVE SHIFT
Confidence: 72.45%

📈 Probability Breakdown:
• Positive Shift: 72.45%
• Stable: 18.32%
• Negative Shift: 9.23%

💡 INTERPRETATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ The model predicts a POSITIVE sentiment shift
   This suggests:
   • Increasing public approval
   • Favorable news coverage
   • Potential improvement in political climate

🎯 High confidence (72.45%) - Prediction is reliable
```

---

## 🔧 Customization Options

### **1. Adjust Hyperparameters**
```python
# In the code, modify these values:
HIDDEN_SIZE = 128        # Increase for more complex patterns
NUM_LAYERS = 3           # Add more LSTM layers
DROPOUT = 0.4            # Adjust regularization
LEARNING_RATE = 0.0005   # Fine-tune learning rate
SEQUENCE_LENGTH = 7      # Change time window
```

### **2. Use Different Sentiment Models**
```python
# Replace DistilBERT with other models:
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"  # For Twitter data
)
```

### **3. Add More Features**
```python
# Extend feature_columns:
feature_columns = [
    'sentiment_mean', 'sentiment_std',
    'word_count', 'author_diversity',  # New features
    'topic_entropy', 'engagement_score'
]
```

---

## 🐛 Troubleshooting

### **Issue 1: CUDA Out of Memory**
```python
# Solution: Reduce batch size
BATCH_SIZE = 2  # Instead of 8
```

### **Issue 2: Gradio Not Launching**
```python
# Solution: Use different port
iface.launch(share=True, server_port=7860)
```

### **Issue 3: Model Not Learning**
```python
# Solutions:
# 1. Increase epochs
EPOCHS = 100

# 2. Adjust learning rate
LEARNING_RATE = 0.001

# 3. Check data quality
print(df['shift_label'].value_counts())  # Should be balanced
```

---

## 📚 Additional Resources

### **Learn More About:**
- [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)
- [Gradio Documentation](https://www.gradio.app/docs)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

### **Datasets for Training:**
- [Kaggle Political Datasets](https://www.kaggle.com/datasets?search=political)
- [Twitter API](https://developer.twitter.com/en/docs) for real-time data
- [News API](https://newsapi.org/) for news articles

---

## 🤝 Contributing

To improve this project:
1. Collect larger, more diverse datasets
2. Experiment with different architectures (GRU, Transformer-based)
3. Add more features (social media metrics, economic indicators)
4. Implement ensemble methods
5. Deploy to cloud platforms (AWS, GCP, Azure)

---

## 📄 License

This project is for educational purposes. Modify and use as needed for your research or applications.

---

## 👨‍💻 Author

**Data Science Project**
- Built with ❤️ using Python, PyTorch, and Transformers
- Optimized for Google Colab
- Complete 20-step implementation

---

## 🎓 Citation

If you use this code in your research or project, please cite:

```bibtex
@software{political_sentiment_prediction,
  title = {Political Sentiment Shift Prediction using Neural Networks},
  author = {Your Name},
  year = {2024},
  description = {LSTM-based system for predicting sentiment shifts in political discourse}
}
```

---

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Experiment with hyperparameters
4. Ensure all dependencies are installed

---

## 🌟 Key Takeaways

✅ **Complete Pipeline**: All 20 steps from data loading to deployment
✅ **Production-Ready**: Includes model saving, loading, and deployment
✅ **Interactive**: User-friendly Gradio interface
✅ **Well-Documented**: Extensive comments and documentation
✅ **Scalable**: Easy to extend with more data and features
✅ **Educational**: Perfect for learning deep learning and NLP

---

**Happy Predicting! 🚀**

*Last Updated: February 2026*
