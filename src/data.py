import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .config import *

def create_sample_dataset():
    """Create sample political news dataset from provided data."""
    data = {
        'author': [
            'The Quint', 'The Quint', 'The Quint', 'The Quint', 'The Quint',
            'Pushpesh Pant', 'Pushpesh Pant', 'Pushpesh Pant', 'Pushpesh Pant',
            'Meeran Chadha Borwankar', 'Meeran Chadha Borwankar', 'Meeran Chadha Borwankar',
            'Mayukh Ghosh', 'Mayukh Ghosh', 'Akanksha Kumar', 'Akanksha Kumar',
            'The Quint', 'Pushpesh Pant', 'Meeran Chadha Borwankar', 'Mayukh Ghosh'
        ],
        'content': [
            'Tulsi Gabbard considering run for American presidency in 2020. Democrat lawmaker from Hawaii receives standing ovation.',
            'Gabbard broke with Democratic Party establishment and endorsed Bernie Sanders instead of Hillary Clinton in 2016.',
            'Critics question Gabbards views on foreign policy and tolerance of dictators like Syrian leader Bashar al-Assad.',
            'Gabbard was one of few to criticise US governments decision to deny visa to Narendra Modi after Gujarat riots.',
            'Gabbard opposed resolution calling for religious freedom in United States-India Strategic Dialogue.',
            'Swami Vivekananda ranks among highest in pantheon of distinguished children of Mother India. His charismatic personality electrified audiences.',
            'Vivekananda was arguably the first and most successful brand ambassador of resurgent India in USA.',
            'Congress and opposition parties desperately trying to get foothold in hearts and minds of voters who proclaim adherence to Sanatana Dharma.',
            'Strategists working overtime on marketing Soft Hinduism to counter machinations of Hindutva Brigade.',
            'Central Bureau of Investigation attracting public attention for misdemeanor of past directors and alleged discord.',
            'Criminal cases against former CBI directors Ranjit Sinha and AP Singh cast shadow over bureau.',
            'High-powered panel for selection of CBI director includes Prime Minister, Leader of Opposition, Chief Justice of India.',
            'Mrinal Sen started shouting out his own politics with cult-classic at time when Naxalite movement was at peak.',
            'Sen developed new language for Indian filmmakers giving birth to Indian New Wave influenced by French New Wave.',
            'Former Delhi Police Joint Commissioner recalls leading probe into murder of Naina Sahni that shocked entire country.',
            'Sushil Sharma former Youth Congress leader shot wife dead and tried to burn body. Supreme Court commuted death sentence.',
            'Political tensions rise as election campaign intensifies across major states with fierce debates.',
            'Religious freedom and tolerance remain critical issues in current political discourse and policy making.',
            'Government institutions face credibility crisis amid allegations of political interference and corruption.',
            'Cultural and artistic movements continue to reflect and shape sociopolitical consciousness of nation.'
        ],
        'date': [
            '2018-11-13', '2018-11-10', '2018-11-08', '2018-11-05', '2018-11-02',
            '2018-07-03', '2018-07-05', '2018-07-08', '2018-07-10',
            '2018-07-18', '2018-07-20', '2018-07-22',
            '2018-05-14', '2018-05-16',
            '2018-04-30', '2018-05-02',
            '2018-10-28', '2018-06-30', '2018-07-25', '2018-05-20'
        ],
        'tag': [
            'Tulsi Gabbard', 'Democratic Party', 'Foreign Policy', 'India Relations', 'Religious Freedom',
            'Swami Vivekananda', 'Hinduism', 'Politics of religion', 'Hindu Monks',
            'Modi Government', 'Central Bureau of Investigation', 'Indian Police Service',
            'Political Satire', 'Indian New Wave',
            'Supreme Court of India', 'Life Imprisonment',
            'Elections', 'Religious Tolerance', 'Government Corruption', 'Political Cinema'
        ],
        'title': [
            'Tulsi Gabbard May Run for Presidency',
            'Gabbard Endorses Bernie Sanders',
            'Foreign Policy Views Questioned',
            'Modi Visa Criticism',
            'Religious Freedom Resolution',
            'Vivekananda Birth Anniversary',
            'Brand Ambassador of India',
            'Soft Hinduism Strategy',
            'Political Strategy Shift',
            'CBI Controversy',
            'Directors Tainted Cases',
            'Selection Procedure Issues',
            'Mrinal Sen Political Cinema',
            'Indian New Wave Birth',
            'Tandoor Murder Case',
            'Death Sentence Commuted',
            'Election Campaign Heats Up',
            'Religious Discourse',
            'Institutional Crisis',
            'Cultural Movements'
        ],
        'url': [
            'https://www.thequint.com/news/1', 'https://www.thequint.com/news/2',
            'https://www.thequint.com/news/3', 'https://www.thequint.com/news/4',
            'https://www.thequint.com/news/5', 'https://www.thequint.com/voices/1',
            'https://www.thequint.com/voices/2', 'https://www.thequint.com/voices/3',
            'https://www.thequint.com/voices/4', 'https://www.thequint.com/voices/5',
            'https://www.thequint.com/voices/6', 'https://www.thequint.com/voices/7',
            'https://www.thequint.com/entertainment/1', 'https://www.thequint.com/entertainment/2',
            'https://www.thequint.com/videos/1', 'https://www.thequint.com/videos/2',
            'https://www.thequint.com/news/6', 'https://www.thequint.com/voices/8',
            'https://www.thequint.com/voices/9', 'https://www.thequint.com/entertainment/3'
        ],
        'website': ['quint'] * 20
    }
    return pd.DataFrame(data)

def process_data(df):
    """Clean and prepare initial dataframe."""
    df = df.copy()
    # Handle duplicates and missing values
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
        
    df['content'].fillna('', inplace=True)
    df['title'].fillna('Untitled', inplace=True)
    df['tag'].fillna('General', inplace=True)
    
    # Feature engineering
    df['content_length'] = df['content'].str.len()
    
    # Date conversion
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    return df

def aggregate_time_series(df):
    """Aggregate sentiment scores by date to create time-series dataset."""
    time_series_df = df.groupby('date').agg({
        'sentiment_score': ['mean', 'std', 'min', 'max', 'count'],
        'combined_sentiment': ['mean', 'std'],
        'content_length': 'mean'
    }).reset_index()

    # Flatten column names
    time_series_df.columns = ['date', 'sentiment_mean', 'sentiment_std', 'sentiment_min', 
                              'sentiment_max', 'article_count', 'combined_sentiment_mean', 
                              'combined_sentiment_std', 'avg_content_length']

    # Fill missing dates and interpolate
    date_range = pd.date_range(start=time_series_df['date'].min(), 
                               end=time_series_df['date'].max(), 
                               freq='D')
    time_series_df = time_series_df.set_index('date').reindex(date_range)
    time_series_df = time_series_df.interpolate(method='linear').bfill().ffill()
    time_series_df = time_series_df.reset_index().rename(columns={'index': 'date'})
    
    # Fill remaining NaNs and Infs
    time_series_df = time_series_df.fillna(0)
    time_series_df = time_series_df.replace([np.inf, -np.inf], 0)
    
    return time_series_df

def create_shift_labels(df, forecast_horizon=FORECAST_HORIZON, threshold=SENTIMENT_CHANGE_THRESHOLD):
    """Create shift labels based on future sentiment changes."""
    df = df.copy()
    df['future_sentiment'] = df['sentiment_mean'].shift(-forecast_horizon)
    df['sentiment_change'] = df['future_sentiment'] - df['sentiment_mean']
    
    def classify_shift(change):
        if pd.isna(change):
            return 'stable'
        elif change > threshold:
            return 'positive_shift'
        elif change < -threshold:
            return 'negative_shift'
        else:
            return 'stable'
    
    df['shift_label'] = df['sentiment_change'].apply(classify_shift)
    df = df.dropna(subset=['future_sentiment'])
    return df

def create_sequences(data, features, target, sequence_length=SEQUENCE_LENGTH):
    """Create sequences for LSTM/GRU input."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[features].iloc[i:i+sequence_length].values)
        y.append(data[target].iloc[i+sequence_length])
    
    X = np.nan_to_num(np.array(X))
    y = np.array(y)
    return X, y

def prepare_dataloaders(X, y, batch_size=BATCH_SIZE, test_size=TEST_SIZE):
    """Split data and create PyTorch DataLoaders."""
    # Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    # Convert to Tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    
    # DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler, label_encoder, X_train, X_test, y_train, y_test
