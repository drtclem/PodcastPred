def preprocess_test_data(df_test, scaler, features_used, features_scaled):
    import numpy as np
    import pandas as pd

    df = df_test.copy()

    # --- Episode Sentiment mapping ---
    df['Episode_Sentiment'] = df['Episode_Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})

    # --- One-hot encode Genre ---
    df = pd.get_dummies(df, columns=['Genre'], drop_first=True)
    for col in [c for c in features_used if c.startswith('Genre_')]:
        if col not in df.columns:
            df[col] = 0
    genre_cols = [col for col in df.columns if col.startswith('Genre_')]
    df[genre_cols] = df[genre_cols].astype(int)

    # --- Cyclical encoding: Day of Week ---
    day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                   'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['Publication_Day'] = df['Publication_Day'].map(day_mapping)
    df['day_sin'] = np.sin(2 * np.pi * df['Publication_Day'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['Publication_Day'] / 7)

    # --- Cyclical encoding: Time of Day ---
    df['Publication_Time'] = df['Publication_Time'].str.strip().str.lower()
    time_mapping = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
    df['time_num'] = df['Publication_Time'].map(time_mapping)
    df['time_sin'] = np.sin(2 * np.pi * df['time_num'] / 4)
    df['time_cos'] = np.cos(2 * np.pi * df['time_num'] / 4)
    df.drop(columns=['time_num'], inplace=True)

    # --- Clean up Episode_Title ---
    df['Episode_Title'] = df['Episode_Title'].str.replace('Episode ', '', regex=False).astype(int)

    # --- Drop unused columns ---
    for col in ['Podcast_Name', 'Publication_Day', 'Publication_Time']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # --- Fill missing Guest_Popularity with 0 (assumed no guest) ---
    if 'Guest_Popularity_percentage' in df.columns:
        df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(0)

    # --- Standardize numeric features using the provided scaler ---
    df[features_scaled] = scaler.transform(df[features_scaled])

    # --- Ensure all expected columns are present ---
    for col in features_used:
        if col not in df.columns:
            df[col] = 0  # fill missing dummy or engineered features

    # --- Final reordering to match training data ---
    df = df[features_used]

    return df
