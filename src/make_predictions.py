import pickle
from joblib import load
import pandas as pd
from preprocess_test_data import preprocess_test_data

model=load('./final_xgb_model.pkl')
scaler=load('./scaler.pkl')
features_used=load('./features_used.pkl')
features_scaled=load('./features_scaled.pkl')
test_df=pd.read_csv(open('../test.csv', 'rb'))

def make_predictions(scaler, features_used, features_scaled, test_df):

    features_df=preprocess_test_data(test_df, scaler, features_used, features_scaled)
    predictions=model.predict(features_df)
    predictions_df=pd.DataFrame.from_dict({'id':test_df['id'], 'Listening_Time_minutes': predictions})
    predictions_df.to_csv('../submission.csv', index=False)


if __name__ == '__main__':

    make_predictions(scaler, features_used, features_scaled, test_df)