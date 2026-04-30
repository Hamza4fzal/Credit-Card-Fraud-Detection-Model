import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(filepath='creditcard.csv'):
    """Loads the dataset."""
    df = pd.read_csv(filepath)
    return df

def clean_and_scale_data(df):
    """Cleans the data and scales Time and Amount features."""
    # Drop any duplicates to clean the data
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Scale Time and Amount
    scaler_time = StandardScaler()
    scaler_amount = StandardScaler()
    
    df['Time'] = scaler_time.fit_transform(df['Time'].values.reshape(-1, 1))
    df['Amount'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # Save scalers for later use during prediction (in Streamlit)
    joblib.dump(scaler_time, 'scaler_time.pkl')
    joblib.dump(scaler_amount, 'scaler_amount.pkl')
    
    return df

def main():
    print("Loading raw data...")
    df = load_data('creditcard.csv')
    
    print("Cleaning and scaling data...")
    df_cleaned = clean_and_scale_data(df)
    
    print("Saving cleaned data to cleaned_creditcard.csv ...")
    df_cleaned.to_csv('cleaned_creditcard.csv', index=False)
    
    print("Preprocessing complete. Cleaned data saved as 'cleaned_creditcard.csv'.")

if __name__ == "__main__":
    main()
