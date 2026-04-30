import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def main():
    print("Loading cleaned data...")
    # Load the cleaned dataset
    try:
        df = pd.read_csv('cleaned_creditcard.csv')
    except FileNotFoundError:
        print("Error: 'cleaned_creditcard.csv' not found. Please run preprocessing.py first.")
        return

    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    print("Splitting data into train and test sets...")
    # Split the data BEFORE applying SMOTE to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Handling imbalanced data with SMOTE on training set...")
    # Apply SMOTE to the training set to balance the classes
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Resampled training set shape: {X_train_resampled.shape}")

    print("Training model with GridSearchCV...")
    # We use Logistic Regression as it's fast and effective for this dataset
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Define the hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    }
    
    # Setup GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=2)
    
    # Fit the model on the resampled data
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    print(f"Best hyperparameters found: {grid_search.best_params_}")
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    print("Evaluating model on test set...")
    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    
    # Print evaluation metrics
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Saving the best model to 'model.pkl'...")
    # Save the trained model for use in Streamlit app
    joblib.dump(best_model, 'model.pkl')
    print("Model training and saving complete!")

if __name__ == "__main__":
    main()
