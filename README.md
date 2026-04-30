# Credit Card Fraud Detection System

This project is a machine learning-based web application built with Streamlit to detect fraudulent credit card transactions. The dataset used is highly imbalanced, containing a small percentage of fraudulent transactions among valid ones. The project involves comprehensive data preprocessing, handling class imbalance, hyperparameter tuning, and an interactive UI for prediction.

## Project Structure

*   `preprocessing.py`: Cleans the dataset, removes duplicates, and scales features (`Time` and `Amount`).
*   `model_training.py`: Handles imbalanced data using SMOTE, performs hyperparameter tuning using `GridSearchCV`, and trains the final Machine Learning model.
*   `visualization.py`: Generates visualizations of the dataset (e.g., class distribution).
*   `app.py`: The Streamlit web application providing a user-friendly UI to simulate and predict transactions and store transaction logs.

## Technologies and Techniques Used

### 1. Data Preprocessing & Cleaning
*   **StandardScaler:** Used to scale the `Time` and `Amount` features so that they have a mean of 0 and a standard deviation of 1. This is crucial for models like Logistic Regression that are sensitive to the scale of input features.
*   **Deduplication:** Removing duplicate rows ensures the model does not overfit to repeated patterns.

### 2. Handling Imbalanced Data
*   **SMOTE (Synthetic Minority Over-sampling Technique):** The dataset is highly imbalanced (very few fraud cases). SMOTE generates synthetic samples for the minority class (fraud) based on the feature space of existing fraud cases. We apply SMOTE *only to the training set* to prevent data leakage and ensure accurate real-world evaluation on the test set.

### 3. Machine Learning Model & Training
*   **Model Chosen (Logistic Regression):** A Logistic Regression model was chosen. It is efficient, interpretable, and performs exceptionally well on binary classification tasks like fraud detection when the data is appropriately scaled and balanced. While Random Forest or XGBoost could be used, Logistic Regression allows for faster training during the Grid Search process, making it ideal for this university project presentation.
*   **Hyperparameter Tuning (GridSearchCV):** `GridSearchCV` is used to exhaustively search through a specified parameter grid (e.g., regularization strength `C` and the `solver` algorithm) to find the most optimal hyperparameters for the Logistic Regression model. It uses Cross-Validation to ensure the model generalizes well.

### 4. User Interface
*   **Streamlit:** Used to build a clean, interactive, and simple UI. It allows you to:
    *   Load dummy valid or fraudulent transactions from the dataset.
    *   Modify the transaction amount manually.
    *   Click "Predict" to see if the model flags it as fraud.
*   **Transaction Logging:** The app automatically logs every prediction into `transaction_logs.csv` and flags if there are anomalies (fraud) detected in the sidebar.

## How to Run the Project

1.  **Install Requirements:** Make sure you have the required libraries installed:
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn streamlit joblib
    ```

2.  **Data Preprocessing:**
    Run the preprocessing script to clean and scale the data. This will generate `cleaned_creditcard.csv` and scaler objects.
    ```bash
    python preprocessing.py
    ```

3.  **Visualization (Optional):**
    Run the visualization script to generate a plot of the class distribution.
    ```bash
    python visualization.py
    ```

4.  **Model Training:**
    Train the machine learning model. This applies SMOTE, runs GridSearchCV, and saves the best model as `model.pkl`.
    ```bash
    python model_training.py
    ```

5.  **Run the UI (Streamlit App):**
    Start the Streamlit application.
    ```bash
    streamlit run app.py
    ```
    This will open the web interface in your default browser.

## Using the Dashboard
During your presentation, you can demonstrate the model by clicking the **"Generate Dummy VALID Transaction"** or **"Generate Dummy FRAUDULENT Transaction"** buttons. This will populate the features automatically. You can change the "Transaction Amount" manually and hit **"Predict Transaction Status"** to see how the model reacts to different inputs and logs the outcomes.
