"""
Module Name: script.py
Description: This code is to facilitate the work in the Course CT2.0 of NUS High School.

@date: 15 Jan , 2026
@version: 0.91

"""
# Importing Necessary Libraries

# For time tracking
import time

# For data manipulation
import pandas as pd
import numpy as np
import random
# For file operations in Google Colab
from google.colab import files
import os

# For preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

# For deep learning with PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# For progress monitoring
from tqdm.notebook import tqdm

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For evaluation metrics
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix


# Global Variables

# Placeholder for the dataset. Will be assigned when the dataset is uploaded.
FULL_DATASET = None
PROCESSED_FULL_DATASET = None
FILE_NAME = None
SCALER = None
SCALER_Y = None
MODEL = None
CRITERION =  None
OPTIMIZER = None
TRAIN_DATA = None
TEST_DATA = None
WAIT_TIME = 1

# Random / Reproducibility Settings
USE_FIXED_SEED = False     # default: random each run
FIXED_SEED = 42            # seed used when fixed mode is enabled

# Constants for dataset format
ACTUAL_VALUE_COL = 0
TYPE_ROW = 0
CLASS_LABEL_MAP = {} # Mapping of class name to integer label
TASK_TYPE = 0
CUTOFF_FOR_LABEL_ENCODING = 5
M = 1
N = 1

def SetRandom(fixed: bool = False, seed: int = 42):
    """
    - fixed=False: random mode (non-reproducible)
    - fixed=True: reproducible using 'seed'
    """
    global USE_FIXED_SEED, FIXED_SEED
    USE_FIXED_SEED = fixed
    FIXED_SEED = int(seed)

    if USE_FIXED_SEED:
        _seed_everything(FIXED_SEED)
        print(f"[Random Setting] Fixed seed enabled: {FIXED_SEED}")
    else:
        print("[Random Setting] Random mode enabled (non-reproducible).")

def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Function Definitions

def FullDataSetUpload():
    """
    Function to upload and read the CSV dataset.
    """
    global FULL_DATASET
    uploaded_file = files.upload()
    # Assuming only one file is uploaded
    file_name = list(uploaded_file.keys())[0]
    print("here==="+file_name+"====here")
    FULL_DATASET = pd.read_csv(file_name)
    return "Dataset Uploaded Successfully!\n\n"

def FullDataSetValidation():
    """
    Function to validate the structure and format of the dataset.
    """
    global FULL_DATASET, TASK_TYPE

    # 1. Check second row for "classification", "regression", "numerical", or "categorical".
    type_row = FULL_DATASET.iloc[TYPE_ROW].values
    type_row =[_.strip() for _ in type_row]
    if type_row[ACTUAL_VALUE_COL] not in ["classification", "regression"]:
        raise ValueError("Leftmost column in second row must be 'classification' or 'regression'.")
    else:
        TASK_TYPE = type_row[ACTUAL_VALUE_COL]
    for feature_type in type_row[1:]:
        if feature_type not in ["numerical", "categorical"]:
            raise ValueError("Features in second row must be 'numerical' or 'categorical'.")

    # 2. Check data types based on the type row.
    for idx, col_name in enumerate(FULL_DATASET.columns):

        # Check for numerical data type
        if type_row[idx] == "numerical":
            for entry in FULL_DATASET[col_name][2:]:
                if not pd.isna(entry):
                    try:
                        # Handle negative numbers stored as text
                        float(str(entry).replace('-', '', 1))
                    except ValueError:
                        raise ValueError(f"Column {col_name} is marked as 'numerical' but contains non-numeric value: {entry}")


        # Check for categorical data type
        elif type_row[idx] == "categorical":
            for entry in FULL_DATASET[col_name][2:]:
                entry_str = str(entry)
                if '.' in entry_str and entry_str.replace('.', '', 1).isdigit():
                    raise ValueError(f"Column {col_name} is marked as 'categorical' but seems to contain a float value: {entry}")

    return f"Validation Successful! This is a {TASK_TYPE} Task.\n\n"

def ProcessFeatures():
    global FULL_DATASET, PROCESSED_FULL_DATASET

    # Extract data without the type row
    data_without_type = FULL_DATASET.drop(TYPE_ROW, axis=0)

    # Get type row
    type_row = FULL_DATASET.iloc[TYPE_ROW].values
    type_row =[_.strip() for _ in type_row]

    # Create empty DataFrame for processed features
    processed_data = pd.DataFrame()

    # Add actual values column (leftmost) to the processed_data
    processed_data[FULL_DATASET.columns[ACTUAL_VALUE_COL]] = data_without_type.iloc[:, ACTUAL_VALUE_COL].astype(str)  # Ensure labels are treated as strings

    # Iterate over columns to process them based on type
    for idx, col_name in enumerate(data_without_type.columns[1:], start=1):  # starting from 1 to skip the actual value column

        # Convert string numbers to actual numbers - work but deprecated method
        # data_without_type[col_name] = pd.to_numeric(data_without_type[col_name], errors='ignore')
        # Convert string numbers to actual numbers (avoid deprecated errors='ignore')
        try:
            data_without_type[col_name] = pd.to_numeric(data_without_type[col_name])
        except (ValueError, TypeError):
            # If conversion fails, keep original column (likely categorical text)
            pass
        # end of (avoid deprecated errors='ignore')
        # Handle missing values
        if type_row[idx] == "numerical":
            imputer = SimpleImputer(strategy='mean')
            data_without_type[col_name] = imputer.fit_transform(data_without_type[col_name].values.reshape(-1, 1)).ravel()

            # Standard Scaling for numerical features
            scaler = StandardScaler()
            processed_data[col_name] = scaler.fit_transform(data_without_type[col_name].values.reshape(-1, 1)).ravel()

        elif type_row[idx] == "categorical":
            imputer = SimpleImputer(strategy='most_frequent')
            data_without_type[col_name] = imputer.fit_transform(data_without_type[col_name].values.reshape(-1, 1)).ravel()

            # Check the number of unique categories
            unique_categories = data_without_type[col_name].nunique()

            if unique_categories > CUTOFF_FOR_LABEL_ENCODING:
                # Label Encoding for categorical features with more than 10 categories
                label_encoder = LabelEncoder()
                processed_data[col_name] = label_encoder.fit_transform(data_without_type[col_name])
            else:
                # One-Hot Encoding for categorical features with CUTOFF_FOR_LABEL_ENCODING or fewer categories
                one_hot = pd.get_dummies(data_without_type[col_name], prefix=col_name)
                processed_data = pd.concat([processed_data, one_hot], axis=1)

    # Update PROCESSED_FULL_DATASET with processed data
    PROCESSED_FULL_DATASET = processed_data
    return "Features Processed Successfully!\n\n"


def MakeClassLabels(msg):
    def _maybe_print(msg):
        if ignore:
            print(msg)
    global FULL_DATASET, CLASS_LABEL_MAP

    if FULL_DATASET.iloc[0, ACTUAL_VALUE_COL] == "classification":

        le = LabelEncoder()
        data_without_type = FULL_DATASET.drop(TYPE_ROW, axis=0).copy()

        if ACTUAL_VALUE_COL >= len(FULL_DATASET.columns):
            _maybe_print(f"Error: ACTUAL_VALUE_COL ({ACTUAL_VALUE_COL}) is out of bounds.")
            return

        try:
            le.fit(data_without_type.iloc[:, ACTUAL_VALUE_COL])
            CLASS_LABEL_MAP = dict(zip(le.classes_, le.transform(le.classes_)))
            _maybe_print (f"For this classification task, {len(CLASS_LABEL_MAP)} class labels mapped successfully, as shown below.")
            _maybe_print(CLASS_LABEL_MAP)
            return
        except Exception as e:
            _maybe_print(f"Error in label encoding: {e}")
            return

def GenereatingNewfiles():
    # Splitting the data
    global TRAIN_DATA, TEST_DATA

    random_state = FIXED_SEED if USE_FIXED_SEED else None
    TRAIN_DATA, TEST_DATA = train_test_split(PROCESSED_FULL_DATASET, test_size=0.2, random_state=random_state)


    # Saving to CSV
    train_file_path = 'training_data.csv'
    test_file_path = 'testing_data.csv'

    TRAIN_DATA.to_csv(train_file_path, index=False)
    TEST_DATA.to_csv(test_file_path, index=False)
    return "Files Generated Successfully!\n\n"

def FilesDownloading():


    try:
        # Trigger automatic download
        files.download('training_data.csv')
        files.download('testing_data.csv')
    except Exception as e:
        print("Error appeared at the FilesDownloading step. Please check your input or seek help from teachers.")
        print(f"Technical details: {e}")
        return


def DataPreProcessing(ignore=True):
    """
    ignore (bool):
      - True  -> keep current behavior (prints intermediate messages)
      - False -> suppress all prints except the final completion message
    """
    
    global WAIT_TIME
    if not ignore: WAIT_TIME = 0 
    else: WAIT_TIME = 1
    def _maybe_print(msg):
        if ignore:
            print(msg)
        # Step 1: FullDataSetUpload
    try:
        '''
        print(FullDataSetUpload())
        '''
        global FULL_DATASET
        FULL_DATASET = pd.read_csv("Full_DataSet.csv")
        _maybe_print("Dataset Uploaded Successfully!\n\n")


    except Exception as e:
        _maybe_print("Error appeared at the FullDataSetUpload step. Please check your input or seek help from teachers.")
        _maybe_print(f"Technical details: {e}")
        return
    _maybe_print("A preview of the DataSet as show below...")
    pd.set_option('display.max_rows', 5)
    pd.set_option('display.max_columns', 5)
    _maybe_print(FULL_DATASET)
    _maybe_print("Now let's validate the dataset...")
    time.sleep(WAIT_TIME)
    # Step 2: FullDataSetValidation
    try:
        _maybe_print(FullDataSetValidation())
    except Exception as e:
        _maybe_print("Error appeared at the FullDataSetValidation step. Please check your input or seek help from teachers.")
        _maybe_print(f"Technical details: {e}")
        return


    _maybe_print("Now let's process the features ...")
    time.sleep(WAIT_TIME)
    # Step 3: ProcessFeatures
    try:
        _maybe_print(ProcessFeatures())
        _maybe_print("A preview of the processed dataset as show below...")
        _maybe_print(PROCESSED_FULL_DATASET)
    except Exception as e:
        _maybe_print("Error appeared at the ProcessFeatures step. Please check your input CSV file or seek help from teachers.")
        _maybe_print(f"Technical details: {e}")
        return
    if TASK_TYPE == "classification":
        _maybe_print("Now let's us map the class names to class labels...")
        time.sleep(WAIT_TIME)

    # Step 4: MakeClassLabels
    if FULL_DATASET.iloc[0, ACTUAL_VALUE_COL] == "classification":
        try:
            MakeClassLabels(msg)
            _maybe_print("\n\n")
        except Exception as e:
            _maybe_print("Error appeared at the MakeClassLabels step. Please check your input or seek help from teachers.")
            _maybe_print(f"Technical details: {e}")
            return

    _maybe_print("Now let's generate files for trainign and testing...")
    time.sleep(WAIT_TIME)
    # Step 5: GenereatingNewfiles
    try:
        _maybe_print(GenereatingNewfiles())
    except Exception as e:
        _maybe_print("Error appeared at the GenereatingNewfiles step. Please check your input or seek help from teachers.")
        _maybe_print(f"Technical details: {e}")
        return

    print("Data Preprocessing completed successfully!")
#DataPreProcessing(file_name)
#filename = input("Please enter the filename on the, without quotaion marks:")
#print("testing", filename)
#DataPreProcessing(file_name)

# Step 2: Build the neural network
def layer_block(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(0.3)
    )


def layer_block(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(0.3)
    )

class DynamicNN(nn.Module):
    def __init__(self, size):
        super(DynamicNN, self).__init__()

        layers = []
        in_features = M

        for i in range(1, size+1):
            out_features = (((size+1-i)*M + i*N) // (size+1))
            layers.append(layer_block(in_features, out_features))
            in_features = out_features

        # Output layer
        self.fc_out = nn.Linear(in_features, N)

        # Combine all layers
        self.layers = nn.Sequential(*layers)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for block in self.layers:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    nn.init.zeros_(layer.bias)

        nn.init.kaiming_normal_(self.fc_out.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        x = self.layers(x)
        x = self.fc_out(x)
        if N == 1:  # Regression
            x = x.squeeze(-1)
        return x

# Define the training function
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def ModelTraining(number_hidden_layers):
    if type(number_hidden_layers) != int or number_hidden_layers > 4 or number_hidden_layers < 0 :
        print("Please enter an integer between 0 and 4, both inclusive")
        return
    if USE_FIXED_SEED:
        _seed_everything(FIXED_SEED)

    global FILE_NAME, SCALER, SCALER_Y, M, N, MODEL, CRITERION, OPTIMIZER, TRAIN_DATA

    # Load the data
    data = TRAIN_DATA.copy()
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    M = X.shape[1]  # Number of features

    # Handle class labels if it's a classification task
    if TASK_TYPE == 'classification':
        N = len(set(y))  # Number of classes
        # Check for missing labels in CLASS_LABEL_MAP
        unique_labels = set(y)
        missing_labels = [label for label in unique_labels if label not in CLASS_LABEL_MAP]
        if missing_labels:
            print("Missing labels in CLASS_LABEL_MAP:", missing_labels)
            return

        # Map class labels
        y = np.vectorize(CLASS_LABEL_MAP.get)(y)

        # Ensure y is of type int after mapping and check for None values
        if None in y:
            print("Error: Some labels were not mapped correctly, resulting in None values.")
            return

        # Ensure y is of type int after mapping
        try:
            y = y.astype(int)
        except ValueError as e:
            print("Error in converting class labels to integers:", e)
            return
        y_tensor = torch.tensor(y, dtype=torch.long)
    else:
        N = 1
        SCALER_Y = StandardScaler()
        y = SCALER_Y.fit_transform(y.reshape(-1, 1)).flatten()
        y_tensor = torch.tensor(y, dtype=torch.float32)

    SCALER = StandardScaler()
    X_train = SCALER.fit_transform(X)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    MODEL = DynamicNN(number_hidden_layers)

    CRITERION = nn.CrossEntropyLoss() if TASK_TYPE == 'classification' else nn.MSELoss()
    # CRITERION = nn.CrossEntropyLoss() if TASK_TYPE == 'classification' else nn.L1Loss()
    OPTIMIZER = optim.Adam(MODEL.parameters(), lr=0.001, weight_decay=0.01)

    # Step 3: Training
    num_epochs = min(int(input("Enter the number of epochs: ")), 1000)
    recording_interval = max(num_epochs // 20, 1)
    losses = []
    for epoch in tqdm(range(num_epochs), desc="Training"):
        train_loss = train_one_epoch(MODEL, train_loader, CRITERION, OPTIMIZER)
        if epoch % recording_interval == recording_interval - 1 or epoch == 0:
            losses.append(train_loss)
            print(f'Epoch {epoch+1:>5}/{num_epochs:>5}, Loss: {train_loss:>.4f}')

    # Step 4: Visualization
    plt.plot([i*recording_interval for i in range(len(losses))], losses)
    plt.xlabel('Epochs (every 5% of num_epochs)')
    plt.ylabel('Loss')
    plt.show()

def ModelEvaluation(DownLoad = True):
    # 1. Setup and Information from the Early Step:
    global FILE_NAME, SCALER, SCALER_Y, MODEL,  TEST_DATA
    # Reading Data
    df = TEST_DATA.copy()
    y_test = df.iloc[:, 0].values
    X_test = df.iloc[:, 1:].values

    # Preprocessing
    X_test = SCALER.transform(X_test)

    # Tensor Conversion
    test_tensor_x = torch.Tensor(X_test)
    if TASK_TYPE == 'classification':
        y_test = np.array([CLASS_LABEL_MAP[label] for label in y_test])
        test_tensor_y = torch.LongTensor(y_test)
    else:
        y_test = y_test.astype(float)
        test_tensor_y = torch.Tensor(y_test)

    test_dataset = TensorDataset(test_tensor_x, test_tensor_y)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    # Set the model to evaluation mode
    MODEL.eval()

    # 2. Evaluate the Test Dataset:

    predictions = []

    for inputs, _ in test_loader:
        with torch.no_grad():
            outputs = MODEL(inputs)
            if TASK_TYPE == 'classification':
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
            else:
                predictions.extend(outputs.cpu().squeeze().numpy()) #0, 1, 2, class label

    # If regression, inverse scale the predictions
    if TASK_TYPE == 'regression':
        predictions = np.array(predictions).reshape(-1, 1)  # Convert to 2D
        predictions = SCALER_Y.inverse_transform(predictions)  # inverse transform
        predictions = predictions.flatten()  # Convert back to 1D
    # 3. Accuracy Calculation:

    if TASK_TYPE == 'classification':
        accuracy = accuracy_score(y_test, predictions)
        y_test = np.array(y_test)
        predictions = np.array(predictions) #0, 1, 2, class label
        correct_predictions = np.sum(y_test == predictions)
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Total Predictions: {len(predictions)}")
        print(f"Accuracy: {accuracy:.4f}")
        print("The accuracy tells us how many times our model correctly predicted")
        print("the class label out of all the predictions it made")
        print("It is the value when dividing the correct predictions by the total predictions.")
        print(f"In this example, it is {correct_predictions}/{len(predictions)}.\n\n")

    else:
        mae = mean_absolute_error(y_test, predictions)
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print("The Mean Absolute Error (MAE) tells us, on average, how far off")
        print("one prediction are from the actual value.")
        print(f"In this example, it means, on avereage, every predition is {mae:.4f} far from the actual value.\n\n" )

        # Optional: Display some of the best and worst predictions
        errors = np.abs(np.array(predictions) - y_test)
        # Handle divide by zero

        mask_zero = (y_test == 0)
        relative_errors = np.where(mask_zero, errors, errors / y_test)

        # Get sorted indices
        sorted_indices = np.argsort(relative_errors)
        print("The error is defined as the difference between the true value and the prediction.")
        print("The relative error is defined as the error divided by the true value when the true value is not 0, and the error itself otherwise.")
        print("The best/worst below is determined by the relative errors.")
        print("5 Best Predictions:")
        for idx in sorted_indices[:5]:
            print(f"True: {y_test[idx]}, Predicted: {predictions[idx]:.4f}")
        print("\n5 Worst Predictions:")
        for idx in sorted_indices[-5:]:
            print(f"True: {y_test[idx]}, Predicted: {predictions[idx]:.4f}")

    # 4. Visualization:
    time.sleep(WAIT_TIME)
    if TASK_TYPE == 'classification':
      # Convert y_test and predictions to string
        y_test = y_test.astype(str) #string
        predictions = predictions.astype(str) #string
        # Calculate the confusion matrix
        cm = confusion_matrix(y_test, predictions)

        # Define the labels for the axes
        labels = list(CLASS_LABEL_MAP.keys())
        x_ticks_labels = ["Predicted " + str(s) for s in labels]
        y_ticks_labels = ["True " + str(s) for s in labels]

        # Use Seaborn to create the heatmap
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt='g',
                    xticklabels=x_ticks_labels,
                    yticklabels=y_ticks_labels)

        # Set the title
        plt.title("Details")
        plt.show()

    elif TASK_TYPE == 'regression':
        plt.figure(figsize=(4, 3))  # Adjust the size of the plot if needed

        # Scatter plot
        plt.scatter(y_test, predictions, alpha=0.5, color='blue', label='Predicted vs True')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal Prediction Line')  # Diagonal line

        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend(loc='upper left')  # Display the legend
        plt.show()

    # Data Reconstruction
    if TASK_TYPE == "classification":
        INVERSE_CLASS_LABEL_MAP = {v: k for k, v in CLASS_LABEL_MAP.items()}
        predictions = [INVERSE_CLASS_LABEL_MAP[int(pred)] for pred in predictions]
        # class names, for example "Yes", "No"

    # CSV Creation
    df.insert(1, 'Predictions', predictions)
    df.to_csv('with_prediction.csv', index=False)

    # Downloading the CSV
    if DownLoad: files.download('with_prediction.csv')
