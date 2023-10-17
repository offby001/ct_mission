# @title Data Pre-Processing check the buttom on the left of the next row(?) earlier version
# %%writefile myscript.py
# Importing Necessary Libraries
import time
# For data manipulation
import pandas as pd
import numpy as np

# For file operations in Google Colab
from google.colab import files

# For preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split


# Global Variables

# Placeholder for the dataset. Will be assigned when the dataset is uploaded.
FULL_DATASET = None
PROCESSED_FULL_DATASET = None

# Constants for dataset format
ACTUAL_VALUE_COL = 0
TYPE_ROW = 0
CLASS_LABEL_MAP = {} # Mapping of class name to integer label
TASK_TYPE = 0
CUTOFF_FOR_LABEL_ENCODING = 5

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
    processed_data[FULL_DATASET.columns[ACTUAL_VALUE_COL]] = data_without_type.iloc[:, ACTUAL_VALUE_COL]

    # Iterate over columns to process them based on type
    for idx, col_name in enumerate(data_without_type.columns[1:], start=1):  # starting from 1 to skip the actual value column

        # Convert string numbers to actual numbers
        data_without_type[col_name] = pd.to_numeric(data_without_type[col_name], errors='ignore')
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


def MakeClassLabels():
    global FULL_DATASET, CLASS_LABEL_MAP

    if FULL_DATASET.iloc[0, ACTUAL_VALUE_COL] == "classification":

        le = LabelEncoder()
        data_without_type = FULL_DATASET.drop(TYPE_ROW, axis=0).copy()

        le.fit(data_without_type.iloc[:, ACTUAL_VALUE_COL])
        CLASS_LABEL_MAP = dict(zip(le.classes_, le.transform(le.classes_)))
        return f"For this classification Task, {len(CLASS_LABEL_MAP)} Class Labels Mapped Successfully!\n\n"

def GenereatingNewfiles():
    # Splitting the data
    # PROCESSED_FULL_DATASET.to_csv("processed_data.csv",index = False)
    train_data, test_data = train_test_split(PROCESSED_FULL_DATASET, test_size=0.2, random_state=42)

    # Saving to CSV
    train_file_path = 'training_data.csv'
    test_file_path = 'testing_data.csv'

    train_data.to_csv(train_file_path, index=False)
    test_data.to_csv(test_file_path, index=False)
    return "Files Generated Successfully!\n\n"

def FilesDownloading():

    
    try:
        # Trigger automatic download
        files.download('training_data.csv'h)
        files.download('testing_data.csv')
    except Exception as e:
        print("Error appeared at the FilesDownloading step. Please check your input or seek help from teachers.")
        print(f"Technical details: {e}")
        return
    

def DataPreProcessing(file_name):
        # Step 1: FullDataSetUpload
    try:
        '''
        print(FullDataSetUpload())
        '''
        global FULL_DATASET
        FULL_DATASET = None
        FULL_DATASET = pd.read_csv(file_name)
        print("Dataset Uploaded Successfully!\n\n")
        
        
    except Exception as e:
        print("Error appeared at the FullDataSetUpload step. Please check your input or seek help from teachers.")
        print(f"Technical details: {e}")
        return
    print("A preview of the DataSet as show below...")
    pd.set_option('display.max_rows', 5)
    pd.set_option('display.max_columns', 5)
    print(FULL_DATASET)
    print("Now let's validate the dataset...")
    #time.sleep(3)
    # Step 2: FullDataSetValidation
    try:
        print(FullDataSetValidation())
    except Exception as e:
        print("Error appeared at the FullDataSetValidation step. Please check your input or seek help from teachers.")
        print(f"Technical details: {e}")
        return


    print("Now let's process the features ...")
    #time.sleep(3)
    # Step 3: ProcessFeatures
    try:
        print(ProcessFeatures())
        print(PROCESSED_FULL_DATASET)
    except Exception as e:
        print("Error appeared at the ProcessFeatures step. Please check your input CSV file or seek help from teachers.")
        print(f"Technical details: {e}")
        return
    if TASK_TYPE == "classification":
        print("Now let's us map the class names to class labels...")
        #time.sleep(3)

    # Step 4: MakeClassLabels
    if FULL_DATASET.iloc[0, ACTUAL_VALUE_COL] == "classification":
        try:
            print(MakeClassLabels())
            print(CLASS_LABEL_MAP)
        except Exception as e:
            print("Error appeared at the MakeClassLabels step. Please check your input or seek help from teachers.")
            print(f"Technical details: {e}")
            return

    print("Now let's generate files for trainign and testing...")
    #time.sleep(3)
    # Step 5: GenereatingNewfiles
    try:
        print(GenereatingNewfiles())
    except Exception as e:
        print("Error appeared at the GenereatingNewfiles step. Please check your input or seek help from teachers.")
        print(f"Technical details: {e}")
        return

    print("Data Preprocessing completed successfully!")
#DataPreProcessing(file_name)
#filename = input("Please enter the filename on the, without quotaion marks:")
#print("testing", filename)
#DataPreProcessing(file_name)

