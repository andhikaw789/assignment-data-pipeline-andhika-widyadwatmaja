import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data/raw"
OUTPUT_DIR = BASE_DIR / "data/processed"


def read_dataset(nama_file):
    file_path = DATA_DIR / f"{nama_file}.csv"
    df = pd.read_csv(file_path)
    return df


def data_shape(df):
    df_shape = df.shape
    return df_shape

def columns(df):
    df_columns = df.columns.tolist()
    return df_columns

def data_type(df):
    df_type = df.dtypes
    return df_type

def check_null(df):
    df_null = df.isna().sum()
    return df_null

def check_duplicate(df):
    df_dup = df.duplicated().sum()
    return df_dup

def check_unique(df, col_name):
    df_unique = df[col_name].unique().tolist()
    return df_unique


def cleansing(df):
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
                df[col] = df[col].astype("string").str.lower().str.strip()
    
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    

    df = df.drop_duplicates()

    df['num-of-doors'] = np.where(df['num-of-doors'] == 'two', 2, 4)

    df['num-of-cylinders'] = np.where(df['num-of-cylinders'] == 'two', 2,
                                np.where(df['num-of-cylinders'] == 'three', 3,
                                np.where(df['num-of-cylinders'] == 'four', 4,
                                np.where(df['num-of-cylinders'] == 'five', 5,
                                np.where(df['num-of-cylinders'] == 'six', 6,
                                np.where(df['num-of-cylinders'] == 'eight', 8, 12))))))

    return df


def transform(df):
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse_output=False)

    encoded = encoder.fit_transform(df[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index)

    df = pd.concat(
        [df.drop(columns=categorical_cols), encoded_df],
        axis=1)

    return df


df = read_dataset("automobileEDA_dirty_training")
data_shape(df)
columns(df)
data_type(df)
check_null(df)
check_duplicate(df)
check_unique(df, "make")
check_unique(df, "aspiration")
check_unique(df, "body-style")
check_unique(df, "drive-wheels")
check_unique(df, "engine-location")
check_unique(df, "engine-type")
check_unique(df, "fuel-system")
check_unique(df, "horsepower-binned")
    
df_cleanse = cleansing(df)
df_transform = transform(df_cleanse)

data_shape(df_transform)
columns(df_transform)
data_type(df_transform)
check_null(df_transform)
check_duplicate(df_transform)

output_file = OUTPUT_DIR / "automobileEDA_processed.csv"
df_transform.to_csv(output_file, index=False)