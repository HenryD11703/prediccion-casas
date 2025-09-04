import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="API de Predicción de Precios de Casas",
    description="Una API simple para predecir el precio de venta de una casa usando un modelo de Random Forest."
)

try:
    rf_model = joblib.load('models/random_forest_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    # La lista de columnas del DataFrame de entrenamiento
    feature_columns = joblib.load('models/feature_columns.pkl')
    print("Modelos y artefactos cargados exitosamente.")
except FileNotFoundError:
    print("Error: Los archivos del modelo no se encontraron. Asegúrate de que los archivos .pkl estén en la carpeta 'models/'.")
    rf_model = None
    scaler = None
    feature_columns = None


class HouseFeatures(BaseModel):
    Id: int
    MSSubClass: int
    MSZoning: Optional[str] = "RL"
    LotFrontage: Optional[float] = None
    LotArea: int
    Street: Optional[str] = "Pave"
    Alley: Optional[str] = None
    LotShape: Optional[str] = "Reg"
    LandContour: Optional[str] = "Lvl"
    Utilities: Optional[str] = "AllPub"
    LotConfig: Optional[str] = "Inside"
    LandSlope: Optional[str] = "Gtl"
    Neighborhood: Optional[str] = "CollgCr"
    Condition1: Optional[str] = "Norm"
    Condition2: Optional[str] = "Norm"
    BldgType: Optional[str] = "1Fam"
    HouseStyle: Optional[str] = "1Story"
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    RoofStyle: Optional[str] = "Gable"
    RoofMatl: Optional[str] = "CompShg"
    Exterior1st: Optional[str] = "VinylSd"
    Exterior2nd: Optional[str] = "VinylSd"
    MasVnrType: Optional[str] = "None"
    MasVnrArea: Optional[float] = None
    ExterQual: Optional[str] = "TA"
    ExterCond: Optional[str] = "TA"
    Foundation: Optional[str] = "PConc"
    BsmtQual: Optional[str] = "TA"
    BsmtCond: Optional[str] = "TA"
    BsmtExposure: Optional[str] = "No"
    BsmtFinType1: Optional[str] = "Unf"
    BsmtFinSF1: int
    BsmtFinType2: Optional[str] = "Unf"
    BsmtFinSF2: int
    BsmtUnfSF: int
    TotalBsmtSF: int
    Heating: Optional[str] = "GasA"
    HeatingQC: Optional[str] = "Ex"
    CentralAir: Optional[str] = "Y"
    Electrical: Optional[str] = "SBrkr"
    FirstFlrSF: int
    SecondFlrSF: int
    LowQualFinSF: int
    GrLivArea: int
    BsmtFullBath: int
    BsmtHalfBath: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: Optional[str] = "TA"
    TotRmsAbvGrd: int
    Functional: Optional[str] = "Typ"
    Fireplaces: int
    FireplaceQu: Optional[str] = None
    GarageType: Optional[str] = "Attchd"
    GarageYrBlt: Optional[float] = None
    GarageFinish: Optional[str] = "Unf"
    GarageCars: int
    GarageArea: int
    GarageQual: Optional[str] = "TA"
    GarageCond: Optional[str] = "TA"
    PavedDrive: Optional[str] = "Y"
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    ThreeSsnPorch: int
    ScreenPorch: int
    PoolArea: int
    PoolQC: Optional[str] = None
    Fence: Optional[str] = None
    MiscFeature: Optional[str] = None
    MiscVal: int
    MoSold: int
    YrSold: int
    SaleType: Optional[str] = "WD"
    SaleCondition: Optional[str] = "Normal"


@app.post("/predict")
def predict(house: HouseFeatures):
    """
    Predice el precio de venta de una casa basándose en sus características.
    """
    if rf_model is None or scaler is None or feature_columns is None:
        return {"error": "El modelo no se ha cargado. Por favor, revisa los archivos .pkl."}

    df = pd.DataFrame([house.dict()])
    
    df['Alley'] = df['Alley'].fillna('NA')
    df['FireplaceQu'] = df['FireplaceQu'].fillna('NA')
    df['PoolQC'] = df['PoolQC'].fillna('NA')
    df['Fence'] = df['Fence'].fillna('NA')
    df['MiscFeature'] = df['MiscFeature'].fillna('NA')
    
    df['LotFrontage'] = df['LotFrontage'].fillna(69.0)
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df['GarageYrBlt'] = df.apply(
        lambda row: row['YearBuilt'] if pd.isna(row['GarageYrBlt']) else row['GarageYrBlt'], axis=1
    )
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Aplicar One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=list(categorical_cols), drop_first=False)
    
    df_final = df_encoded.reindex(columns=feature_columns, fill_value=0)
    
    # Hacer la predicción con el modelo
    prediction = rf_model.predict(df_final)

    return {"predicted_price": prediction[0]}