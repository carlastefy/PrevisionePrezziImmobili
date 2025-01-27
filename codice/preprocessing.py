import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_and_preprocess_data(file_path):
    # Caricamento dei dati da un file CSV indicato dal percorso specificato
    data = pd.read_csv(file_path)

    # Pulizia dei dati: selezione delle righe con valori validi per 'Size' e 'Bathrooms'
    data_cleaned = data[(data['Size'] >= 10) & (data['Size'] < 10000) & (data['Bathrooms'] < 10)].copy()

    # Applicazione di trasformazioni logaritmiche a colonne specifiche
    data_cleaned.loc[:, 'Size'] = np.log1p(data_cleaned['Size'])  # Trasforma 'Size' usando log(1 + valore)
    data_cleaned.loc[:, 'Bathrooms'] = np.log1p(data_cleaned['Bathrooms'])  # Trasforma 'Bathrooms' allo stesso modo
    data_cleaned.loc[:, 'Area_Avg_Price'] = np.log1p(data_cleaned['Area_Avg_Price'])  # Trasforma 'Area_Avg_Price'

    # Raggruppamento di valori categorici meno frequenti nella categoria 'Others'
    area_counts = data_cleaned['Area'].value_counts()  # Conta le occorrenze di ciascuna 'Area'
    postcode_counts = data_cleaned['Postcode'].value_counts()  # Conta le occorrenze di ciascun 'Postcode'
    data_cleaned.loc[:, 'Area'] = data_cleaned['Area'].apply(lambda x: x if area_counts[x] > 10 else 'Others')
    data_cleaned.loc[:, 'Postcode'] = data_cleaned['Postcode'].apply(lambda x: x if postcode_counts[x] > 10 else 'Others')

    # Separazione delle feature (variabili indipendenti) dal target (variabile dipendente)
    X = data_cleaned.drop(columns=['Price', 'Price_Category'])  # Rimuove 'Price' e 'Price_Category' dalle feature
    y = data_cleaned['Price']  # La variabile target è 'Price'

    # Divisione dei dati in train (70%), validation (15%) e test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # Prima divisione
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Divisione dei dati temporanei

    # Definizione delle colonne numeriche e categoriche per il preprocessing
    numerical_columns = ['Bedrooms', 'Bathrooms', 'Size', 'Area_Avg_Price']  # Colonne numeriche da scalare
    categorical_columns = ['Property Type', 'Postcode', 'Area']  # Colonne categoriche da codificare

    # Creazione del preprocessor per trasformazioni numeriche e categoriche
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),  # Standardizzazione delle colonne numeriche
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # Codifica one-hot per le categoriche
        ]
    )

    # Applicazione del preprocessor ai dati di addestramento
    X_train_prepared = preprocessor.fit_transform(X_train)  # Adatta e trasforma i dati di train
    X_val_prepared = preprocessor.transform(X_val)  # Trasforma i dati di validation senza riadattare

    return X_train_prepared, X_val_prepared, y_train, y_val, preprocessor, X_val
