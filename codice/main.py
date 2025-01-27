# Main
from training import train_and_evaluate_models
from preprocessing import load_and_preprocess_data

file_path = '/Users/carla/PycharmProjects/pythonProject/London Property Listings Dataset.csv'

# Preprocessing
X_train_prepared, X_val_prepared, y_train, y_val, preprocessor, X_val_raw = load_and_preprocess_data(file_path)

# Training e valutazione
best_model, results = train_and_evaluate_models(X_train_prepared, X_val_prepared, y_train, y_val, X_val_raw)

# Mostra i risultati finali
print("\nFinal Results:")
for model_name, metrics in results.items():
    print(f"{model_name}: {metrics}")