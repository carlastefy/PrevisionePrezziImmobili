import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

def analyze_false_negatives(y_true, y_pred, X_raw, threshold=0.2):
    # Calcola gli errori assoluti (differenza tra valore vero e predetto)
    errors = y_true - y_pred

    # Calcola l'errore relativo rispetto al valore vero
    relative_error = errors / y_true

    # Identifica i false negatives: casi in cui l'errore relativo supera la soglia e il valore predetto è inferiore a quello vero
    false_negatives = (relative_error > threshold) & (errors > 0)

    # Restituisce le righe corrispondenti dei dati originali, i valori veri e quelli predetti
    return X_raw[false_negatives], y_true[false_negatives], y_pred[false_negatives]


def train_and_evaluate_models(X_train, X_val, y_train, y_val, X_val_raw):
    # modelli con configurazione e parametri iperottimizzabili
    models = {
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=-1),
            "params": {
                'n_estimators': [100, 200, 300, 400],  # Numero di alberi nella foresta
                'max_depth': [15, 30, None],  # Massima profondità dell'albero
                'min_samples_split': [2, 5, 10],  # Minimo numero di campioni richiesti per uno split
                'min_samples_leaf': [1, 2, 4],  # Minimo numero di campioni per foglia
                'max_features': ['sqrt', 'log2'],  # Numero massimo di feature considerate per split
                'bootstrap': [True, False]  # Utilizzo del campionamento con rimpiazzo
            }
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(random_state=42),
            "params": {
                'max_depth': [10, 20, 30, None],  # Come sopra, ma per Decision Tree
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        "Linear Regression": {
            "model": LinearRegression(),
            "params": {}
        }
    }

    # Variabili per tracciare il miglior modello
    best_model_name = None
    best_model = None
    best_score = float("inf")
    results = {}
    best_y_val_pred = None

    # Ciclo per addestrare e valutare ogni modello
    for name, config in models.items():
        print(f"\nTraining {name}...")  # Output per monitorare il progresso
        if config["params"]:
            # Ricerca iperparametri con RandomizedSearchCV
            search = RandomizedSearchCV(
                config["model"],
                config["params"],
                n_iter=10,  # Numero di combinazioni testate
                cv=5,  # Numero di fold per validazione incrociata
                scoring='neg_mean_absolute_error',  # Scoring basato sull'errore assoluto medio
                random_state=42,
                n_jobs=-1  # Utilizzo parallelo di tutti i core disponibili
            )
            search.fit(X_train, y_train)  # Adatta il modello ai dati di train
            best_model_candidate = search.best_estimator_  # Miglior modello trovato
        else:
            # Addestra il modello con i parametri di default
            config["model"].fit(X_train, y_train)
            best_model_candidate = config["model"]

        # Valutazione incrociata sui dati di train
        scores = cross_val_score(best_model_candidate, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        mean_cv_score = -np.mean(scores)  # Media dell'errore assoluto medio negativo

        # Valutazione sul validation set
        y_val_pred = best_model_candidate.predict(X_val)  # Predizione sul validation set
        mae = mean_absolute_error(y_val, y_val_pred)  # Calcolo MAE (Mean Absolute Error)
        r2 = r2_score(y_val, y_val_pred)  # Calcolo R2 (coefficiente di determinazione)

        # Salva i risultati
        results[name] = {
            "Best Params": search.best_params_ if config["params"] else "Default",  # Parametri migliori
            "CV MAE": mean_cv_score,  # Media del MAE nella validazione incrociata
            "Val MAE": mae,  # MAE sul validation set
            "Val R2": r2  # R2 sul validation set
        }

        print(f"{name} - CV MAE: {mean_cv_score:.4f}, Val MAE: {mae:.4f}, Val R2: {r2:.4f}")

        # Aggiorna il miglior modello se il MAE è il più basso finora
        if mae < best_score:
            best_score = mae
            best_model_name = name
            best_model = best_model_candidate
            best_y_val_pred = y_val_pred

    # Analisi del miglior modello
    print(f"\nAnalisi del miglior modello: {best_model_name}")
    false_negatives, y_fn_true, y_fn_pred = analyze_false_negatives(y_val, best_y_val_pred, X_val_raw)
    print(f"False negatives detected: {len(false_negatives)}")
    print(false_negatives.head())  # Stampa i primi falsi negativi rilevati

    # Analisi della distribuzione degli errori residui
    residuals = y_val - best_y_val_pred
    sns.histplot(residuals, kde=True)  # Istogramma dei residui con densità
    plt.title(f"Distribuzione degli errori residui - {best_model_name}")
    plt.show()

    # Analisi dei residui rispetto alle feature
    for feature in X_val_raw.columns:
        sns.scatterplot(x=X_val_raw[feature], y=residuals)  # Grafico scatter residui vs feature
        plt.title(f"Residui vs {feature} - {best_model_name}")
        plt.show()

    # Restituzione del miglior modello e dei risultati
    return best_model, results
