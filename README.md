Questo repository contiene un progetto per la predizione dei prezzi immobiliari utilizzando modelli di machine learning. Include il caricamento, la pulizia, la trasformazione dei dati e l’addestramento di diversi algoritmi di regressione per ottimizzare le prestazioni.

Struttura del progetto
	•	load_and_preprocess_data: Funzione per il caricamento e la preparazione dei dati, inclusa la pulizia, trasformazioni logaritmiche e codifica delle variabili categoriche.
	•	train_and_evaluate_models: Funzione per addestrare e valutare diversi modelli di regressione (Random Forest, Decision Tree, Linear Regression). Include il tuning iperparametri tramite RandomizedSearchCV e l’analisi del miglior modello.
	•	analyze_false_negatives: Funzione per identificare e analizzare i casi di false negatives (previsioni significativamente inferiori al valore reale).
	•	Visualizzazioni: Analisi della distribuzione degli errori residui e scatter plot residui-feature per il miglior modello.

Caratteristiche principali
  •	Preprocessing:
  	•	Trasformazioni logaritmiche per ridurre la varianza delle feature.
  	•	Gestione delle variabili categoriche con raggruppamenti personalizzati.
  	•	Standardizzazione delle feature numeriche e codifica one-hot delle categoriche.
 
	•	Addestramento e tuning dei modelli:
  	•	Supporto per tre modelli principali: Random Forest, Decision Tree e Linear Regression.
  	•	Tuning automatico degli iperparametri tramite RandomizedSearchCV.
  	•	Valutazione delle prestazioni su validation set con MAE e R².
   
  •	Analisi degli errori:
  	•	Identificazione dei false negatives per comprendere i casi problematici.
  	•	Visualizzazione della distribuzione degli errori residui.
  	•	Analisi dei residui rispetto alle feature per individuare eventuali trend non catturati dal modello.

Requisiti
	•	Python 3.x
	•	Librerie principali: pandas, numpy, scikit-learn, seaborn, matplotlib

Come utilizzare il repository
	1.	Carica i dati nel formato CSV contenente le seguenti colonne: Price, Size, Bedrooms, Bathrooms, Property Type, Area, Postcode, Area_Avg_Price.
	2.	Esegui il notebook o lo script principale per:
	•	Pulire e preprocessare i dati.
	•	Addestrare i modelli e selezionare il migliore.
	•	Analizzare gli errori e le prestazioni del miglior modello.
	3.	Consulta le visualizzazioni per interpretare i risultati e ottimizzare ulteriormente.

Risultati
Il progetto identifica il miglior modello (Random Forest) in base alla metrica MAE e fornisce un’analisi completa delle sue prestazioni, incluso il comportamento sugli errori più significativi.
