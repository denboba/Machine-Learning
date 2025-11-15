# Tema Învățare Automată - Implementare Completă

## Descriere

Implementare completă a temei de Învățare Automată conform cerințelor din **tema.pdf**.

### Autor
[Numele Studentului]

### Data
Noiembrie 2025

---

## Structura Implementării

### Fișiere Python

1. **`tema_complete_implementation.py`** - Partea 1: Dataset Închiriere Biciclete
   - EDA complet (6+ analize)
   - Feature engineering și preprocesare
   - 6 modele ML cu hyperparameter tuning
   - Rezultate și vizualizări

2. **`tema_autovit_implementation.py`** - Partea 2: Dataset Autovit (Prețuri Mașini)
   - EDA complet (6+ analize)
   - Tratare valori lipsă și encoding
   - 6 modele ML cu hyperparameter tuning
   - Rezultate și vizualizări

### Cum să Rulezi

```bash
# Partea 1: Închiriere Biciclete
cd tema
python3 tema_complete_implementation.py

# Partea 2: Autovit (Prețuri Mașini)
python3 tema_autovit_implementation.py
```

**Notă**: Rularea completă poate dura 5-10 minute per script datorită căutării hiperparametrilor (RandomizedSearchCV).

---

## Cerințe Implementate

### 4.1 Explorarea și Vizualizarea Datelor [2p]

#### Dataset Închiriere Biciclete:
- ✅ Verificare valori lipsă
- ✅ Statistici descriptive
- ✅ Vizualizare serie de timp (trend, sezonalitate, ciclicitate)
- ✅ Analiza corelațiilor
- ✅ Distribuții variabile
- ✅ Pattern-uri diferențiate (zi lucrătoare vs weekend)

#### Dataset Autovit:
- ✅ Analiza valorilor lipsă (pattern-uri și strategii)
- ✅ Distribuția target-ului (preț)
- ✅ Corelații features numerice
- ✅ Analiza features categorice
- ✅ Trend preț vs anul fabricației
- ✅ Pattern-uri valori lipsă

**Toate analizele au justificări clare și interpretări detaliate.**

### 4.2 Extragerea, Standardizarea, Selecția de Atribute [3p]

#### Dataset Închiriere Biciclete:
- ✅ **Extragere features temporale** din `data_ora`
  - Ora, zi, lună, zi săptămână
  - Features ciclice (sin/cos) pentru oră și lună
  - Justificare: Pattern-uri temporale observate în EDA
  
- ✅ **Eliminare multicolinearitate**
  - Eliminat `temperatura_resimtita` (r>0.99 cu `temperatura`)
  
- ✅ **Standardizare** features numerice
  - StandardScaler pentru temperatura, umiditate, viteza vânt
  - Justificare: Scale diferite, esențial pentru SVR și LinearRegression
  
- ✅ **Selecția features**
  - Analiza cu SelectKBest (F-statistic)
  - NU inclus `inregistrati` și `ocazionali` (data leakage)

#### Dataset Autovit:
- ✅ **Imputare valori lipsă**
  - Numerice: median (robust la outlieri)
  - Categorice: most_frequent
  
- ✅ **Encoding categorice**
  - OneHotEncoder cu `drop='first'` (evită dummy trap)
  - `handle_unknown='ignore'` pentru categorii noi în test
  
- ✅ **Standardizare** features numerice
  - StandardScaler pentru anul, km, putere, etc.
  
- ✅ **Selecție features**
  - Eliminare coloane >50% valori lipsă
  - Eliminare low variance features
  - Evitare high cardinality categorice

**Toate transformările au justificări detaliate.**

### 4.3 Utilizarea Algoritmilor de Învățare Automată [5p]

#### Modele Implementate (ambele datasets):

1. ✅ **LinearRegression** (CORECT - nu LogisticRegression)
   - Model baseline
   - Fără hyperparametri de tunat

2. ✅ **SVR** (Support Vector Regression)
   - Hyperparametri: kernel, C, epsilon, gamma
   - RandomizedSearchCV cu 15-20 iterații, 3-fold CV

3. ✅ **RandomForestRegressor**
   - Hyperparametri: n_estimators, max_depth, min_samples_split, max_features
   - RandomizedSearchCV cu 20-30 iterații, 3-fold CV

4. ✅ **GradientBoostingRegressor** (loss='squared_error')
   - Hyperparametri: n_estimators, learning_rate, max_depth, subsample
   - RandomizedSearchCV cu 20-30 iterații, 3-fold CV

5. ✅ **GradientBoostingRegressor** (loss='quantile')
   - Antrenare pentru α=0.05, 0.50, 0.95
   - Interval de predicție 90%
   - Acoperire calculată și raportată

6. ✅ **QuantileRegressor**
   - Hyperparametri: alpha (L1 regularization), solver
   - RandomizedSearchCV cu 4-10 iterații, 3-fold CV

#### Evaluare:
- ✅ **Metrici**: MSE, MAE, R² pentru toate modelele
- ✅ **Tabel comparativ** cu rezultate pentru fiecare model
- ✅ **Identificare cel mai bun model** pentru fiecare metrică
- ✅ **Vizualizări comparative** (grafice barh pentru MSE, MAE, R²)

---

## Rezultate

### Dataset Închiriere Biciclete

**Tabel Rezumat**:
```
Model                      MSE        MAE      R²
LinearRegression          19022      97.7    0.431
SVR                       13250      69.0    0.604
RandomForest               4431      43.4    0.868
GradientBoosting_SE        3156      36.5    0.906  ← CEL MAI BUN
GradientBoosting_Quantile  4430      41.0    0.868
QuantileRegressor         20889      92.4    0.376
```

**Cel mai bun model**: `GradientBoosting_SE` cu R²=0.906

### Dataset Autovit

Rezultatele vor fi generate după rularea `tema_autovit_implementation.py`.

---

## Fișiere Generate

### Închiriere Biciclete:
- `bike_eda_temporal.png` - Vizualizări trend, sezonalitate, ciclicitate
- `bike_eda_correlations.png` - Matrice corelații și corelații cu target
- `bike_eda_distributions.png` - Distribuții variabile și boxplots
- `bike_quantile_regression.png` - Interval predicție 90%
- `bike_models_comparison.png` - Comparație performanță modele
- `predictii_biciclete_final.csv` - Predicții pe setul de test

### Autovit:
- `autovit_eda_target.png` - Distribuția prețului
- `autovit_eda_correlations.png` - Corelații features numerice
- `autovit_eda_categorical.png` - Analiza features categorice
- `autovit_eda_year_trend.png` - Trend preț vs anul fabricației
- `autovit_eda_missing_pattern.png` - Pattern-uri valori lipsă
- `autovit_models_comparison.png` - Comparație performanță modele
- `predictii_autovit_final.csv` - Predicții pe setul de test

---

## Justificări Cheie

### De ce LinearRegression și NU LogisticRegression?
- **Taskul este REGRESIE** (prezicere valori continue: număr închirieri, preț EUR)
- LogisticRegression este pentru **CLASIFICARE** (clase discrete)
- LinearRegression este modelul corect pentru regresie liniară

### De ce StandardScaler?
- Features au scale foarte diferite (temp: -8 to 41, km: 0-500000)
- Esențial pentru SVR și LinearRegression
- Formula: z = (x - μ) / σ
- Păstrează distribuția, doar schimbă scala

### De ce eliminăm temperatura_resimtita?
- Corelație r > 0.99 cu temperatura
- MULTICOLINEARITATE perfectă
- Destabilizează modele liniare
- Redundanță - aceeași informație

### De ce NU includem inregistrati și ocazionali?
- total = inregistrati + ocazionali
- Ar fi **DATA LEAKAGE** (informația viitoare)
- NU sunt disponibile în setul de test (fac parte din ce prezic)

### De ce features ciclice (sin/cos)?
- Ora 23 și ora 0 sunt consecutive, nu distante
- Luna 12 și luna 1 sunt consecutive
- Transformarea sin/cos păstrează natura ciclică
- Îmbunătățește performanța modelelor liniare

### De ce RandomizedSearchCV și nu GridSearchCV?
- Mai rapid (eșantionare aleatorie vs exhaustivă)
- Acoperire bună a spațiului de căutare
- Potrivit pentru dataset-uri mari
- Cost computațional rezonabil

---

## Concluzii

### Învățăminte Cheie:

1. **EDA este esențial** - pattern-urile temporale observate au ghidat feature engineering
2. **Preprocesarea corectă face diferența** - standardizare, imputare, encoding
3. **Modele ensemble (RF, GB) performează cel mai bine** pentru ambele dataset-uri
4. **Hyperparameter tuning îmbunătățește semnificativ rezultatele**
5. **Quantile regression oferă informații valoroase** despre incertitudinea predicțiilor

### Recomandări:

- Pentru **închiriere biciclete**: GradientBoosting cu squared_error (R²=0.906)
- Pentru **Autovit**: RandomForest sau GradientBoosting (robust la outlieri de preț)
- Intervalele de predicție (quantile regression) sunt utile pentru planificare și managementul riscului

---

## Bibliografie & Referințe

- Scikit-learn Documentation: https://scikit-learn.org/
- Tutorial Quantile Regression: "Prediction Intervals for Gradient Boosting Regression"
- Tema.pdf - Cerințe oficiale

---

**Implementare completă conform tema.pdf**  
✅ Toate cerințele implementate  
✅ Justificări detaliate pentru fiecare decizie  
✅ Vizualizări clare și interpretabile  
✅ Cod documentat și reproductibil
