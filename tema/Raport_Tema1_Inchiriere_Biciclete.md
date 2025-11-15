# Raport Tehnică - Tema 1: Predicția Închirierii Bicicletelor

**Student:** Abdulkadir Gobena-Denboba  
**Data:** Noiembrie 2024  
**Dataset:** Bike Sharing Dataset (Actualizat)

---

## 1. Introducere

### 1.1 Obiectivul Proiectului
Scopul acestui proiect este de a dezvolta un model de predicție pentru numărul de biciclete închiriate pe baza condițiilor meteorologice, sezonalității și altor factori contextuali. Proiectul utilizează tehnici de învățare automată supervizată pentru a prezice intervalele de încredere ale numărului total de închirieri.

### 1.2 Actualizare Dataset
Conform anunțului oficial din cadrul cursului, setul de date a fost actualizat pentru a include etichetele (labels) în subsetul de test:

- **Dataset anterior:** `train.csv` și `test.csv` (test.csv fără etichete)
- **Dataset actualizat:** `train_split.csv` și `eval_split.csv` (ambele cu etichete)

Această actualizare permite evaluarea corectă a performanței modelului pe datele de test.

---

## 2. Analiza Exploratorie a Datelor (EDA)

### 2.1 Descrierea Datasetului

#### Train Dataset (`train_split.csv`)
- **Număr de înregistrări:** 6,878 observații
- **Perioada:** 1 ianuarie 2011 - 12 decembrie 2012
- **Caracteristici:** 12 variabile

#### Test Dataset (`eval_split.csv`)
- **Număr de înregistrări:** 4,008 observații
- **Perioada:** 13 ianuarie 2011 - 19 decembrie 2012
- **Caracteristici:** 12 variabile (inclusiv `total` - variabila țintă)

### 2.2 Variabilele Datasetului

| Variabilă | Tip | Descriere |
|-----------|-----|-----------|
| `data_ora` | datetime | Data și ora înregistrării |
| `sezon` | int | Sezonul (1=Primăvară, 2=Vară, 3=Toamnă, 4=Iarnă) |
| `sarbatoare` | int | Indicator de sărbătoare (0=Nu, 1=Da) |
| `zi_lucratoare` | int | Indicator zi lucrătoare (0=Weekend/Sărbătoare, 1=Zi lucrătoare) |
| `vreme` | int | Condiții meteo (1=Senin, 2=Înnnorat, 3=Ploaie ușoară, 4=Ploaie abundentă) |
| `temperatura` | float | Temperatura normalizată în Celsius |
| `temperatura_resimtita` | float | Temperatura resimțită normalizată |
| `umiditate` | int | Umiditatea relativă (%) |
| `viteza_vant` | float | Viteza vântului |
| `ocazionali` | int | Număr de utilizatori ocazionali |
| `inregistrati` | int | Număr de utilizatori înregistrați |
| `total` | int | **Variabila țintă** - Total închirieri (ocazionali + înregistrați) |

### 2.3 Statistici Descriptive

#### Train Dataset
- **Total închirieri:**
  - Minim: 1 închiriere
  - Maxim: 977 închirieri
  - Medie: 188.87 închirieri/oră
  - Deviație standard: ~181 închirieri

#### Test Dataset
- **Total închirieri:**
  - Minim: 1 închiriere
  - Maxim: 943 închirieri
  - Medie: 196.22 închirieri/oră
  - Deviație standard: ~185 închirieri

### 2.4 Valori Lipsă
**Rezultat:** Nu au fost identificate valori lipsă în niciuna dintre cele două seturi de date. Toate cele 12 variabile au 100% completitudine.

### 2.5 Observații din EDA

1. **Distribuția temporală:** Există un overlap temporal între seturile de train și test, ceea ce permite evaluarea modelului pe date contemporane.

2. **Sezonalitate:** Se observă variații clare ale numărului de închirieri în funcție de:
   - Ora din zi (vârfuri la orele de vârf - dimineața și seara)
   - Sezon (vară > primăvară/toamnă > iarnă)
   - Zi lucrătoare vs. weekend

3. **Factori meteo:** Condițiile meteorologice au impact semnificativ:
   - Vremea senină corelată cu mai multe închirieri
   - Temperature moderate favorizează utilizarea bicicletelor
   - Viteza vântului și umiditatea influențează negativ

---

## 3. Ingineria Caracteristicilor (Feature Engineering)

### 3.1 Caracteristici Derivate

Pe lângă cele 12 caracteristici originale, au fost create 4 caracteristici temporale suplimentare:

```python
def extract_features(df):
    df = df.copy()
    df['ora'] = df['data_ora'].dt.hour          # Ora din zi (0-23)
    df['zi_saptamana'] = df['data_ora'].dt.dayofweek  # Ziua săptămânii (0-6)
    df['luna'] = df['data_ora'].dt.month        # Luna (1-12)
    df['este_weekend'] = (df['zi_saptamana'] >= 5).astype(int)  # Weekend (0/1)
    return df
```

### 3.2 Set Final de Caracteristici

**Total caracteristici utilizate:** 12 (excludem `data_ora`, `ocazionali`, `inregistrati`)

Caracteristici pentru modelare:
- `sezon`, `sarbatoare`, `zi_lucratoare`, `vreme`
- `temperatura`, `temperatura_resimtita`, `umiditate`, `viteza_vant`
- `ora`, `zi_saptamana`, `luna`, `este_weekend`

---

## 4. Metodologia și Modelarea

### 4.1 Alegerea Modelului

S-a ales **Gradient Boosting Regressor** pentru următoarele motive:

1. **Performanță superioară** pentru probleme de regresie cu date tabulare
2. **Capacitate de a captura relații non-liniare** complexe
3. **Rezistență la overfitting** prin regularizare
4. **Suport pentru regresie cantilă** - permite predicția de intervale de încredere

### 4.2 Predicție prin Regresie Cantilă (Quantile Regression)

Pentru a oferi estimări mai robuste, s-au antrenat 3 modele separate:

1. **Model cantilă inferioară (α=0.05):** Limita inferioară a predicției (percentila 5%)
2. **Model median (α=0.50):** Predicția mediană
3. **Model cantilă superioară (α=0.95):** Limita superioară a predicției (percentila 95%)

### 4.3 Hiperparametri

```python
params = {
    'n_estimators': 100,      # Număr de arbori
    'learning_rate': 0.1,     # Rata de învățare
    'max_depth': 3,           # Adâncimea maximă a arborilor
    'random_state': 42        # Pentru reproducibilitate
}
```

### 4.4 Antrenarea Modelului

```python
# Împărțire train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Antrenare modele cantilă
gbr_low = GradientBoostingRegressor(loss="quantile", alpha=0.05, **params)
gbr_med = GradientBoostingRegressor(loss="quantile", alpha=0.50, **params)
gbr_high = GradientBoostingRegressor(loss="quantile", alpha=0.95, **params)

gbr_low.fit(X_train, y_train)
gbr_med.fit(X_train, y_train)
gbr_high.fit(X_train, y_train)
```

---

## 5. Evaluarea Modelului

### 5.1 Metrici de Performanță

**RMSE (Root Mean Squared Error)** pe setul de test:

- **RMSE Test:** ~97 închirieri

Această valoare reprezintă eroarea medie pătrată pe setul de evaluare, indicând că predicțiile modelului se abat în medie cu aproximativ 97 de închirieri de la valorile reale.

### 5.2 Interpretarea Rezultatelor

#### Context:
- Media închirierilor: ~196/oră
- RMSE relativ: ~49% din medie

#### Performanță:
- **Bună** pentru date cu variabilitate mare
- Model capabil să capteze pattern-uri temporale și sezonale
- Intervale de predicție oferă încredere în estimări

### 5.3 Exemple de Predicții

Exemplu pentru prima observație din setul de test:

| Metric | Valoare |
|--------|---------|
| Predicție Low (5%) | 4.2 închirieri |
| Predicție Median (50%) | 17.8 închirieri |
| Predicție High (95%) | 57.7 închirieri |

---

## 6. Avantajele Actualizării Datasetului

### 6.1 Înainte de Actualizare
- **Limitare:** test.csv nu avea etichete
- **Impact:** Imposibilitatea evaluării modelului pe date de test
- **Consecință:** Evaluare doar pe setul de validare interne

### 6.2 După Actualizare
- **Îmbunătățire:** eval_split.csv include etichete (`total`)
- **Beneficiu:** Evaluare completă a performanței pe date de test
- **Rezultat:** Metrici de performanță reale și credibile

---

## 7. Concluzii

### 7.1 Realizări

1. ✅ **Dataset actualizat cu succes** - Migrare de la train.csv/test.csv la train_split.csv/eval_split.csv
2. ✅ **Analiză exploratorie completă** - Identificarea pattern-urilor și relațiilor
3. ✅ **Inginerie caracteristici** - Crearea de caracteristici temporale relevante
4. ✅ **Model performant** - Gradient Boosting cu RMSE ~97
5. ✅ **Predicții intervale** - Regresie cantilă pentru incertitudine
6. ✅ **Evaluare completă** - Posibilă datorită etichetelor în setul de test

### 7.2 Puncte Forte

- **Abordare robustă:** Utilizarea regresiei cantilă pentru estimări de încredere
- **Feature engineering efectiv:** Caracteristici temporale îmbunătățesc predicțiile
- **Evaluare corectă:** Dataset actualizat permite metrici reale

### 7.3 Limitări și Îmbunătățiri Viitoare

**Limitări:**
- RMSE relativ mare (49% din medie) - variabilitate naturală în date
- Overlap temporal între train și test - nu testează generalizarea strictă

**Îmbunătățiri posibile:**
1. **Feature engineering avansat:**
   - Interacțiuni între caracteristici (ex: temperatura × ora)
   - Lag features (valori anterioare)
   - Rolling statistics (medii mobile)

2. **Optimizare hiperparametri:**
   - Grid Search sau Random Search
   - Cross-validation pentru selecție optimă

3. **Modele alternative:**
   - XGBoost sau LightGBM (mai rapide)
   - Ensemble de modele diverse
   - Neural Networks pentru pattern-uri complexe

4. **Tratarea outliers:**
   - Analiza și gestionarea valorilor extreme
   - Robust scaling pentru caracteristici numerice

---

## 8. Referințe și Resurse

### 8.1 Librării Utilizate

- **pandas** (2.x) - Manipulare și analiză date
- **numpy** (1.x) - Operații numerice
- **scikit-learn** (1.x) - Modelare machine learning
- **matplotlib** (3.x) - Vizualizare date
- **seaborn** (0.x) - Vizualizare statistică

### 8.2 Fișiere Proiect

```
tema/
├── train_split.csv              # Dataset antrenare (6,878 rânduri)
├── eval_split.csv               # Dataset evaluare (4,008 rânduri)
├── tema1_abdulkadir gobena-denboba.ipynb  # Notebook principal
├── predictii_biciclete.csv      # Predicții generate
├── tema.pdf                     # Cerințe temă
└── Raport_Tema1_Inchiriere_Biciclete.md  # Acest raport
```

---

## 9. Anexe

### 9.1 Cod Esențial

#### Încărcare Date
```python
df_train = pd.read_csv('train_split.csv', parse_dates=['data_ora'])
df_test = pd.read_csv('eval_split.csv', parse_dates=['data_ora'])
```

#### Feature Engineering
```python
def extract_features(df):
    df = df.copy()
    df['ora'] = df['data_ora'].dt.hour
    df['zi_saptamana'] = df['data_ora'].dt.dayofweek
    df['luna'] = df['data_ora'].dt.month
    df['este_weekend'] = (df['zi_saptamana'] >= 5).astype(int)
    return df
```

#### Antrenare și Predicție
```python
features = ['sezon', 'sarbatoare', 'zi_lucratoare', 'vreme', 'temperatura',
            'temperatura_resimtita', 'umiditate', 'viteza_vant', 'ora',
            'zi_saptamana', 'luna', 'este_weekend']

X = df_train_ext[features]
y = df_train_ext['total']

gbr_med = GradientBoostingRegressor(
    loss="quantile", alpha=0.50, 
    n_estimators=100, learning_rate=0.1, 
    max_depth=3, random_state=42
).fit(X, y)

predictions = gbr_med.predict(df_test_ext[features])
```

### 9.2 Verificare Rezultate

Toate testele au fost executate cu succes:
- ✅ Încărcare date
- ✅ Feature extraction
- ✅ Antrenare model
- ✅ Evaluare pe test set
- ✅ Generare predicții cantilă

---

**Notă:** Acest raport documentează implementarea completă a Temei 1 pentru predicția închirierii bicicletelor, utilizând dataset-ul actualizat conform anunțului oficial al cursului.

**Data raport:** Noiembrie 2024  
**Versiune:** 1.0 (Dataset actualizat)
