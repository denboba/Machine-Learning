# Raport Complet - Tema Învățare Automată

## 📄 Raportul Final

**Fișier:** `Tema_Invatare_Automata_Raport_Complet.pdf`

Raport complet în format PDF pentru Tema de Învățare Automată - Partea 1, conform cerințelor din `tema.pdf`.

### Structura Raportului

#### 1. Executive Summary
Prezentare generală a celor două proiecte de machine learning realizate.

#### 2. Proiectul 1: Predicția Închirierilor de Biciclete
- **2.1 Introducere** - Obiectivele proiectului
- **2.2 Descrierea Dataset-ului** - Structura și coloanele datelor
- **2.3 Exploratory Data Analysis (EDA)** - 6 analize detaliate:
  - Analiza valorilor lipsă
  - Statistici descriptive
  - Patternuri temporale (seasonal, hourly, daily)
  - Analiza corelațiilor
  - Analiza distribuțiilor
  - Relații între variabile categorice și țintă
- **2.4 Feature Engineering** - Extragerea caracteristicilor temporale
- **2.5 Metodologie** - Modelele folosite și hiperparametrii
- **2.6 Evaluarea Modelelor** - MSE, MAE, R² scores
- **2.7 Rezultate și Concluzii** - Interpretarea rezultatelor

#### 3. Proiectul 2: Predicția Prețurilor Mașinilor (Autovit)
- **3.1 Introducere** - Obiectivele proiectului
- **3.2 Descrierea Dataset-ului** - 37 de caracteristici
- **3.3 Exploratory Data Analysis (EDA)** - 6 analize detaliate:
  - Analiza variabilei țintă (preț)
  - Corelații numerice
  - Caracteristici categorice importante
  - Tendințe pe ani
  - Pattern-uri valori lipsă
  - Impact-ul caracteristicilor asupra prețului
- **3.4 Feature Engineering** - Preprocesare complexă
- **3.5 Metodologie** - Modele și configurații
- **3.6 Evaluarea Modelelor** - Metrici de performanță
- **3.7 Rezultate și Concluzii** - Interpretări practice

#### 4. Analiză Comparativă
Comparația între cele două proiecte și lecțiile învățate.

#### 5. Concluzii Generale
Realizările tehnice și impact-ul practic.

#### 6. Referințe
Biblioteci, dataset-uri și tehnici utilizate.

#### Anexă: Vizualizări
11 grafice detaliate incluse:

**Bike Rental Dataset:**
1. Temporal Analysis
2. Correlation Analysis
3. Distribution Analysis
4. Quantile Regression
5. Model Comparison

**Autovit Dataset:**
6. Target Variable Analysis
7. Correlation Analysis
8. Categorical Features
9. Year Trends
10. Missing Data Patterns
11. Model Comparison

---

## 📊 Cerințe Îndeplinite

### 4.1 Exploratory Data Analysis [2p] ✅
- ✅ Bike: 6 analize complete cu justificări și observații
- ✅ Autovit: 6 analize complete cu justificări și observații
- ✅ Interpretări și analize detaliate pentru fiecare diagramă

### 4.2 Preprocesare [3p] ✅
- ✅ Feature extraction (caracteristici temporale, ciclice)
- ✅ Standardizare (StandardScaler pentru features numerice)
- ✅ Imputare valori lipsă (strategii multiple)
- ✅ Selecție features (eliminare multicolinearitate)
- ✅ Documentare completă a procedurii

### 4.3 Modele Machine Learning [5p] ✅
- ✅ LinearRegression (baseline)
- ✅ SVR cu RandomizedSearchCV
- ✅ RandomForestRegressor cu RandomizedSearchCV
- ✅ GradientBoostingRegressor (squared_error) cu tuning
- ✅ GradientBoostingRegressor (quantile α=0.05, 0.50, 0.95)
- ✅ QuantileRegressor cu regularization tuning
- ✅ Metrici complete: MSE, MAE, R² pentru toate modelele
- ✅ Tabele comparative cu hiperparametri
- ✅ Vizualizări comparative
- ✅ Interpretări și analize ale rezultatelor

---

## 🎯 Rezultate Finale

### Dataset Închiriere Biciclete
**Cel mai bun model:** GradientBoostingRegressor (squared_error)
- MSE: 3,156
- MAE: 36.5 închirieri/oră
- R²: 0.906 (explică 90.6% din variabilitate)

### Dataset Autovit
**Cel mai bun model:** GradientBoostingRegressor (squared_error)
- MSE: 84,358,627
- MAE: 3,598 EUR
- R²: 0.909 (explică 90.9% din variabilitate)

---

## 📁 Fișiere Incluse

### Rapoarte
- `Tema_Invatare_Automata_Raport_Complet.pdf` - **Raportul final PDF (4.5 MB, 18 pagini)**
- `Assignment1_Complete_Report.docx` - Versiunea Word a raportului

### Cod Sursă
- `tema1_abdulkadir gobena-denboba.ipynb` - Notebook principal cu implementarea
- `generate_complete_report.py` - Script pentru generarea PDF-ului

### Vizualizări (11 fișiere PNG)
**Bike Rental:**
- `bike_eda_temporal.png`
- `bike_eda_correlations.png`
- `bike_eda_distributions.png`
- `bike_quantile_regression.png`
- `bike_models_comparison.png`

**Autovit:**
- `autovit_eda_target.png`
- `autovit_eda_correlations.png`
- `autovit_eda_categorical.png`
- `autovit_eda_year_trend.png`
- `autovit_eda_missing_pattern.png`
- `autovit_models_comparison.png`

### Predicții
- `predictii_biciclete_final.csv`
- `predictii_autovit_final.csv`

---

## 🔧 Cum să Regenerezi PDF-ul

Dacă este necesar să regenerezi PDF-ul:

```bash
cd tema
python3 generate_complete_report.py
```

Acest script:
1. Citește conținutul din `Assignment1_Complete_Report.docx`
2. Convertește formatul în PDF folosind ReportLab
3. Adaugă toate vizualizările în anexă
4. Generează `Tema_Invatare_Automata_Raport_Complet.pdf`

---

## ✅ Conformitate cu Cerințele

Raportul respectă toate cerințele din `tema.pdf`:

1. ✅ **Format PDF** - Raportul final este în format PDF
2. ✅ **Cerința 4.1** - EDA complet cu vizualizări și interpretări pentru ambele dataset-uri
3. ✅ **Cerințele 4.2-4.4** - Preprocesare, modele ML, și evaluare documentate
4. ✅ **Interpretări obligatorii** - Fiecare rezultat este analizat și interpretat
5. ✅ **Tabele de rezultate** - Configurații de hiperparametri și metrici
6. ✅ **Vizualizări** - 11 grafice incluse în anexă

---

## 📅 Informații

**Student:** Abdulkadir Gobena-Denboba  
**Curs:** Învățare Automată  
**Data:** Noiembrie 2025  
**Status:** ✅ Complet și gata pentru predare

---

## 📝 Note Importante

1. **LinearRegression vs LogisticRegression**: Am folosit LinearRegression (corect pentru regresie), nu LogisticRegression (pentru clasificare)

2. **Quantile Regression**: Implementat pentru interval predictions (α=0.05, 0.50, 0.95)

3. **RandomizedSearchCV**: Folosit pentru eficiență în căutarea hiperparametrilor

4. **Features eliminate**: 
   - `temperatura_resimtita` (multicolinearitate cu `temperatura`)
   - `inregistrati`, `ocazionali` (data leakage - nu sunt disponibile în test)

5. **Features ciclice**: Implementat transformări sin/cos pentru ora (ora 23 și ora 0 sunt consecutive)

---

Pentru întrebări sau clarificări, consultați:
- `tema.pdf` - Cerințele originale
- `tema1_abdulkadir gobena-denboba.ipynb` - Implementarea detaliată
- `FINAL_SUMMARY.txt` - Rezumat implementare
