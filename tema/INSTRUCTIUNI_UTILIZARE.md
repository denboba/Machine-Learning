# Instrucțiuni de Utilizare - Tema 1: Închirierea Bicicletelor

## Actualizare Importantă! ⚠️

Dataset-ul a fost actualizat conform anunțului oficial. Noile fișiere includ etichete în setul de test.

### Fișiere Vechi (❌ NU MAI SUNT VALIDE)
- `train.csv` - ELIMINAT
- `test.csv` - ELIMINAT (lipseau etichetele)

### Fișiere Noi (✅ UTILIZAȚI ACESTEA)
- `train_split.csv` - Dataset de antrenare (6,878 înregistrări)
- `eval_split.csv` - Dataset de evaluare (4,008 înregistrări) **cu etichete!**

---

## Cum să Rulați Notebook-ul

### 1. Instalare Dependențe

```bash
pip install -r requirements.txt
```

Dependențe necesare:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

### 2. Structura Fișierelor

Asigurați-vă că aveți următoarea structură în directorul `tema/`:

```
tema/
├── train_split.csv                          # Dataset antrenare
├── eval_split.csv                           # Dataset evaluare
├── tema1_abdulkadir gobena-denboba.ipynb   # Notebook principal
├── Raport_Tema1_Inchiriere_Biciclete.md    # Raport tehnic
└── INSTRUCTIUNI_UTILIZARE.md               # Acest fișier
```

### 3. Rulare Notebook

1. **Deschideți notebook-ul:**
   ```bash
   jupyter notebook tema1_abdulkadir\ gobena-denboba.ipynb
   ```

2. **Rulați toate celulele:** 
   - Menu: `Cell` → `Run All`
   - Sau folosiți `Shift + Enter` pentru fiecare celulă

3. **Verificați rezultatele:**
   - EDA (Exploratory Data Analysis) - grafice și statistici
   - Antrenarea modelului - Gradient Boosting Regressor
   - Predicții - genereaza `predictii_biciclete.csv`

---

## Ce Face Notebook-ul?

### 1. Încărcare Date ✅
```python
df_train = pd.read_csv('train_split.csv', parse_dates=['data_ora'])
df_test = pd.read_csv('eval_split.csv', parse_dates=['data_ora'])
```

### 2. Analiză Exploratorie 📊
- Statistici descriptive
- Vizualizări (boxplot, lineplot)
- Analiza distribuției temporale

### 3. Feature Engineering 🔧
Creare caracteristici temporale:
- `ora` - ora din zi (0-23)
- `zi_saptamana` - ziua săptămânii (0-6)
- `luna` - luna (1-12)
- `este_weekend` - indicator weekend (0/1)

### 4. Antrenare Model 🤖
Gradient Boosting Regressor cu regresie cantilă:
- **Low (α=0.05)** - limita inferioară (percentila 5%)
- **Medium (α=0.50)** - predicție mediană
- **High (α=0.95)** - limita superioară (percentila 95%)

### 5. Evaluare 📈
- RMSE pe setul de validare
- RMSE pe setul de test (acum posibil cu noile date!)
- Generare fișier predicții

### 6. Export Rezultate 💾
Fișier generat: `predictii_biciclete.csv` cu coloanele:
- `data_ora` - timestamp
- `low` - predicție limita inferioară
- `med` - predicție mediană
- `high` - predicție limita superioară

---

## Întrebări Frecvente (FAQ)

### Q1: De ce au fost schimbate fișierele CSV?
**R:** Conform anunțului oficial, vechiul `test.csv` nu avea etichete (coloana `total`). Noul `eval_split.csv` include etichetele, permițând evaluarea corectă a modelului.

### Q2: Ce metrici folosim pentru evaluare?
**R:** RMSE (Root Mean Squared Error) - eroarea medie pătrată între predicții și valorile reale.

### Q3: De ce folosim regresie cantilă?
**R:** Pentru a oferi intervale de încredere în predicții, nu doar o valoare punctuală. Astfel știm incertitudinea predicției.

### Q4: Pot folosi vechile fișiere train.csv și test.csv?
**R:** NU! Acestea au fost eliminate. Folosiți doar `train_split.csv` și `eval_split.csv`.

### Q5: Cum verific că totul funcționează corect?
**R:** Rulați următorul test rapid:

```python
import pandas as pd

# Încărcare date
df_train = pd.read_csv('train_split.csv', parse_dates=['data_ora'])
df_test = pd.read_csv('eval_split.csv', parse_dates=['data_ora'])

# Verificare
print(f"Train: {df_train.shape[0]} rânduri, {df_train.shape[1]} coloane")
print(f"Test: {df_test.shape[0]} rânduri, {df_test.shape[1]} coloane")
print(f"Test are coloana 'total': {'total' in df_test.columns}")

# Așteptat:
# Train: 6878 rânduri, 12 coloane
# Test: 4008 rânduri, 12 coloane
# Test are coloana 'total': True
```

---

## Performanță Așteptată

### Dataset
- **Train:** 6,878 observații (2011-01-01 până 2012-12-12)
- **Test:** 4,008 observații (2011-01-13 până 2012-12-19)

### Model
- **Algoritm:** Gradient Boosting Regressor
- **RMSE:** ~97 închirieri (pe setul de test)
- **Eroare relativă:** ~49% din media închirierilor

### Timp Execuție (estimat)
- Încărcare date: < 1 secundă
- Feature engineering: < 1 secundă
- Antrenare model (3 modele cantilă): ~30-60 secunde
- Predicții: < 1 secundă
- **Total:** ~1-2 minute

---

## Suport și Depanare

### Eroare: "FileNotFoundError: train.csv"
**Soluție:** Folosiți `train_split.csv` în loc de `train.csv`

### Eroare: "ModuleNotFoundError: No module named 'pandas'"
**Soluție:** Instalați dependențele:
```bash
pip install -r requirements.txt
```

### Eroare: "KeyError: 'total'"
**Soluție:** Asigurați-vă că folosiți `eval_split.csv`, nu vechiul `test.csv`

### Warning-uri în timpul antrenării
**Notă:** Unele warning-uri de la scikit-learn sunt normale și pot fi ignorate.

---

## Verificare Finală (Checklist)

Înainte de predare, verificați:

- [ ] ✅ Folosiți `train_split.csv` și `eval_split.csv` (NU train.csv/test.csv)
- [ ] ✅ Toate celulele din notebook rulează fără erori
- [ ] ✅ Graficele EDA sunt generate corect
- [ ] ✅ Modelul se antrenează cu succes
- [ ] ✅ RMSE este calculat pe setul de test
- [ ] ✅ Fișierul `predictii_biciclete.csv` este generat
- [ ] ✅ Raportul tehnic este completat
- [ ] ✅ Codul este documentat și lizibil

---

## Contact și Resurse

- **Raport Tehnic Complet:** `Raport_Tema1_Inchiriere_Biciclete.md`
- **Notebook:** `tema1_abdulkadir gobena-denboba.ipynb`
- **Dataset:** Archive `inchiriere-biciclete.zip` (conține train_split.csv și eval_split.csv)

---

**Ultima actualizare:** Noiembrie 2024  
**Versiune dataset:** 2.0 (cu etichete în setul de test)  
**Status:** ✅ Complet și funcțional
