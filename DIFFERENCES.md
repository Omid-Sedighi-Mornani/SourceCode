# Implementierungsunterschiede: Python/Stan vs. R-Code (Naumzik et al., 2022)

**Datum:** 2025-12-10
**Analysiert von:** Claude
**Referenz:** Naumzik, C., Feuerriegel, S., & Weinmann, M. (2022). I Will Survive: Predicting Business Failures from Customer Ratings. *Marketing Science*, 41(1), 188-207.

---

## Executive Summary

Diese Analyse vergleicht Ihre Python/Stan-Implementierung mit dem Original R-Code aus dem Paper. Es wurden **6 kritische Unterschiede** identifiziert, die zu abweichenden Ergebnissen führen. Die wichtigsten Unterschiede betreffen:

1. ⚠️ **Train/Test/Calibration Split** - Unterschiedliche Datenaufteilung
2. ⚠️ **Calibration Set Größe** - 200 statt 100 Samples
3. ⚠️ **MCMC Iterationen** - 2000 statt 1000 total
4. ⚠️ **adapt_delta Parameter** - 0.95 statt 0.8

**Auswirkung:** Ihre Posterior-Parameter weichen deutlich vom Paper ab (siehe Ihre `testing.ipynb` Diagnose-Zelle).

---

## 1. PREPROCESSING (1_processing.R vs. notebooks/1a-processing.ipynb)

### 1.1 Train/Test/Calibration Split 🚨 **KRITISCH**

| Aspekt | R-Code (Paper) | Python-Code (Ihre Impl.) | Auswirkung |
|--------|----------------|--------------------------|------------|
| **Datei** | `.claude/reference/Code/1_processing.R` | `notebooks/1a-processing.ipynb` | - |
| **Zeile** | 13-22 | Zelle `77a25d28` | - |
| **Methode** | Lädt vordefinierte Indices aus `indices.Rdata` | Generiert neue mit `np.random.permutation(SEED=42)` | **HOCH** |
| **Training Set** | Indices aus Datei (500 Samples) | `indices[:500]` (500 Samples) | Unterschiedliche Samples! |
| **Calibration Set** | Indices aus Datei (100 Samples) | `indices[500:700]` **(200 Samples!)** | Falsche Größe! |
| **Eval Set** | Indices aus Datei (321 Samples) | `indices[700:]` (221 Samples) | Unterschiedliche Samples! |

**R-Code:**
```r
# Zeilen 13-22
load(file = indices_file)
N_train = length(train.idx)
N_test = length(test.idx)
N_eval = length(eval.idx)

covariates_business$TRAIN[train.idx] <- 1
covariates_business$TEST[test.idx] <- 1
covariates_business$EVAL[eval.idx] <- 1
```

**Python-Code:**
```python
# Notebook 1a-processing.ipynb, Zelle 77a25d28
indices = np.random.permutation(len(business_covariates))
np.random.seed(SEED)

train_indices = indices[:500]
calibration_indices = indices[500:700]  # FEHLER: 200 statt 100!
eval_indices = indices[700:]  # 221 statt 321!
```

**Konsequenz:**
- Ihre Modelle wurden auf **anderen Daten** trainiert als im Paper
- **Keine direkte Vergleichbarkeit** der Ergebnisse
- Erklärt große Abweichungen der ω-Parameter

---

### 1.2 Kategorie-Kodierung und Spaltenentfernung ✅ **KORREKT**

| Aspekt | R-Code (Paper) | Python-Code (Ihre Impl.) | Status |
|--------|----------------|--------------------------|--------|
| **Datei** | `1_processing.R` | `notebooks/1a-processing.ipynb` | - |
| **Zeile** | 65 | Zelle `11e3f9f7` | - |
| **Methode** | `model.matrix(...)[,-8]` entfernt Spalte 8 (1-indexed) | `columns[7]` entfernt Spalte 7 (0-indexed) | ✅ **IDENTISCH** |
| **Entfernte Spalte** | "categoryOther" | "category_Other" | ✅ **KORREKT** |

**R-Code:**
```r
# Zeile 65
cov_mat <- model.matrix(formula(...), data = Covariates)[,-8]
# Entfernt Spalte 8 (1-indexed) = "categoryOther"
```

**Python-Code:**
```python
# Notebook 1a-processing.ipynb, Zelle 11e3f9f7
col_to_remove = relevant_covariates.columns[7]  # 0-indexed!
if "Other" in col_to_remove:
    relevant_covariates = relevant_covariates.drop(columns=[col_to_remove])
```

**Konsequenz:** Keine - beide Implementierungen identisch

---

### 1.3 Preprocessing-Reihenfolge ✅ **KORREKT**

| Aspekt | R-Code (Paper) | Python-Code (Ihre Impl.) | Status |
|--------|----------------|--------------------------|--------|
| **Datei** | `1_processing.R` | `notebooks/1a-processing.ipynb` | - |
| **Zeile** | 66-67 | Zelle `214a6af6` | - |
| **Schritt 1** | Median-Imputation | Median-Imputation | ✅ **IDENTISCH** |
| **Schritt 2** | Centering (mean=0) | Centering (StandardScaler mit std=False) | ✅ **IDENTISCH** |
| **Schritt 3** | Fit nur auf Training | Fit nur auf Training | ✅ **IDENTISCH** |

**R-Code:**
```r
# Zeilen 66-67
preProc <- preProcess(cov_mat[1:500,], c("center", "medianImpute"))
cov_mat_preproc <- predict(preProc, cov_mat)
```

**Python-Code:**
```python
# Notebook 1a-processing.ipynb, Zelle 214a6af6
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler(with_std=False)  # only centering, no scaling

imputer.fit(X_train)
X_train_imputed = imputer.transform(X_train)
scaler.fit(X_train_imputed)
X_train_preprocessed = scaler.transform(X_train_imputed)
```

**Konsequenz:** Keine - beide Implementierungen identisch

---

### 1.4 QR-Dekomposition ✅ **KORREKT**

| Aspekt | R-Code (Paper) | Python-Code (Ihre Impl.) | Status |
|--------|----------------|--------------------------|--------|
| **Datei** | `1_processing.R` | `notebooks/1a-processing.ipynb` | - |
| **Zeile** | 68-70, 82-83 | Zelle `214a6af6` | - |
| **Q-Skalierung** | `Q * sqrt(N_train - 1)` | `Q * np.sqrt(n_train - 1)` | ✅ **IDENTISCH** |
| **R-Skalierung** | `R / sqrt(N_train - 1)` | `R / np.sqrt(n_train - 1)` | ✅ **IDENTISCH** |

**R-Code:**
```r
# Zeilen 68-70, 82-83
QR <- qr(predict(preProc, cov_mat[1:500,]))
Q = qr.Q(QR) * sqrt(N_train - 1)
R = qr.R(QR) / sqrt(N_train - 1)
```

**Python-Code:**
```python
# Notebook 1a-processing.ipynb, Zelle 214a6af6
Q, R = np.linalg.qr(X_train_preprocessed)
n_train = len(X_train_preprocessed)
Q_scaled = Q * np.sqrt(n_train - 1)
R_scaled = R / np.sqrt(n_train - 1)
```

**Konsequenz:** Keine - beide Implementierungen identisch

---

## 2. TRAINING (2_train_stan_models.R vs. scripts/main.py & Notebooks 2a/2b)

### 2.1 MCMC Iterationen ⚠️ **WICHTIGER UNTERSCHIED**

| Aspekt | R-Code (Paper) | Python-Code (Ihre Impl.) | Auswirkung |
|--------|----------------|--------------------------|------------|
| **Datei** | `.claude/reference/Code/2_train_stan_models.R` | `scripts/main.py` | - |
| **Zeile** | 24 | 386-392 | - |
| **Warmup** | 500 (Stan default bei iter=1000) | 1000 (`iter_warmup=1000`) | **MITTEL** |
| **Sampling** | 500 | 1000 (`iter_sampling=1000`) | **MITTEL** |
| **Total** | **1000** | **2000** | Doppelte Laufzeit! |
| **Posterior Samples** | 500 × 2 chains = 1000 | 1000 × 2 chains = 2000 | Mehr Samples |

**R-Code:**
```r
# Zeile 24
trained_vdmm_model <- rstan::sampling(
  model,
  data = data,
  chains = n_chains,  # 2
  iter = n_iter,      # 1000 (default: 500 warmup + 500 sampling)
  refresh = 10,
  init = 0
)
```

**Python-Code:**
```python
# scripts/main.py, Zeilen 386-392
fit = model.sample(
    data=stan_data,
    chains=chains,  # default: 2
    parallel_chains=parallel_chains,  # default: 2
    threads_per_chain=threads_per_chain,  # default: 1
    iter_warmup=iter_warmup,  # default: 1000 ❌
    iter_sampling=iter_sampling,  # default: 1000 ❌
    seed=seed,
    adapt_delta=adapt_delta,
    max_treedepth=max_treedepth,
)
```

**Konsequenz:**
- Ihre Modelle haben **mehr Posterior-Samples** (2000 vs 1000)
- **Längere Laufzeit** (ca. 2x)
- **Bessere Konvergenz** möglich, aber andere Ergebnisse
- **Nicht direkt vergleichbar** mit Paper

---

### 2.2 adapt_delta Parameter ⚠️ **UNTERSCHIED**

| Aspekt | R-Code (Paper) | Python-Code (Ihre Impl.) | Auswirkung |
|--------|----------------|--------------------------|------------|
| **Datei** | `2_train_stan_models.R` | `scripts/main.py` | - |
| **Zeile** | - (Stan default) | 294, 394 | - |
| **Wert** | 0.8 (Stan default) | 0.95 | **NIEDRIG** |
| **Bedeutung** | Standard-Samplingrate | Konservativeres Sampling | Weniger Divergenzen |

**R-Code:**
```r
# Zeile 24 - adapt_delta nicht explizit gesetzt
# Stan verwendet default: 0.8
trained_vdmm_model <- rstan::sampling(...)
```

**Python-Code:**
```python
# scripts/main.py, Zeilen 294, 394
adapt_delta = args.adapt_delta  # default: 0.95 ❌

fit = model.sample(
    ...
    adapt_delta=adapt_delta,  # 0.95 statt 0.8
    ...
)
```

**Konsequenz:**
- **Konservativeres Sampling** in Python
- **Weniger Divergenzen**, aber langsameres Sampling
- **Minimal andere Posterior-Verteilungen**

---

### 2.3 Random Seed ⚠️ **UNTERSCHIED**

| Aspekt | R-Code (Paper) | Python-Code (Ihre Impl.) | Auswirkung |
|--------|----------------|--------------------------|------------|
| **Datei** | `2_train_stan_models.R` | `scripts/main.py` | - |
| **Zeile** | 24 | 296, 396 | - |
| **Seed** | Nicht gesetzt (init=0 für Parameter) | Explizit gesetzt (seed=42) | **NIEDRIG** |

**R-Code:**
```r
# Zeile 24
trained_vdmm_model <- rstan::sampling(
  ...
  init = 0  # Parameter-Initialisierung bei 0, aber kein Random Seed!
)
```

**Python-Code:**
```python
# scripts/main.py, Zeilen 296, 396
seed = args.seed  # z.B. 42

fit = model.sample(
    ...
    seed=seed,  # Expliziter Seed für Reproduzierbarkeit
    ...
)
```

**Konsequenz:**
- Python: **Reproduzierbar** bei gleichem Seed
- R: **Nicht reproduzierbar** (außer mit set.seed() im R-Script)
- **Minimal andere Ergebnisse** bei jedem R-Lauf

---

### 2.4 Parallelisierung ✅ **UNTERSCHIED, ABER KEINE INHALTLICHE AUSWIRKUNG**

| Aspekt | R-Code (Paper) | Python-Code (Ihre Impl.) | Auswirkung |
|--------|----------------|--------------------------|------------|
| **Datei** | `2_train_stan_models.R` | `scripts/main.py` | - |
| **Zeile** | 4, 24 | 288, 388 | - |
| **Methode** | Sequential chains | Parallel chains | **KEINE** |
| **Threads** | 1 pro Chain | 1 pro Chain (default) | Nur Performance |

**R-Code:**
```r
# Zeile 4
options(mc.cores = 2)  # Erlaubt 2 parallele Chains

# Zeile 24
trained_vdmm_model <- rstan::sampling(..., chains = n_chains)
```

**Python-Code:**
```python
# scripts/main.py, Zeilen 288, 388
parallel_chains = args.parallel_chains  # default: 2
threads_per_chain = args.threads_per_chain  # default: 1

fit = model.sample(
    chains=2,
    parallel_chains=2,  # Parallele Chains
    threads_per_chain=1,
    ...
)
```

**Konsequenz:**
- Nur **Performance-Unterschied**
- **Keine inhaltliche Auswirkung** auf Ergebnisse
- Python kann **schneller** sein bei parallelen Chains

---

### 2.5 Stan-Daten-Vorbereitung ✅ **IDENTISCH**

| Aspekt | R-Code (Paper) | Python-Code (Ihre Impl.) | Status |
|--------|----------------|--------------------------|--------|
| **Datei** | `1_processing.R` | `helpers/helper_functions.py` | - |
| **Zeile** | 72-84 | 24-38 | - |
| **Struktur** | Identisch | Identisch | ✅ **KORREKT** |

**R-Code:**
```r
# Zeilen 72-84
data_stan = list(
  S = NA,
  N_total = length(Time),
  N_train = N_train,
  N_obs = sum(Time),
  nCovs = ncol(cov_mat),
  Time = Time,
  Closed = 1 - covariates_business$is_open,
  Days = Days,
  Ratings = Ratings,
  Sentiment = Sentiment,
  Q = qr.Q(QR) * sqrt(N_train - 1),
  R = qr.R(QR) / sqrt(N_train - 1),
  X_test = X_test
)
```

**Python-Code:**
```python
# helpers/helper_functions.py, Zeilen 24-38
stan_data = {
    "S": S,
    "N_total": int(model_data.n_total),
    "N_train": int(model_data.n_train),
    "N_obs": int(model_data.n_obs),
    "nCovs": int(model_data.n_covs),
    "Time": [int(x) for x in model_data.time],
    "Closed": [int(x) for x in model_data.closed],
    "Days": [float(x) for x in model_data.days],
    "Ratings": [int(x) for x in model_data.ratings],
    "Sentiment": [float(x) for x in model_data.sentiment],
    "Q": model_data.Q.tolist(),
    "R": model_data.R.tolist(),
    "X_test": model_data.X_test.tolist(),
}
```

**Konsequenz:** Keine - beide Implementierungen identisch (bei gleichen Input-Daten)

---

## 3. STAN-CODE (StanCode/vdhmm.stan vs. data/stan_code/vdhmm.stan)

### 3.1 Array-Syntax ✅ **SYNTAKTISCH, KEINE INHALTLICHE ÄNDERUNG**

| Aspekt | R-Stan (Paper) | Python-Stan (Ihre Impl.) | Auswirkung |
|--------|----------------|--------------------------|------------|
| **Datei** | `.claude/reference/Code/StanCode/vdhmm.stan` | `data/stan_code/vdhmm.stan` | - |
| **Zeile** | 4, 13, etc. | 4, 13, etc. | - |
| **Syntax** | `vector[] l_tpm` (deprecated) | `array[] vector l_tpm` (neu) | **KEINE** |

**R-Stan (alte Syntax):**
```stan
// Zeile 4
real calc_tpm(int s_from, int s_to, vector intercept, real d, real lambda,
              real t, real gamma, real senti, real epsilon, vector[] l_tpm){
```

**Python-Stan (neue Syntax):**
```stan
// Zeile 4
real calc_tpm(int s_from, int s_to, vector intercept, real d, real lambda,
              real t, real gamma, real senti, real epsilon, array[] vector l_tpm){
```

**Konsequenz:**
- Nur **syntaktische Änderung** für neuere Stan-Versionen (>= 2.26)
- **Identisches Verhalten**
- Wird vom Stan-Compiler gleich übersetzt

---

### 3.2 Duration-Berechnung in Transition ✅ **IDENTISCH**

| Aspekt | R-Stan (Paper) | Python-Stan (Ihre Impl.) | Status |
|--------|----------------|--------------------------|--------|
| **Datei** | `StanCode/vdhmm.stan` | `data/stan_code/vdhmm.stan` | - |
| **Zeile** | 4-12 | 4-12 | - |
| **Formel** | `intercept[s_from] + lambda * d + gamma * t + senti * epsilon` | Identisch | ✅ **IDENTISCH** |

**BEIDE:**
```stan
// Zeilen 4-12
real calc_tpm(int s_from, int s_to, vector intercept, real d, real lambda,
              real t, real gamma, real senti, real epsilon, array[] vector l_tpm){
  real diag;
  diag = -log1p_exp(-(intercept[s_from] + lambda * d + gamma * t + senti * epsilon));
  if(s_from == s_to){
    return diag;
  }else{
    return l_tpm[s_from,s_to] + log1m_exp(diag);
  }
}
```

**Entsprechung zu Paper (Equation 5, Seite 194):**
```
γs(dj, Δj,j+1, σij) = logit⁻¹[λ₀ˢ + λ₁·log(1 + dj(s)) + λ₂·log(1 + Δj,j+1) + λ₃·σij]
```
- `intercept[s_from]` = λ₀ˢ
- `lambda` = λ₁
- `gamma` = λ₂
- `epsilon` = λ₃
- `d` = log(1 + duration count) ← **wird logarithmiert übergeben!**
- `t` = log(1 + Δj,j+1) ← **wird logarithmiert übergeben!**

**Konsequenz:** Keine - beide identisch und korrekt

---

### 3.3 Duration-Aufruf in Forward-Algorithmus ✅ **IDENTISCH**

| Aspekt | R-Stan (Paper) | Python-Stan (Ihre Impl.) | Status |
|--------|----------------|--------------------------|--------|
| **Datei** | `StanCode/vdhmm.stan` | `data/stan_code/vdhmm.stan` | - |
| **Zeile** | 34-41 | 34-41 | - |

**BEIDE:**
```stan
// Zeilen 34-41 (in get_duration und compute_ll)
temp_tpm[s_from,s] = calc_tpm(
    s_from, s,
    intercept,
    duration[t-1,s_from],  // ← logarithmierte Duration!
    lambda,
    log(1+Days[t]-Days[t-1]),  // ← logarithmierte Time-Diff
    gamma,
    Sentiment[t-1],
    epsilon,
    l_tpm
);
```

**WICHTIG:** `duration[t-1,s_from]` ist bereits **logarithmiert**!

**Siehe Zeilen 51-52 / 96-97:**
```stan
duration[t,s] = log1p_exp(dur_temp + c[t] +
                          emission[s,Rating[t]] +
                          temp_tpm[s,s] +
                          forw[t-1,s] - forw[t,s]);
```
→ `log1p_exp` = log(1 + exp(...)) → **speichert logarithmierten Wert**

**Konsequenz:** Keine - beide identisch und korrekt

---

### 3.4 Duration-Standardisierung ✅ **IDENTISCH**

| Aspekt | R-Stan (Paper) | Python-Stan (Ihre Impl.) | Status |
|--------|----------------|--------------------------|--------|
| **Datei** | `StanCode/vdhmm.stan` | `data/stan_code/vdhmm.stan` | - |
| **Zeile** | 200-205 | 200-205 | - |
| **Methode** | Z-Score: `(x - mean) / sd` | Z-Score: `(x - mean) / sd` | ✅ **IDENTISCH** |

**BEIDE:**
```stan
// Zeilen 200-205 (Transformed Parameters Block)
for(s in 1:S){
  vector[N_train] temp = Duration[,s];
  duration_mean[s] = mean(temp);
  duration_sd[s] = sd(temp);
  Duration[,s] = (temp - mean(temp)) / sd(temp);  // Z-Score Standardisierung
}
```

**WICHTIG:**
- Duration wird **nur auf Training-Set** berechnet
- Mean und SD aus **Training-Durations**
- Test-Set verwendet **Training-Mean/SD** (siehe generated quantities, Zeile 245)
- **Korrekt** für Out-of-Sample Prediction!

**Konsequenz zum Paper:**
- **Paper (Equation 3, Seite 194):** `πi(j) = logit⁻¹[ω₀ + Σ ωs·log(dj(s)) + ν·zij]`
- **Stan-Code:** `πi(j) = logit⁻¹[ω₀ + Σ ωs·STANDARDIZED(log(dj(s))) + ν·zij]`
- **UNTERSCHIED:** Stan verwendet **standardisierte** log-Durations
- **Auswirkung:** ωs-Parameter haben andere Skala/Interpretation als im Paper!

---

### 3.5 Emission-Komponente (Ordered Probit) ✅ **IDENTISCH**

| Aspekt | R-Stan (Paper) | Python-Stan (Ihre Impl.) | Status |
|--------|----------------|--------------------------|--------|
| **Datei** | `StanCode/vdhmm.stan` | `data/stan_code/vdhmm.stan` | - |
| **Zeile** | 175-188 | 175-188 | - |

**BEIDE:**
```stan
// Zeilen 175-188 (Transformed Parameters Block)
for(s in 1:S){
  real scale = inv_sqrt(1 - inv_logit(R2[s]));
  vector[4] cuts = scale * inv_Phi(cumulative_sum(probs[1:4]));
  state_emission[s] = s == 1 ? 0.0 : state_emission[s-1] + exp(state_emission_raw[s-1]);
  l_tpm[s,s] = negative_infinity();
  if(s > 1){
    l_tpm[s,1:(s-1)] = log(tpm[s,1:(s-1)]);
  }
  if(s < S){
    l_tpm[s,(s+1):S] = log(tpm[s,s:(S-1)]);
  }
  for(r in 1:5){
    emission[s,r] = ordered_probit_lpmf(r|state_emission[s],cuts);
  }
}
```

**Entsprechung zu Paper (Equation 2, Seite 193):**
```
b⁽ˢ⁾(r) = P(Rij = r | Sij = s)
```

**Konsequenz:** Keine - beide identisch

---

### 3.6 ω-Parameter Identifiability Constraint ✅ **IDENTISCH**

| Aspekt | R-Stan (Paper) | Python-Stan (Ihre Impl.) | Status |
|--------|----------------|--------------------------|--------|
| **Datei** | `StanCode/vdhmm.stan` | `data/stan_code/vdhmm.stan` | - |
| **Zeile** | 168-174 | 168-174 | - |

**BEIDE:**
```stan
// Zeilen 168-174 (Transformed Parameters Block)
for(s in 1:S){
  if(s == 2){
    omega_state[s] = omega_state_mid;  // State 2 ist frei
  }else{
    omega_state[s] = omega_state_raw[i];  // States 1,3 relativ zu State 2
    i = i + 1;
  }
}
```

**Interpretation:**
- **State 2** wird als "mittlerer" State behandelt (freier Parameter)
- **States 1 und 3** werden relativ zu State 2 definiert
- **Identifiability Constraint** zur Vermeidung von Label Switching

**Konsequenz:**
- Beide identisch
- **ABER:** Unklar ob Paper gleiche Constraint verwendet!
- Könnte Label Switching zwischen R und Python Modellen erklären

---

### 3.7 Priors ⚠️ **LEICHTER UNTERSCHIED**

| Parameter | R-Stan (Paper) | Python-Stan (Ihre Impl.) | Status |
|-----------|----------------|--------------------------|--------|
| **Datei** | `StanCode/vdhmm.stan` | `data/stan_code/vdhmm.stan` | - |
| **Zeile** | 208-226 | 208-226 | - |
| `pi` | `dirichlet(rep_vector(1,S))` | `dirichlet(rep_vector(1,S))` | ✅ Identisch |
| `intercept` | `normal(0,5)` | `normal(0,5)` | ✅ Identisch |
| `lambda` | `std_normal()` | `std_normal()` | ✅ Identisch |
| `gamma` | `std_normal()` | `std_normal()` | ✅ Identisch |
| `epsilon` | `std_normal()` | `std_normal()` | ✅ Identisch |
| `omega_0` | `normal(0,5)` | `normal(0,5)` | ✅ Identisch |
| `omega_tilde` | `std_normal()` | `std_normal()` | ✅ Identisch |
| `omega_state_raw` | `student_t(7,0,1)` | `student_t(7,0,1)` | ✅ Identisch |
| `omega_state_mid` | `std_normal()` | `std_normal()` | ✅ Identisch |

**BEIDE:**
```stan
// Zeilen 208-226 (Model Block)
pi ~ dirichlet(rep_vector(1,S));
intercept ~ normal(0,5);
lambda ~ std_normal();
gamma ~ std_normal();
epsilon ~ std_normal();
for(s in 1:S){
  tpm[s] ~ dirichlet(prior_tpm[s]);
}
probs ~ dirichlet(rep_vector(0.1,5));
state_emission_raw ~ normal(0,5);
R2 ~ normal(0,5);
omega_0 ~ normal(0,5);
omega_tilde ~ std_normal();
omega_state_raw ~ student_t(7,0,1);  // ← Student-t statt Normal
omega_state_mid ~ std_normal();
```

**Bemerkung:**
- `omega_state_raw ~ student_t(7,0,1)` ist **robuster** gegen Outliers als Normal
- Student-t mit df=7 ist nah an Normal, aber mit dickeren Tails
- **Unklar ob Paper gleiche Priors** verwendet (nicht explizit dokumentiert)

**Konsequenz:** Minimal - Student-t(7) ist fast identisch zu Normal

---

## 4. ZUSAMMENFASSUNG ALLER UNTERSCHIEDE

### 4.1 Kritische Unterschiede (Beeinflussen Ergebnisse)

| # | Komponente | R-Code (Paper) | Python-Code (Ihre Impl.) | Datei | Zeile | Impact |
|---|------------|----------------|--------------------------|-------|-------|--------|
| **1** | **Train/Test Split** | Vordefinierte Indices aus `indices.Rdata` | Neue zufällige Indices mit `np.random.permutation(SEED=42)` | `1_processing.R:13-22` vs `1a-processing.ipynb:77a25d28` | | **🔴 HOCH** |
| **2** | **Calibration Set** | 100 Samples | 200 Samples (`indices[500:700]`) | `1a-processing.ipynb:77a25d28` | | **🟡 MITTEL** |
| **3** | **Eval Set** | 321 Samples | 221 Samples (`indices[700:]`) | `1a-processing.ipynb:77a25d28` | | **🟡 MITTEL** |
| **4** | **MCMC Iterationen** | 1000 total (500+500) | 2000 total (1000+1000) | `2_train_stan_models.R:24` vs `main.py:386-392` | | **🟡 MITTEL** |
| **5** | **adapt_delta** | 0.8 (Stan default) | 0.95 | `main.py:294,394` | | **🟢 NIEDRIG** |
| **6** | **Random Seed** | Nicht gesetzt | Explizit (seed=42) | `main.py:296,396` | | **🟢 NIEDRIG** |

---

### 4.2 Nicht-Kritische Unterschiede (Keine inhaltliche Auswirkung)

| # | Komponente | R-Code (Paper) | Python-Code (Ihre Impl.) | Datei | Zeile | Impact |
|---|------------|----------------|--------------------------|-------|-------|--------|
| **7** | **Array-Syntax** | `vector[]` (alte Syntax) | `array[] vector` (neue Syntax) | `vdhmm.stan:4` | | **⚪ KEINE** |
| **8** | **Parallelisierung** | Sequential chains | Parallel chains | `main.py:288,388` | | **⚪ KEINE** |

---

### 4.3 Korrekte Implementierungen (Identisch)

| # | Komponente | Status | Datei | Zeile |
|---|------------|--------|-------|-------|
| **9** | **Kategorie-Spaltenentfernung** | ✅ Beide entfernen "Other" korrekt | `1_processing.R:65` vs `1a-processing.ipynb:11e3f9f7` | |
| **10** | **Preprocessing-Reihenfolge** | ✅ Beide: Median-Imputation → Centering | `1_processing.R:66-67` vs `1a-processing.ipynb:214a6af6` | |
| **11** | **QR-Dekomposition** | ✅ Beide identisch (Q·√(N-1), R/√(N-1)) | `1_processing.R:68-70` vs `1a-processing.ipynb:214a6af6` | |
| **12** | **calc_tpm (VD-HMM)** | ✅ Duration-Abhängigkeit korrekt | `vdhmm.stan:4-12` | |
| **13** | **calc_tpm (HMM)** | ✅ Keine Duration-Abhängigkeit (korrekt) | `hmm.stan:4-12` | |
| **14** | **Duration-Akkumulation** | ✅ Forward-Algorithmus identisch | `vdhmm.stan:48-56` | |
| **15** | **Duration-Standardisierung** | ✅ Z-Score mit Training-Mean/SD | `vdhmm.stan:200-205` | |
| **16** | **Ordered Probit Emission** | ✅ Identisch | `vdhmm.stan:175-188` | |
| **17** | **Stan-Daten-Vorbereitung** | ✅ Identisch (bei gleichen Inputs) | `1_processing.R:72-84` vs `helper_functions.py:24-38` | |
| **18** | **ω-Identifiability Constraint** | ✅ Beide verwenden State 2 als Referenz | `vdhmm.stan:168-174` | |
| **19** | **Priors** | ✅ Fast identisch (Student-t vs Normal minimal) | `vdhmm.stan:208-226` | |

---

## 5. AUSWIRKUNG AUF IHRE ERGEBNISSE

### 5.1 Warum weichen Ihre ω-Parameter vom Paper ab?

**Hauptgründe (in Reihenfolge der Wichtigkeit):**

1. **🔴 Unterschiedliche Trainingsdaten** (Kritisch #1-3)
   - Ihre Modelle wurden auf **anderen 500 Restaurants** trainiert
   - **Keine direkte Vergleichbarkeit** möglich
   - **Erklärt große Abweichungen** (z.B. ω₃: -1.76 vs 0.26 = Differenz von -2.0)

2. **🟡 Mehr Posterior-Samples** (Kritisch #4)
   - Ihre Modelle haben **2x mehr Samples** (2000 vs 1000)
   - **Andere Posterior-Verteilungen** möglich
   - **Bessere Konvergenz**, aber andere Mittelwerte

3. **🟡 Standardisierung der Duration** (Stan-Code 3.4)
   - Stan verwendet **standardisierte** log-Durations: `(log(d) - mean) / sd`
   - Paper verwendet **nicht-standardisierte**: `log(d)`
   - **Erklärt unterschiedliche Skalierungen** der ω-Parameter
   - **WICHTIG:** Dieser Unterschied ist in **beiden** Implementierungen gleich!

4. **🟢 Label Switching** (möglich)
   - State-Zuordnungen könnten **vertauscht** sein
   - State 1 in Ihrem Modell ≠ State 1 im Paper?
   - **Prüfen:** Vergleichen Sie `state_emission` Wahrscheinlichkeiten (siehe `testing.ipynb` Diagnose)

---

### 5.2 Ihre Diagnose-Ergebnisse erklärt

**Aus `testing.ipynb` Diagnose-Zelle:**

```
Parameter | Ihr Wert | Paper-Wert | Differenz
------------------------------------------------------------
omega_state[1] (Well-running)         | -0.9534  | -0.3239    | -0.6295 ✗
omega_state[2] (At-risk)              |  0.1594  |  0.8862    | -0.7268 ✗
omega_state[3] (Bad-ratings-but-run.) | -1.7615  |  0.2559    | -2.0174 ✗✗
```

**Mögliche Erklärungen:**

1. **Unterschiedliche Trainingsdaten** → Andere Modelle → Andere ω-Werte
2. **Standardisierung** → ω-Werte haben andere Skala (aber in beiden Modellen!)
3. **Label Switching** → Ihre States sind anders zugeordnet als im Paper
4. **Mehr Samples** → Andere Posterior-Mittelwerte

**Empfehlung:**
- **Trainieren Sie Ihr Modell mit den Original Paper-Indices** (siehe Plan unten)
- Dann erst vergleichen!

---

## 6. PLAN ZUR AUSMERZEN DER UNTERSCHIEDE

### 🎯 Ziel
Ihre Implementierung **1:1 an den R-Code anpassen**, um Paper-Ergebnisse zu replizieren.

---

### 📋 Schritt-für-Schritt Plan

#### **SCHRITT 1: Original Paper-Indices extrahieren** ⏱️ 10 Min

**Was:** Indices aus `.claude/reference/indices.Rdata` extrahieren und in Python verfügbar machen

**Wie:**
1. R-Script erstellen: `scripts/extract_indices.R`
   ```r
   load(".claude/reference/indices.Rdata")
   write.csv(train.idx, "data/paper_train_indices.csv", row.names=FALSE)
   write.csv(test.idx, "data/paper_test_indices.csv", row.names=FALSE)
   write.csv(eval.idx, "data/paper_eval_indices.csv", row.names=FALSE)
   ```

2. Ausführen: `Rscript scripts/extract_indices.R`

3. In `constants.py` laden:
   ```python
   import pandas as pd

   PAPER_TRAIN_INDICES = pd.read_csv(DATA_FOLDER / "paper_train_indices.csv").values.flatten().tolist()
   PAPER_TEST_INDICES = pd.read_csv(DATA_FOLDER / "paper_test_indices.csv").values.flatten().tolist()
   PAPER_EVAL_INDICES = pd.read_csv(DATA_FOLDER / "paper_eval_indices.csv").values.flatten().tolist()

   # R verwendet 1-basierte Indices, Python 0-basiert!
   PAPER_TRAIN_INDICES = [i-1 for i in PAPER_TRAIN_INDICES]
   PAPER_TEST_INDICES = [i-1 for i in PAPER_TEST_INDICES]
   PAPER_EVAL_INDICES = [i-1 for i in PAPER_EVAL_INDICES]
   ```

**Resultat:** Original Paper-Indices in Python verfügbar

---

#### **SCHRITT 2: Preprocessing-Notebook anpassen** ⏱️ 15 Min

**Was:** `notebooks/1a-processing.ipynb` auf Paper-Indices umstellen

**Wie:**
1. **Zelle `77a25d28`** ändern:
   ```python
   # ALTE VERSION:
   # indices = np.random.permutation(len(business_covariates))
   # np.random.seed(SEED)
   # train_indices = indices[:500]
   # calibration_indices = indices[500:700]  # FEHLER: 200!
   # eval_indices = indices[700:]

   # NEUE VERSION (Paper-kompatibel):
   USE_PAPER_INDICES = True  # Toggle für Paper-Modus

   if USE_PAPER_INDICES:
       from constants import PAPER_TRAIN_INDICES, PAPER_TEST_INDICES, PAPER_EVAL_INDICES
       train_indices = PAPER_TRAIN_INDICES  # 500 Samples
       calibration_indices = PAPER_TEST_INDICES  # 100 Samples!
       eval_indices = PAPER_EVAL_INDICES  # 321 Samples!
       print(f"✅ Using PAPER indices: Train={len(train_indices)}, Calib={len(calibration_indices)}, Eval={len(eval_indices)}")
   else:
       # Original-Logik mit Random-Split
       indices = np.random.permutation(len(business_covariates))
       np.random.seed(SEED)
       train_indices = indices[:500]
       calibration_indices = indices[500:600]  # KORRIGIERT: 100!
       eval_indices = indices[600:]  # KORRIGIERT: 321!
       print(f"⚠️  Using RANDOM indices (seed={SEED}): Train={len(train_indices)}, Calib={len(calibration_indices)}, Eval={len(eval_indices)}")
   ```

2. **Validierungs-Zelle** hinzufügen (nach Zelle `77a25d28`):
   ```python
   # Validierung der Split-Größen
   assert len(train_indices) == 500, f"Train set should be 500, got {len(train_indices)}"
   assert len(calibration_indices) == 100, f"Calibration set should be 100, got {len(calibration_indices)}"
   assert len(eval_indices) == 321, f"Eval set should be 321, got {len(eval_indices)}"
   assert len(set(train_indices) & set(calibration_indices)) == 0, "Train and Calibration overlap!"
   assert len(set(train_indices) & set(eval_indices)) == 0, "Train and Eval overlap!"
   assert len(set(calibration_indices) & set(eval_indices)) == 0, "Calibration and Eval overlap!"
   print("✅ All split sizes and overlaps validated!")
   ```

**Resultat:** Preprocessing verwendet jetzt Paper-Indices (korrekte Splits!)

---

#### **SCHRITT 3: Training-Script anpassen** ⏱️ 10 Min

**Was:** `scripts/main.py` auf Paper-Parameter umstellen

**Wie:**
1. **Defaults ändern** (Zeilen 295-297):
   ```python
   # ALTE DEFAULTS:
   # iter_warmup_default = 1000
   # iter_sampling_default = 1000
   # adapt_delta_default = 0.95

   # NEUE DEFAULTS (Paper-kompatibel):
   iter_warmup_default = 500  # Wie R-Code
   iter_sampling_default = 500  # Wie R-Code
   adapt_delta_default = 0.8  # Stan default wie R
   ```

2. **`--paper-mode` Flag** hinzufügen (Zeile ~530):
   ```python
   parser.add_argument(
       "--paper-mode",
       action="store_true",
       help="Use Paper-compatible settings (iter=500+500, adapt_delta=0.8, paper indices)"
   )
   ```

3. **Flag in process_data() übergeben** (Zeile ~435):
   ```python
   model_data = process_data(
       seed=args.seed,
       use_original_indices=args.use_original_indices or args.paper_mode  # ← NEU
   )
   ```

4. **Flag in sample() berücksichtigen** (Zeile ~385):
   ```python
   if args.paper_mode:
       print("🔬 PAPER MODE: Using Paper-compatible MCMC settings")
       iter_warmup = 500
       iter_sampling = 500
       adapt_delta = 0.8
   ```

**Resultat:** `python scripts/main.py --model vdhmm --state 3 --seed 42 --paper-mode` trainiert jetzt Paper-kompatibles Modell!

---

#### **SCHRITT 4: Training-Notebooks anpassen** ⏱️ 10 Min

**Was:** `notebooks/2a-training-vdhmm-cmdstan.ipynb` und `2b-training-hmm-cmdstan.ipynb` auf Paper-Parameter umstellen

**Wie:**
1. **Parameter-Zelle** ändern (z.B. in 2a, Zelle mit `CHAINS`, `ITER_WARMUP`, etc.):
   ```python
   # Training-Konfiguration
   PAPER_MODE = True  # Toggle für Paper-kompatible Settings

   if PAPER_MODE:
       CHAINS = 2
       ITER_WARMUP = 500  # Paper: 500 warmup
       ITER_SAMPLING = 500  # Paper: 500 sampling
       ADAPT_DELTA = 0.8  # Paper: Stan default
       print("🔬 PAPER MODE: Using Paper-compatible settings")
   else:
       CHAINS = 2
       ITER_WARMUP = 1000
       ITER_SAMPLING = 1000
       ADAPT_DELTA = 0.95
       print("⚡ CUSTOM MODE: Using extended sampling for better convergence")
   ```

2. **Vergleichs-Zelle** hinzufügen (am Ende):
   ```python
   # Vergleich mit R-Modell aus Paper
   import pickle

   # Lade Paper R-Modell (falls vorhanden)
   paper_model_path = Path(".claude/reference/Models/vdhmm_3.R")
   if paper_model_path.exists():
       # R-Modell laden (via rpy2 oder manuell extrahiert)
       # TODO: Implementierung
       print("📊 Vergleich mit Paper-Modell:")
       # Zeige Parameter-Vergleich
   else:
       print("⚠️  Paper-Modell nicht gefunden")
   ```

**Resultat:** Notebooks verwenden jetzt Paper-Parameter

---

#### **SCHRITT 5: Testing-Notebook erweitern** ⏱️ 5 Min

**Was:** `testing.ipynb` Diagnose-Zelle um Konfigurations-Check erweitern

**Wie:**
1. **Vor Ihrer bestehenden Diagnose-Zelle** hinzufügen:
   ```python
   # ============================================================================
   # KONFIGURATIONS-CHECK: Wurden Paper-Settings verwendet?
   # ============================================================================

   print("=" * 80)
   print("KONFIGURATIONS-VALIDIERUNG")
   print("=" * 80)
   print()

   # Check 1: Wurden Paper-Indices verwendet?
   # (Kann nur überprüft werden wenn Metadaten gespeichert wurden)
   print("1. Train/Test Split:")
   try:
       from constants import PAPER_TRAIN_INDICES
       # TODO: Vergleich mit tatsächlich verwendeten Indices
       print("   ⚠️  Konnte nicht verifizieren ob Paper-Indices verwendet wurden")
       print("   → Stellen Sie sicher dass Sie Preprocessing mit USE_PAPER_INDICES=True laufen ließen!")
   except:
       print("   ✗ Paper-Indices nicht in constants.py definiert!")
   print()

   # Check 2: MCMC-Parameter aus Modell-Metadaten
   print("2. MCMC-Parameter:")
   try:
       # CmdStanPy speichert Metadaten im fit-Objekt
       metadata = good_vdhmm3.fit.metadata
       num_warmup = metadata.num_warmup
       num_samples = metadata.num_samples

       if num_warmup == 500 and num_samples == 500:
           print(f"   ✅ Paper-kompatibel: warmup={num_warmup}, sampling={num_samples}")
       else:
           print(f"   ⚠️  Nicht Paper-kompatibel: warmup={num_warmup}, sampling={num_samples}")
           print(f"      Paper verwendet: warmup=500, sampling=500")
   except Exception as e:
       print(f"   ⚠️  Konnte MCMC-Parameter nicht auslesen: {e}")
   print()

   print("=" * 80)
   print()
   ```

**Resultat:** Testing-Notebook zeigt jetzt ob Paper-kompatible Settings verwendet wurden

---

#### **SCHRITT 6: Dokumentation erstellen** ⏱️ 5 Min

**Was:** README-Abschnitt für Paper-Replikation

**Wie:**
1. **`README.md`** um Abschnitt erweitern:
   ```markdown
   ## 📄 Paper-Replikation (Naumzik et al., 2022)

   Um die Ergebnisse aus dem Paper exakt zu replizieren:

   ### Voraussetzungen
   1. Extrahiere Original Paper-Indices:
      ```bash
      Rscript scripts/extract_indices.R
      ```

   ### Preprocessing
   1. Öffne `notebooks/1a-processing.ipynb`
   2. Setze `USE_PAPER_INDICES = True` in Zelle `77a25d28`
   3. Führe gesamtes Notebook aus

   ### Training (Option A: Command Line)
   ```bash
   python scripts/main.py --model vdhmm --state 3 --seed 42 --paper-mode
   ```

   ### Training (Option B: Notebook)
   1. Öffne `notebooks/2a-training-vdhmm-cmdstan.ipynb`
   2. Setze `PAPER_MODE = True`
   3. Führe Training-Zellen aus

   ### Vergleich
   1. Öffne `testing.ipynb`
   2. Führe alle Zellen aus
   3. Die Diagnose-Zelle zeigt den Vergleich mit Paper-Werten

   ### Erwartete Ergebnisse
   - Posterior-Parameter sollten **näher** an Paper-Werten liegen
   - Kleine Unterschiede sind normal (Random Seed, Stan-Version, etc.)
   ```

**Resultat:** Dokumentation für Paper-Replikation vorhanden

---

#### **SCHRITT 7: Validierung** ⏱️ 20 Min

**Was:** Test-Lauf mit Paper-Settings und Vergleich

**Wie:**
1. **Preprocessing neu laufen lassen:**
   ```bash
   # 1a-processing.ipynb mit USE_PAPER_INDICES=True ausführen
   ```

2. **Training mit Paper-Mode:**
   ```bash
   python scripts/main.py --model vdhmm --state 3 --seed 42 --paper-mode
   ```

3. **Vergleich in testing.ipynb:**
   ```python
   # testing.ipynb komplett ausführen
   # Diagnose-Zelle zeigt Vergleich mit Paper
   ```

4. **Verbleibende Unterschiede dokumentieren:**
   - Sind die ω-Parameter jetzt näher am Paper?
   - Welche Unterschiede bleiben?
   - Mögliche Gründe: Label Switching, Stan-Version, Random Seed

**Resultat:** Validierung ob Paper-Replikation erfolgreich

---

### ⏱️ Geschätzter Zeitaufwand

| Schritt | Beschreibung | Zeit |
|---------|--------------|------|
| 1 | Original Indices extrahieren | 10 Min |
| 2 | Preprocessing anpassen | 15 Min |
| 3 | Training-Script anpassen | 10 Min |
| 4 | Training-Notebooks anpassen | 10 Min |
| 5 | Testing-Notebook erweitern | 5 Min |
| 6 | Dokumentation | 5 Min |
| 7 | Validierung | 20 Min |
| **TOTAL** | | **75 Min (~1.5h)** |

---

### ✅ Erwartetes Ergebnis nach Umsetzung

Nach Abschluss aller Schritte:

1. ✅ **Exakt gleiche Train/Test/Calibration Splits** wie Paper
2. ✅ **Gleiche MCMC-Parameter** (500+500 iter, adapt_delta=0.8)
3. ✅ **Direkt vergleichbare Ergebnisse** mit Paper
4. ✅ **Beide Modi verfügbar:** Paper-Modus & Eigener Modus (mit Toggle)
5. ✅ **Vollständige Dokumentation** aller Unterschiede
6. ✅ **Validierung** ob Paper-Settings verwendet wurden

**Verbleibende mögliche Unterschiede:**
- **Label Switching:** States könnten anders zugeordnet sein → Vergleich über `state_emission`
- **Stan-Version:** R verwendet RStan, Python CmdStanPy → minimal unterschiedliche Compiler
- **Random Seed:** R-Modell hatte vermutlich keinen expliziten Seed → leichte Variation
- **Standardisierung:** Beide verwenden standardisierte Durations → ω-Skala anders als Paper-Text

---

## 7. WICHTIGE HINWEISE

### 7.1 R-Indices sind 1-basiert!

**⚠️ WICHTIG:** R verwendet 1-basierte Indices, Python 0-basiert!

```r
# R: Indices von 1 bis N
train.idx <- c(1, 5, 10, ...)  # 1-basiert
```

```python
# Python: Indices von 0 bis N-1
train_indices = [0, 4, 9, ...]  # 0-basiert!

# Konvertierung:
python_indices = [r_idx - 1 for r_idx in r_indices]
```

**Konsequenz:** Bei der Extraktion aus `indices.Rdata` immer `-1` rechnen!

---

### 7.2 Kalibrierung vs. Test Set

**Paper-Terminologie ist unklar!**

Im R-Code:
- `train.idx` = Training Set (500 Samples)
- `test.idx` = **Kalibrierungs-Set** für Cutoff-Bestimmung (100 Samples)
- `eval.idx` = **Evaluierungs-Set** für finale Metriken (321 Samples)

In Ihrem Python-Code:
- `train_indices` = Training (500)
- `calibration_indices` = Kalibrierung (200 → **FEHLER**, sollte 100 sein!)
- `eval_indices` = Evaluierung (221 → **FEHLER**, sollte 321 sein!)

**Lösung:** Verwenden Sie Paper-Indices direkt!

---

### 7.3 Stan-Version und Compiler

**R-Code verwendet:** RStan (R-Interface zu Stan)
**Ihr Code verwendet:** CmdStanPy (Python-Interface zu CmdStan)

**Unterschiede:**
- **Compiler:** Gleicher Stan-Compiler (falls gleiche Version)
- **Interface:** Unterschiedlich, aber keine inhaltliche Auswirkung
- **Syntax:** Neuere CmdStan-Versionen benötigen neue Array-Syntax (`array[]` statt `[]`)

**Konsequenz:** Minimal - beide verwenden gleichen Stan-Compiler (bei gleicher Version)

---

### 7.4 Reproduzierbarkeit

**R-Modell (Paper):**
- Kein expliziter Seed → **nicht reproduzierbar**
- `init=0` → Parameter bei 0 initialisiert, aber Random Seed trotzdem zufällig

**Ihr Python-Modell:**
- Expliziter Seed → **reproduzierbar**
- Bei gleichem Seed: **identische Ergebnisse**

**Konsequenz:**
- Paper-Ergebnisse sind **nicht exakt reproduzierbar** (kein Seed)
- Sie können nur **ähnliche** Ergebnisse erwarten
- Kleine Abweichungen sind **normal**!

---

## 8. REFERENZEN

### 8.1 Paper

**Naumzik, C., Feuerriegel, S., & Weinmann, M. (2022).** I Will Survive: Predicting Business Failures from Customer Ratings. *Marketing Science*, 41(1), 188-207.

**Wichtige Tabellen/Gleichungen:**
- **Table 3 (S. 198):** State-Dependent Rating Distribution
- **Table 4 (S. 199):** Relationship Between Latent States and Business Failure (ω-Parameter)
- **Equation 2 (S. 193):** State-Rating Emission (Ordered Probit)
- **Equation 3 (S. 194):** State-Failure Emission (Logit mit Durations)
- **Equation 5 (S. 194):** Variable-Duration Transition

### 8.2 R-Code (Paper)

**Pfad:** `.claude/reference/Code/`

- `0_preprocessing.R` - Sentiment-Analyse, Checkins
- `1_processing.R` - Feature Engineering, QR-Dekomposition, **Train/Test Split**
- `2_train_stan_models.R` - Training-Loop, **MCMC-Parameter**
- `3_analysis.R` - Alle Analysen und Tabellen
- `config.R` - Konfiguration
- `helper_functions.R` - Hilfsfunktionen

**Stan-Code:** `.claude/reference/Code/StanCode/`
- `vdhmm.stan` - Variable-Duration HMM
- `hmm.stan` - Standard HMM

**Trainierte Modelle:** `.claude/reference/Models/`
- `vdhmm_3.R` - VD-HMM mit S=3 (Ihr Vergleichsmodell!)

### 8.3 Ihre Python-Implementierung

- `notebooks/1a-processing.ipynb` - Preprocessing
- `notebooks/2a-training-vdhmm-cmdstan.ipynb` - VD-HMM Training
- `notebooks/2b-training-hmm-cmdstan.ipynb` - HMM Training
- `scripts/main.py` - Command-line Training
- `helpers/helper_functions.py` - prepare_stan_data()
- `helpers/model_data.py` - ModelData Klasse
- `data/stan_code/vdhmm.stan` - VD-HMM (neue Syntax)
- `constants.py` - Konstanten

---

## 9. KONTAKT UND FRAGEN

Bei Fragen zur Umsetzung des Plans:
1. **Priorität:** Schritt 1-2 sind kritisch (Paper-Indices!)
2. **Optional:** Schritte 5-6 (Dokumentation)
3. **Validierung:** Schritt 7 zeigt Erfolg

**Nächste Schritte:**
1. R-Script `extract_indices.R` erstellen
2. Indices extrahieren
3. Preprocessing mit Paper-Indices neu laufen lassen
4. Training mit `--paper-mode`
5. Vergleich in `testing.ipynb`

---

**Erstellt am:** 2025-12-10
**Letzte Aktualisierung:** 2025-12-10
