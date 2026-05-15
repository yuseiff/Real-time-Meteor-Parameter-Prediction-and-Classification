# 🌠 A Flash of Insight: Real-Time Meteor Shower Classification with Machine Learning

A research-grade machine learning project for classifying meteor showers in real time using only initial geocentric and topocentric observational data — bypassing the need for computationally expensive full orbital calculations.

> **Associated Paper:** *"A Flash of Insight: Real-Time Meteor Shower Classification with Machine Learning"*  
> Youssef Maaod & Mohamed Abd Elaziz — Galala University, Faculty of Computer Science and Engineering

---

## 📖 Overview

Traditionally, associating an observed meteor with its parent shower requires computing its full heliocentric orbit (semi-major axis, eccentricity, inclination, etc.) — a process that is computationally intensive and unsuitable for real-time data triage. This project proposes a machine learning alternative: using only the **early-stage geocentric parameters** that are computed much earlier in the processing pipeline, we train classifiers to directly predict the shower association.

Four model architectures are benchmarked on a dataset of over **2 million meteor events** from the **Global Meteor Network (GMN)**:

| Model | Training Time (s) | Accuracy | F1-Score (weighted) |
|---|---|---|---|
| **XGBoost** | 35.67 | **98.36%** | **0.9839** |
| LightGBM | 25.29 | 97.73% | 0.9775 |
| TabTransformer | 7396.93 | 97.44% | 0.9753 |
| Logistic Regression | 120.09 | 86.97% | 0.8277 |

---

## ✨ Key Features

- **Real-time capable classification** — uses only 6 early observational features, not full orbital elements
- **Multi-model benchmark** — Logistic Regression, XGBoost, LightGBM, and a custom PyTorch TabTransformer
- **Physically interpretable results** — feature importance analysis confirms the model learned real meteor physics
- **Large-scale dataset** — trained on 2,055,506 cleaned meteor records across 11 classes
- **Exploratory data analysis** — scatter plots and radiant maps visually validate feature discriminability

---

## 🗂️ Repository Structure

```
├── main.ipynb                  # Full pipeline: EDA, preprocessing, all models
├── main_epoch100.ipynb         # TabTransformer trained for 100 epochs
├── main_epoch340.ipynb         # TabTransformer trained for 340 epochs (converged)
├── bar_graph.ipynb             # Comparative bar chart visualizations of model results
├── A_Flash_of_Insight_Real-Time_Meteor_Shower.pdf   # Full research paper (IEEE format)
├── A_Flash_of_Insight_Real-Time_Meteor_Shower.docx  # Editable version of the paper
└── A_FLAS_1.PPT                # Project presentation slides
```

---

## 📊 Dataset

**Source:** [Global Meteor Network (GMN)](https://globalmeteornetwork.org/) — `traj_summary_all.txt`

- **Raw records:** 4,111,325 meteor detections
- **After cleaning:** 2,055,506 samples
- **Format:** semicolon-delimited (`;`) text file with `#`-prefixed comment lines
- **Classes:** 11 total — 10 most frequent named showers + `Sporadic` background

### Named Showers Modeled

| Code | Shower | Peak Period | Parent Body |
|---|---|---|---|
| PER | Perseids | August | Comet Swift-Tuttle |
| GEM | Geminids | December | Asteroid 3200 Phaethon |
| QUA | Quadrantids | January | Asteroid 2003 EH1 |
| ORI | Orionids | October | Comet 1P/Halley |
| SDA | S. Delta Aquariids | July–August | — |
| STA | S. Taurids | October–November | Comet 2P/Encke |
| ETA | Eta Aquariids | May | Comet 1P/Halley |
| LEO | Leonids | November | Comet 55P/Tempel-Tuttle |
| HYD | Alpha Hydrids | — | — |
| CAP | Capricornids | July–August | — |
| — | Sporadic | Year-round | No specific parent |

---

## 🔬 Methodology

### 1. Feature Selection

Only **6 early-stage observational features** are used as model inputs — no heliocentric orbital elements:

| Feature | Description |
|---|---|
| `sol_lon` | Solar Longitude (°) — proxy for time of year |
| `rageo` | Geocentric Right Ascension (°) — radiant position |
| `decgeo` | Geocentric Declination (°) — radiant position |
| `vgeo` | Geocentric Velocity (km/s) |
| `azim` | Local Azimuth (°) |
| `elev` | Local Elevation (°) |

**Target:** `iau_code` — the IAU three-letter shower identifier (e.g., `PER`, `GEM`) or `Sporadic`

### 2. Preprocessing Pipeline

1. Parse the semicolon-delimited GMN file with `pandas.read_csv`
2. Rename `'...'` entries in `iau_code` to `'Sporadic'`
3. Convert feature columns to numeric; drop rows with `NaN`
4. Filter to the 11 most frequent classes
5. Encode labels with `sklearn.preprocessing.LabelEncoder`
6. Stratified 80/20 train/test split (`stratify=y`)
7. Feature scaling with `StandardScaler` (fit on train only — no data leakage)

### 3. Models

**Logistic Regression** — linear baseline that quantifies the non-linearity of the problem.

**XGBoost** — gradient boosted decision trees; highest accuracy overall. Configured with:
- `objective='multi:softmax'`, `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`

**LightGBM** — faster GBDT alternative with default configuration. Trains in ~70% of XGBoost's time.

**TabTransformer (PyTorch)** — custom deep learning architecture for tabular data:
- 2 Transformer encoder layers, `d_model=64`, 4 attention heads
- Architecture: Linear embedding → Transformer encoder → global average pooling → MLP classifier
- Trained with Adam optimizer (`lr=0.001`), CrossEntropyLoss, for **340 epochs** to convergence

---

## 📈 Results

### Model Comparison

GBDT models (XGBoost, LightGBM) achieved **>97% accuracy** versus **~87%** for Logistic Regression, proving the fundamentally non-linear nature of the classification problem. The fully-trained TabTransformer reached competitive accuracy (97.44%) but required ~2 hours of training vs. under 1 minute for GBDTs.

### Feature Importance

The XGBoost and LightGBM models independently ranked features as follows:

1. `decgeo` — Geocentric Declination *(most important)*
2. `rageo` — Geocentric Right Ascension
3. `vgeo` — Geocentric Velocity
4. `sol_lon` — Solar Longitude
5. `elev` — Local Elevation
6. `azim` — Local Azimuth *(least important)*

This directly reflects known meteor physics: showers are defined by their **radiant point** on the sky, their **entry velocity**, and the **time of year** Earth intersects their debris stream. The local observer-dependent coordinates (`azim`, `elev`) are the least physically meaningful.

### Confusion Matrix (XGBoost)

The confusion matrix shows a strong diagonal. The dominant misclassification pattern is shower members being labeled as `Sporadic` — likely caused by physical stream outliers displaced by gravitational perturbations (e.g., from Jupiter) or measurement errors in atmospheric triangulation. The reverse error (a sporadic mislabeled as a shower member) is extremely rare, confirming the model's robustness against false positives.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm torch tqdm
```

### Dataset

Download the GMN trajectory summary file (`traj_summary_all.txt`) from the [Global Meteor Network](https://globalmeteornetwork.org/data/) and place it one directory above the notebook:

```
parent_directory/
├── traj_summary_all.txt   ← dataset goes here
└── repo/
    └── main.ipynb
```

### Running the Notebook

```bash
jupyter notebook main.ipynb
```

The notebook is organized into phases — run cells sequentially:

| Phase | Description |
|---|---|
| Phase 1 | Data loading and parsing |
| Phase 2 | Data cleaning and feature selection |
| Phase 3 | Exploratory Data Analysis with visualizations |
| Phase 4 | Model training and evaluation (XGBoost, LightGBM, Logistic Regression) |
| TabTransformer | Deep learning model training and evaluation |

For TabTransformer convergence experiments:
- `main_epoch100.ipynb` — early training checkpoint (100 epochs)
- `main_epoch340.ipynb` — fully converged model (340 epochs)

---

## 🔭 Discussion & Implications

**Why GBDTs outperform Logistic Regression:** The relationship between geocentric parameters and shower membership is fundamentally non-linear. GBDTs learn hierarchical decision rules such as *"IF sol_lon ≈ 140° AND vgeo ≈ 60 km/s THEN likely Perseid"* — boundaries a linear model cannot express.

**LightGBM for production:** LightGBM matches near-top accuracy in ~70% of XGBoost's training time, making it the preferred choice for latency-sensitive real-time deployment on telescope data streams.

**Physics validation:** The feature importance rankings are not arbitrary — they directly reflect the physical definition of meteor showers. This gives high confidence that the model has learned a physically meaningful representation rather than fitting to noise.

---

## ⚠️ Limitations

- The study models only the **11 most frequent classes**; performance on rare showers with few training examples is not evaluated.
- Misclassifications into `Sporadic` can result from physical stream outliers or measurement error — not solely model failure.
- The TabTransformer requires ~2 hours to train; not suitable for rapid retraining in operational environments without GPU acceleration.
- Default hyperparameters were used for GBDT models — tuning could further improve performance.

---

## 🔮 Future Work

- **All-class scaling** — Extend to the full 397-class problem, potentially grouping rare showers into an `Other` category.
- **Hyperparameter tuning** — Systematic Bayesian or grid search for GBDT and Transformer hyperparameters.
- **Advanced deep learning** — Benchmark against FT-Transformer, TabSTAR, and other tabular-specific architectures.
- **Synthetic data generation** — Use feature importance insights to generate physically realistic synthetic meteor datasets.

---

## 📚 References

1. Jenniskens, P. (2006). *Meteor showers and their parent comets.* Cambridge University Press.
2. Vida, D., et al. (2021). The Global Meteor Network — methodology and first results. *MNRAS*, 506(4), 5046–5074.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*, pp. 785–794.
4. Huang, X., et al. (2020). TabTransformer: Tabular data modeling using contextual embeddings. *arXiv:2012.06678*.
5. Peña-Asensio, E., et al. (2023). Deep machine learning for meteor monitoring. *Planetary and Space Science*, 238, 105802.

See the full paper for a complete reference list: ([Full Paper](https://ieeexplore.ieee.org/document/11440853)).

---

## 👥 Authors

- **Youssef Maaod** — Galala University, AI Science Program ([yhf400294@gu.edu.eg](mailto:yhf400294@gu.edu.eg))
- **Mohamed Abd Elaziz** — Galala University, AI Science Program ([m.elsaied@gu.edu.eg](mailto:m.elsaied@gu.edu.eg))

---

## 📄 License

This project is intended for academic and research purposes. Please cite the associated paper if you use this work.
