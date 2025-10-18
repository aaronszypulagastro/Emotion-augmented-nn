# Phase 7.0 Quick Start Guide

## 🚀 In 3 Schritten starten

### 1️⃣ Installation überprüfen

```bash
# Benötigte Packages
pip install torch numpy scipy gymnasium matplotlib tqdm
```

### 2️⃣ Quick Training starten

```bash
cd src/DQN
python train_phase7.py --episodes 200
```

**Das war's!** Phase 7.0 optimiert automatisch alle Hyperparameter.

### 3️⃣ Ergebnisse prüfen

```bash
# Training Log
cat phase7_training/training_log.csv

# Phase 7 Checkpoints
ls phase7_training/checkpoints/
```

---

## 📊 Was Phase 7.0 macht

```
┌────────────────────────────────────────────────────┐
│  🎯 Bayesian Optimization                          │
│  → Schlägt optimale Hyperparameter vor            │
│                                                    │
│  📈 Stability Analysis                             │
│  → Überwacht Performance-Stabilität               │
│                                                    │
│  ⚙️  Adaptive Configuration                        │
│  → Passt Layer-Weights dynamisch an               │
│                                                    │
│  🔮 Performance Prediction                         │
│  → Sagt Performance voraus                        │
└────────────────────────────────────────────────────┘
```

**Automatisch. Ohne manuelles Tuning.**

---

## 🎮 Kommandos

```bash
# Standard Training (200 Episodes)
python train_phase7.py

# Längeres Training (500 Episodes)
python train_phase7.py --episodes 500

# Baseline-Vergleich (ohne Phase 7)
python train_phase7.py --no-phase7

# Andere Umgebung
python train_phase7.py --env LunarLander-v2

# Benchmark Suite
python phase7_benchmark.py
```

---

## 📈 Erwartete Resultate

### vs. Phase 6.3

| Metrik | Phase 6.3 | Phase 7.0 | Verbesserung |
|--------|-----------|-----------|--------------|
| avg100 | 25.90 | **40-45** | **+50-70%** ✅ |
| Stabilität | Mittel | **Hoch** | **Varianz < 10%** ✅ |

**Ergebnis:** Deutlich bessere und stabilere Performance!

---

## 🔍 Logs verstehen

### training_log.csv

```csv
episode,return,td_error,eta,emotion,bho_iteration,psa_stability_score,...
0,23.0,1.234,0.01,0.402,1,0.0,...
1,28.0,1.102,0.012,0.415,1,0.234,...
...
```

**Wichtige Spalten:**
- `return`: Episode-Reward
- `bho_iteration`: Optimierungs-Iteration
- `psa_stability_score`: Stabilitäts-Score (0-1)
- `acm_system_state`: System-Zustand (exploring/exploiting/stable)
- `mpp_predicted_performance`: Vorhergesagte Performance

---

## 🏆 Beste Hyperparameter abrufen

Nach dem Training:

```python
from core.phase7_integration_manager import Phase7IntegrationManager

manager = Phase7IntegrationManager()
manager.load_checkpoint(prefix="final")

best_params = manager.get_best_hyperparams()
print(f"Best performance: {manager.bho.best_performance:.2f}")
print(f"Best config: {best_params}")
```

---

## ⚡ Performance-Tipps

### Schneller konvergieren
```bash
python train_phase7.py --episodes 300  # Mehr Episoden
```

### Stabilere Ergebnisse
```python
# In train_phase7.py anpassen:
manager = Phase7IntegrationManager(
    optimization_interval=50  # Häufiger optimieren (Standard: 100)
)
```

### Mehr Exploration
```python
# Custom Hyperparameter-Space
from core.bayesian_hyperparameter_optimizer import HyperparameterSpace

space = HyperparameterSpace()
space.eta_max = (0.2, 1.5)  # Breiterer Bereich

manager = Phase7IntegrationManager(hyperparameter_space=space)
```

---

## 🐛 Häufige Probleme

### "CUDA out of memory"
```bash
# MPP auf CPU forcieren
export CUDA_VISIBLE_DEVICES=""
python train_phase7.py
```

### "Module not found"
```bash
# Sicherstellen, dass Sie im richtigen Verzeichnis sind
cd src/DQN
python train_phase7.py
```

### Training sehr langsam
```python
# Kleinere Buffer-Size
manager.mpp.buffer_size = 500  # Standard: 1000
```

---

## 📚 Mehr Informationen

- **Vollständige Docs:** `PHASE_7_README.md`
- **Implementation Log:** `PHASE_7_IMPLEMENTATION_LOG.md`
- **Performance Table:** `PERFORMANCE_COMPARISON_TABLE.md`

---

## 🎉 Das war's!

**Phase 7.0 ist produktionsbereit.**

Einfach starten und die automatische Optimierung arbeiten lassen! 🚀

```bash
python train_phase7.py --episodes 200
```

**Viel Erfolg!** ✨

