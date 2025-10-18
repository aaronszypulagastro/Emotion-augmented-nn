# Vanilla DQN Baseline - Systematischer Debugging-Plan

**Datum:** 2025-10-16  
**Ziel:** Finde die Root Cause des Training-Collapse durch schrittweises Testing

---

## 🎯 STRATEGIE: Bottom-Up Approach

```
Stufe 1: Vanilla DQN (NICHTS extra)
   ↓
Stufe 2: + Emotion-Engine (simpel)
   ↓
Stufe 3: + SelfRegulationController
   ↓
Stufe 4: + EPRU
   ↓
Stufe 5: + AZPv2
   
Teste jede Stufe einzeln!
Bei welcher Stufe tritt der Collapse auf?
→ Dort ist das Problem!
```

---

## 📋 STUFE 1: VANILLA DQN BASELINE

### Ziel:
Teste ob **Basis-DQN** ohne jegliche Emotion-Features stabil läuft

### Änderungen in `training/train_finetuning.py`:

```python
CONFIG = {
    'env_name': 'CartPole-v1',
    'episodes': 500,
    'batch_size': 64,
    'gamma': 0.99,
    'lr': 5e-4,
    'target_update_freq': 10,     # Häufiger (war 25)
    'epsilon_start': 1.0,
    'epsilon_end': 0.1,            # Höher (war 0.05)
    'epsilon_decay': 0.995,
    'buffer_capacity': 50000,      # Kleiner (war 200000)
    'emotion_enabled': False       # ← DEAKTIVIERT!
}

# DEAKTIVIERE:
# - EmotionEngine
# - SelfRegulationController
# - EPRU
# - AZPv2
# - ECL
# - MOO
# - BDH-Plastizität

# NUR BEHALTEN:
# - Basis DQN
# - PSA (für Monitoring)
```

### Erwartetes Ergebnis:

**Falls Vanilla DQN STABIL ist (avg100 > 150):**
```
✅ Problem liegt in Emotion-Features
→ Nächste Stufe: Füge Emotion-Engine hinzu
```

**Falls Vanilla DQN auch COLLAPSED (avg100 < 50):**
```
❌ Problem liegt in Basis-DQN-Hyperparametern
→ Tune: lr, gamma, target_update_freq, buffer_size
```

---

## 🔧 KONKRETE IMPLEMENTIERUNG:

### Erstelle: `training/train_vanilla_dqn.py`

Vereinfachte Version OHNE:
- ❌ Emotion-Engine
- ❌ BDH-Plastizität
- ❌ Alle Phase 6 Features
- ✅ NUR: Standard DQN + PSA Monitoring

**Zeitaufwand:** 30 Min Vorbereitung + 2 Std Training

---

## ⏰ ZEITPLAN:

```
JETZT:     Vanilla DQN Config erstellen (30 Min)
+30 Min:   Training starten
+2.5 Std:  Training fertig
+3 Std:    ERGEBNIS analysieren

Falls stabil:
+3 Std:    Emotion hinzufügen
+5.5 Std:  Training
+6 Std:    Vergleich

Falls nicht:
+3 Std:    Hyperparameter tunen
```

---

## 📊 ERFOLGS-KRITERIEN:

### Vanilla DQN sollte erreichen:
```
CartPole-v1 Standard-Benchmarks:
├─ avg100: > 150 (mindestens)
├─ avg100: > 200 (gut)
├─ avg100: > 300 (sehr gut)
└─ TD-Error: < 2.0

Falls erreicht: ✅ Basis funktioniert
Falls nicht: ❌ Grundlegende Hyperparameter-Probleme
```

---

## 🚀 SOFORTIGER NÄCHSTER SCHRITT:

**Ich erstelle jetzt:**
1. `training/train_vanilla_dqn.py` - Vereinfachtes Training
2. Minimale Konfiguration (nur DQN Essentials)
3. PSA für Monitoring

**Möchten Sie, dass ich das jetzt implementiere?** 

**Zeitaufwand:** 30 Minuten Vorbereitung, dann 2 Stunden Training

**Erwartung:** Klare Antwort ob Problem in Features oder Basis-DQN liegt


