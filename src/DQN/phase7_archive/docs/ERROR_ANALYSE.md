# Error-Analyse - Terminal-Fehler identifiziert

**Datum:** 2025-10-16  
**Kontext:** Phase 7.0 Training mit PSA

---

## 🔍 IDENTIFIZIERTE FEHLER:

### Fehler 1: Import-Fehler (NICHT KRITISCH - bereits behoben)

```
ImportError: attempted relative import beyond top-level package
  File "...\training\agent.py", line 8
    from ..core.emotion_engine import EmotionEngine
```

**Ursache:**
- Python-Import-System Problem beim direkten Ausführen
- Relative Imports funktionieren nur in Paketen

**Status:** ✅ BEHOBEN
- Fallback-Import in agent.py eingefügt (Zeile 12-21)
- Training läuft trotzdem

**Auswirkung:** KEINE - Training funktioniert

---

### Fehler 2: AZPv2 Batch-Size-Mismatch (BEKANNT, NICHT KRITISCH)

```
AZPv2 Training fehlgeschlagen: Expected input batch_size (19) to match target batch_size (17).
```

**Häufigkeit:** Tritt sporadisch auf (z.B. Episode 154, 245, 485)

**Ursache:**
- AdaptiveZonePredictorV2 LSTM-Training
- Batch-Size variiert während Episode
- Nicht alle Batches haben gleiche Größe

**Auswirkung:** 
- ⚠️ **GERING** - Training läuft weiter
- AZPv2 Vorhersage wird für diese Episode übersprungen
- Kein Abbruch des Haupttrainings

**Status:** BEKANNTES PROBLEM aus Phase 6.1
- Erwähnt in PHASE_6_IMPLEMENTATION_LOG.md
- Nicht kritisch, wird behoben in zukünftigen Versionen

---

### Fehler 3: HOHE TD-Errors in späten Episoden (KRITISCH!)

```
Episode 261: TD-Err=111.214
Episode 271: TD-Err=68.611
Episode 471: TD-Err=812.924
Episode 491: TD-Err=1158.684
```

**Entwicklung:**
- Frühe Episoden (1-100): TD-Error ~0.4-2.0 ✅ GUT
- Mittlere Episoden (150-250): TD-Error ~50-200 ⚠️ STEIGEND
- Späte Episoden (400-491): TD-Error ~600-1200 ❌ SEHR HOCH!

**Ursache:**
1. **η (Lernrate) wird zu klein:**
   - Episode 491: eta=0.00048 (extrem niedrig!)
   - Episode 1: eta=0.00200 (Start)
   - **Reduktion um 76%!**

2. **Emotion fällt auf Minimum:**
   - Episode 1: emotion=0.934
   - Episode 491: emotion=0.320 (niedrig)
   - System im "niedrig-emotionalen" Zustand

3. **Feedback-Loop-Problem:**
   - Niedriger Return → Niedrige Emotion → Niedrigeres η
   - Niedriges η → Schlechtes Lernen → Niedrigerer Return
   - **Negativer Verstärkungs-Kreislauf!**

**Auswirkung:** ❌ **SEHR KRITISCH**
- Performance verschlechtert sich (descending trend)
- avg100 fällt: 127.39 (Ep 250) → 11.20 (Ep 500)
- System lernt nicht mehr effektiv

---

## 📊 DATEN-BELEGE:

### Performance-Entwicklung (aus CSV):

```
Episode   Return   avg100   Emotion   η        TD-Error
------------------------------------------------------
1         25.0     -        0.934     0.00200  0.405
91        95.8     17.83    0.980     0.00142  0.955
151       598.8    -        0.960     0.00200  2.390  ← PEAK!
251       12.9     127.39   0.300     0.00004  54.382 ← CRASH
491       11.7     11.20    0.320     0.00048  1158.7 ← ENDE
```

**Klar erkennbar:**
- **Episode 151:** BESTE Performance (598.8 Return!)
- **Episode 251:** CRASH (η fällt auf 0.00004)
- **Episode 491:** KOLLAPS (TD-Error explodiert)

---

## 🎯 ROOT CAUSE (Hauptursache):

### **η-Decay zu aggressiv!**

```python
# In train_finetuning.py (Zeile ~287):
base_eta = 2.5e-3
eta_prop = base_eta * emotion_factor * np.exp(-0.5 * td_error_norm)

# Problem:
# - exp(-0.5 * td_error_norm) dämpft η
# - Bei hohem TD-Error wird η NOCH KLEINER
# - → Führt zu Abwärtsspirale
```

**Feedback-Loop:**
```
Hoher TD-Error
    ↓
exp(-0.5 * td_error) → Kleines η
    ↓
Schlechtes Lernen
    ↓
Niedrigere Returns
    ↓
Niedrigere Emotion
    ↓
NOCH kleineres η
    ↓
EXPLOSION des TD-Errors!
```

---

## 🚀 LÖSUNGSVORSCHLÄGE:

### Lösung 1: η-Bounds erhöhen (SOFORT)

```python
# In train_finetuning.py:
eta_min = 1e-4   # Statt 1e-5
eta_max = 5e-3   # Statt 2.5e-3

# Verhindert, dass η zu klein wird
```

### Lösung 2: Anti-Collapse-Mechanismus (EMPFOHLEN)

```python
# Nach η-Berechnung:
if avg_td_error > 100:  # TD-Error explodiert
    eta = max(eta, 1e-3)  # Force höheres η
    print("[ANTI-COLLAPSE] η erhöht wegen hohem TD-Error")
```

### Lösung 3: PSA-basierte Intervention (PHASE 7!)

```python
# PSA erkennt Collapse:
if psa_trend == 'descending' and psa_stability_score < 0.4:
    # Erhöhe η
    eta *= 2.0
    # Reset Emotion
    emotion = 0.5
    print("[PSA-INTERVENTION] Performance-Crash erkannt!")
```

---

## 📋 NÄCHSTE SCHRITTE (KONKRET):

### SCHRITT 1: η-Bounds anpassen (5 Min)

**Datei:** `training/train_finetuning.py`  
**Zeilen:** ~287-310

**Änderungen:**
```python
# ALT:
base_eta = 2.5e-3
eta = float(np.clip(eta, 1e-5, eta_cap))

# NEU:
base_eta = 3.0e-3  # Höherer Basis-Wert
eta_min_bound = 5e-4  # Höhere Untergrenze
eta = float(np.clip(eta, eta_min_bound, eta_cap))

# Anti-Collapse:
if len(td_error_history) > 0 and np.mean(td_error_history[-10:]) > 100:
    eta = max(eta, 1e-3)  # Force minimum bei hohen TD-Errors
```

---

### SCHRITT 2: PSA-Intervention aktivieren (15 Min)

**Datei:** `training/train_finetuning.py`  
**Zeile:** Nach PSA-Update (~Zeile 642)

```python
# PSA-basierte Intervention
if episode > 50:  # Nach Warm-up
    metrics = psa.compute_stability_metrics()
    
    # Erkenne Performance-Collapse
    if (metrics.trend == 'descending' and 
        metrics.stability_score < 0.4 and 
        episode % 50 == 0):
        
        print(f"\n⚠️  [PSA-INTERVENTION] Performance-Collapse erkannt!")
        print(f"   Stability: {metrics.stability_score:.3f}")
        print(f"   Trend: {metrics.trend}")
        
        # Korrekturmaßnahmen
        eta = max(eta, 2e-3)  # Erhöhe η
        agent.emotion.value = 0.6  # Reset Emotion
        
        print(f"   → η erhöht auf {eta:.5f}")
        print(f"   → Emotion reset auf 0.6\n")
```

---

### SCHRITT 3: Neues Training mit Fixes (2-3 Std)

```bash
# Backup alte Daten
copy results\training_log.csv results\training_log_option_b_without_intervention.csv

# Lösche für neues Training
del results\training_log.csv

# Starte mit Fixes
python training\train_finetuning.py
```

**Erwartung:**
- Kein Collapse bei Episode 250
- TD-Error bleibt unter 50
- avg100 bleibt stabil bei 50-100+

---

## 📊 WAS WIR GELERNT HABEN:

### Option A & B Ergebnis:
```
Beide zeigen GLEICHES Problem:
├─ Episode 1-150:   EXZELLENT (Peak 598.8!)
├─ Episode 150-250: CRASH (η zu klein)
└─ Episode 250-500: KOLLAPS (TD-Error explodiert)
```

### PSA hat erfolgreich:
✅ Anomalien erkannt (23 Stück)  
✅ Trend identifiziert (descending)  
✅ Instabilität gemessen (Score 0.348)  
✅ **ABER:** Noch keine automatische Intervention!

---

## 🎯 AKTIONSPLAN:

### PRIORITÄT 1 (JETZT - 20 Min):
1. ✅ Fehler analysiert (fertig!)
2. [ ] η-Bounds anpassen (Lösung 1)
3. [ ] PSA-Intervention hinzufügen (Lösung 2)

### PRIORITÄT 2 (DANN - 2-3 Std):
4. [ ] Neues Training mit Fixes
5. [ ] Validierung der Lösungen

### PRIORITÄT 3 (DANACH):
6. [ ] Falls erfolgreich: Phase 7.1 (BHO, ACM)
7. [ ] Falls nicht: Weitere Debugging

---

**Problem-Ursache:** ✅ IDENTIFIZIERT  
**Lösung:** ✅ VERFÜGBAR  
**Nächster Schritt:** η-Bounds & PSA-Intervention implementieren


