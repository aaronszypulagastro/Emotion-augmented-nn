# Phase 6.1 + PSA Integration - Konfiguration

**Ziel:** Kombiniere funktionierende Phase 6.1 (40.05 avg100) mit validiertem PSA  
**Erwartung:** Stabile 40+ Performance mit Real-time Monitoring  
**Zeitaufwand:** 1-2 Stunden

---

## 🎯 STRATEGIE:

```
Phase 6.1 (funktionierte):
✅ avg100: 40.05
✅ EPRU + AZPv2 aktiv
✅ Stabile Performance

+ Phase 7.0 PSA (validiert):
✅ Real-time Monitoring
✅ Anomalie-Detection
✅ Trend-Erkennung

= Beste Kombination!
```

---

## 🔧 ÄNDERUNGEN NOTWENDIG:

### RÜCKGÄNGIG machen (Phase 7.0 Fixes entfernen):

**Problem:** Die Phase 7.0 η-Fixes haben NICHT geholfen  
**Lösung:** Zurück zu ursprünglichen Werten

**Datei:** `training/train_finetuning.py`

#### Änderung 1: base_eta zurücksetzen (Zeile 295)
```python
# AKTUELL (Phase 7.0):
base_eta = 3.5e-3

# ZURÜCK ZU (Phase 6.1):
base_eta = 2.5e-3  # Original-Wert
```

#### Änderung 2: Dämpfung zurücksetzen (Zeile 300)
```python
# AKTUELL (Phase 7.0):
eta_prop = base_eta * emotion_factor * np.exp(-0.3 * td_error_norm)

# ZURÜCK ZU (Phase 6.1):
eta_prop = base_eta * emotion_factor * np.exp(-0.5 * td_error_norm)  # Original
```

#### Änderung 3: η-Bounds zurücksetzen (Zeile 321-322)
```python
# AKTUELL (Phase 7.0):
eta_min_bound = 5e-4
eta = float(np.clip(eta, eta_min_bound, 7e-3))

# ZURÜCK ZU (Phase 6.1):
eta = float(np.clip(eta, 1e-5, 7e-3))  # Original-Bounds
```

#### Änderung 4: Anti-Collapse ENTFERNEN (Zeile 324-327)
```python
# LÖSCHEN (funktionierte nicht):
# Phase 7.0 Anti-Collapse Mechanismus
# if len(td_error_history) > 10 and np.mean(td_error_history[-10:]) > 100:
#     eta = max(eta, 1e-3)
#     print(f"[ANTI-COLLAPSE] η erhöht auf {eta:.5f}")
```

---

### BEHALTEN (PSA funktioniert):

**Datei:** `training/train_finetuning.py`

#### PSA Import (Zeile 40) - ✅ BEHALTEN
```python
from core.performance_stability_analyzer import PerformanceStabilityAnalyzer
```

#### PSA Initialisierung (Zeile 233-239) - ✅ BEHALTEN
```python
psa = PerformanceStabilityAnalyzer(
    window_size=100,
    anomaly_threshold=3.0,
    trend_threshold=0.3
)
```

#### PSA Update & Reports (Zeile 632-665) - ✅ BEHALTEN
```python
psa.update(episode, total_reward)

if episode % 50 == 0 and episode > 0:
    metrics = psa.compute_stability_metrics()
    print(f"\n📊 [PSA] Stability Report...")
```

#### CSV PSA-Spalten (Zeile 656, 693-710) - ✅ BEHALTEN

---

## 📋 SCHRITT-FÜR-SCHRITT ANLEITUNG:

### Schritt 1: Fixes rückgängig machen

Öffne `training/train_finetuning.py` und:

1. **Zeile 295:** `base_eta = 3.5e-3` → `base_eta = 2.5e-3`
2. **Zeile 300:** `np.exp(-0.3 * ...)` → `np.exp(-0.5 * ...)`
3. **Zeile 321-322:** Entferne `eta_min_bound`, nutze `1e-5`
4. **Zeile 324-327:** Lösche Anti-Collapse Block (kommentiere aus)

### Schritt 2: PSA behalten

- ✅ Alles PSA-bezogene BEHALTEN (funktioniert!)
- ✅ Import, Initialisierung, Updates, CSV

### Schritt 3: PSA-Intervention VEREINFACHEN

**Zeile 654-665:** Ändere PSA-Intervention

```python
# AKTUELL (zu komplex):
if (metrics.trend == 'descending' and 
    metrics.stability_score < 0.4 and 
    episode > 100):
    agent.emotion.value = max(agent.emotion.value, 0.6)

# NEU (nur Monitoring, keine Intervention):
# Entferne Intervention - nur Monitoring
# PSA sollte nur beobachten, nicht eingreifen
```

---

## 🎯 ERWARTETES ERGEBNIS:

```
Mit Phase 6.1 Config + PSA Monitoring:
├─ avg100: 35-45 (wie Phase 6.1) ✅
├─ TD-Error: < 2.0 ✅
├─ PSA Reports: Alle 50 Episoden ✅
└─ Kein Collapse ✅
```

---

## ⏰ ZEITPLAN:

```
JETZT: Code-Änderungen (10 Min)
+10 Min: Training starten
+2-3 Std: Training läuft
+3 Std: ERGEBNIS - hoffentlich stabile 40+ avg100!
```

---

**Bereit die Änderungen umzusetzen?** ✅


