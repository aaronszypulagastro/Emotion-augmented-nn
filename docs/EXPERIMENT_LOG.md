# Experiment Log - Emotion-Augmented DQN

**Dieses Dokument protokolliert die täglichen Änderungen, Experimnente und Ergebnisse des Projekts.*
## 2025-10-01
**ZIel:** Basis-DQN implementieren & erstes Training starten 
**Änderungen:** 
- DQN Agent mit Replay Buffer und Epsilon-Greedy Strategie erstellt
- REplay BUffer Kapazität: 50.000
- Learning Rate: 1e-4
- Batch Size: 64
- Target NEtwork alle 100 Schritte aktualisiert (hard update)

**Ergebnisse:**
- Training über 2500 Episoden
- Durchschnitts-Score schwankte stark zwischen 20-100
- Modell sehr instabil, keine konsistente Lösung von CartPOle

  <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/0b9746ef-508b-407f-b495-1bd2412c4d1b" />

  <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f2174e36-1460-4c2e-98d9-23536c99a9ae" />

**Notizen:**
- Erste Tests zeigen, dass Replay Buffer zu klein sein könnte
- Exploration zu langsam abgebaut - Agent bleibt lange "zufällig"

**---**

## 2025-10-02
**Ziel:** Erste Stabilisierung des Agenten 
**Änderungen:**
- Replay Buffer Kapazität auf 100.000 gesetzt
- Batch Size von 64 auf 128 erhöht
- LEarning Rate von 1e-4 zu 5e-4
- Soft Target-Update ('tau=0.01') eingeführt
- Mehrfache Updates pro Schritt ('updates_per_step=2)
- HUber-Loss statt MSELoss
- GRadient Clipping ('max_norm=10')
- AdamW Optimizer statt Adam














  

