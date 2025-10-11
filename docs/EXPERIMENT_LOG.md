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

# 2025-10-10 
# **Aktueller Stand des Emotion DQN Projekts:** 
**1. Modellstruktur:**
- Q-Network besteht aus zwei PlasticLInear-Layern (BDH-Style) und einem linearen Output
  
- Jede Schicht besitzt eine σ-Matrix, die sich emotional moduliert anpasst:
        **σij​←decay⋅σij​+η⋅mod⋅(tanh(posti​)⋅prej​)**

- Diese σ-Aktivität wird im Training nach jeder Episode resettet → stabiler, aber evtl. zu kurzlebig.

**2. Emotion-Engine:**
- Hat einen Zustand state ∈ [0, 1], der:
    - durch Rewards (TD-Error) im Training (update()) schnell reagiert,
    - nach jeder Episode (update_after_episode()) eine langsamere Trendanpassung bekommt.

- Unterstützt Winner/Loser-Mentality, NOise, Floor/Ceil-Grenzen

- Der emotionale Zustand beeinflusst:
    - Reward-Shaping
    - Plastizität (mod-Gain)
    - Exploration (ε-Shift)

  **3. Trainingslogik:**
  - Umgebung: CartPole-v1
  - Reward wird um Emotion modifiziert (reward *= (0.8 + 0.4 * emotion)
  - Nach jeder Episode: EmotionUpdate, BDH-Reset
  - PLots erzeugen:
    - emotion_vs_reward
    - emotion_reward_correlation
    - emotion_bdh_dynamics

# Verbesserungspotenziale
~~**1. Emotion-PLastizitäts-Kopplung:**~~
Der Modulator (mod = agent._emotion_gain()) beeinflusst σ, aber immer mit fixem η=1

**Idee: dynamisches η basierend auf Emotion**

Ausagbe: Bei positiver EMotion höhere Lernrate (Hebbian Boost) , bei negativer weniger

~~**2. Emotion-Decay adaptiv**~~
Die Emotion bleibt akuell noch lange hoch, was Reward-Intstabilität erzeugen kann.

**Idee: self.state() ändern, Emotion dämpft sich stärker bei hohem TD-Error (unsicherem Lernen)**

~~**3. σ-Reset weicher gestalten:**~~
Aktuell wird nach jeder Episode sigma.zero() aufgerufen - das komplette Vergessen.

**Idee: partielles Resetting**

Ausgabe: Erinnerung bleibt teilweise bestehen, biologisch realistischer

~~**4. Hyperparameter Sweep:**~~
Die aktuell fixierten Parameter könnten besser abgestimmt werden.

~~**5. Visualisierung erweitern**~~
  
**AUFGABEN ABGESCHLOSSEN*

# 2025/10/11
# **ZIEL: Vorbereitung auf nächste Modell-Iteration**

# Langfristiges Ziel: 
Entwicklung eiens emotional-modulierten Lernagenten, der später auf Robotics-Umgebungen übertragen werden kann. 

# **Aktuelleer Stand**

- Emotion-BDH-DQN läuft stabil in CartPole-v1
  
- EmotionEngine liefert sinnvolle dynamische Werte (Winner/Loser-Effekte, Boosts)

- σ-Plastizität reagiert auf Emotion (mod-Faktor) und η-Anpassung

- Logging & Visualisierung über plot_utils.py zeigen klare Korrelationen zwischen EMotion, TD_Error und       Reward

# **Geplante Verbesserungen**
**Adaptive η-Regulierung (Emotion ↔ TD-Error)**

- Lernrate η soll abhängig von akuteller Emotion und TD_Error werden.
    -> Ziel: feinfühligere Anpassung an gute oder schlechte Lernphasen.
  
- **Formel-Ansatz:**
  
    **η=η0​⋅(0.3+E)⋅(1−TD_Error/TD_max)*

- **Erwarteter Effekt:** stabileres Lernen, weniger Overreaction bei hohen Fehlern.



**BDH-σ-Homeostase**

- Neue Kontrollschleife zur Vermeidung von σ-Explosion oder -Verlust

- **ZIEL:** konstante Plastizität über längere Trainingsphasen


**Transfer zu realistischeren Umgebungen**

- Nach Stabilisierung: Test auf **MountainCar, Pendulum, oder PyBullet-Reacher*
- Vorbereitung auf Robotics-Integration (Sensor-INput, Motor-Output)

# Erwartete Forschungsfragen

- Wie stark beeinflusst Emotuion die effektive Exploration (ε-Modulation)?
- Wie verändert sich die σ-Dynamik bei adaptivem η?
- Kann der Agent durch emotionale Rückkopplung robustere Policies entwickeln?

**Nächste Aktionen**

- [x] ~~Implementierung adaptiver η-Funktion in train_finetuning.py~~

- [x] ~~Hinzufügen von σ-Norm-Logging (Homeostase-Überwachung)~~

- [x] ~~Vorbereitung von Transfer-Tests (MountainCar-v0)~~

- [x] ~~Vergleich der Lernkurven (TD-Error / Emotion / σ-Norm)~~











  

