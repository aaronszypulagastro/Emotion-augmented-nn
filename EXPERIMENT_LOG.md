# Experiment Log - Emotion-Augmented DQN

**Dieses Dokument protokolliert die täglichen Änderungen, Experimnente und Ergebnisse des Projekts.*
## 2025-10-01
## **ZIel:** 
Basis-DQN implementieren & erstes Training starten 
**Änderungen:** 
- DQN Agent mit Replay Buffer und Epsilon-Greedy Strategie erstellt
- REplay BUffer Kapazität: 50.000
- Learning Rate: 1e-4
- Batch Size: 64
- Target NEtwork alle 100 Schritte aktualisiert (hard update)

## **Ergebnisse:**
- Training über 2500 Episoden
- Durchschnitts-Score schwankte stark zwischen 20-100
- Modell sehr instabil, keine konsistente Lösung von CartPOle

  <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/0b9746ef-508b-407f-b495-1bd2412c4d1b" />

  <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f2174e36-1460-4c2e-98d9-23536c99a9ae" />

## **Notizen:**
- Erste Tests zeigen, dass Replay Buffer zu klein sein könnte
- Exploration zu langsam abgebaut - Agent bleibt lange "zufällig"

**---**

## 2025-10-02
## **Ziel:** 
Erste Stabilisierung des Agenten 
**Änderungen:**
- Replay Buffer Kapazität auf 100.000 gesetzt
- Batch Size von 64 auf 128 erhöht
- LEarning Rate von 1e-4 zu 5e-4
- Soft Target-Update ('tau=0.01') eingeführt
- Mehrfache Updates pro Schritt ('updates_per_step=2)
- HUber-Loss statt MSELoss
- GRadient Clipping ('max_norm=10')
- AdamW Optimizer statt Adam

 ## **Training** 
 
 <img width="1920" height="1080" alt="Screenshot (599)" src="https://github.com/user-attachments/assets/5f50693a-576e-47de-b028-ab6b00c00b61" />


 - 600 Episoden abgeschlossen
 - Bestleistung: 'avg100 = 425' um Episode 300

   
<img width="964" height="800" alt="Screenshot (597)" src="https://github.com/user-attachments/assets/f227ecb8-1f38-4234-8ec4-575ee4358fb1" />


- Danach Leitungsabfall -> vermutlich **Overfitting** oder zu aggressiver ## **Lernraten-/Epsilon-Decay**
- Finale Phase: Modell stabilisiert sich wieder auf 500 Punkte


## **Evaluation** 
- Modell über 10 Episoden getestet -> Score konstant **500.0**


<img width="919" height="800" alt="Screenshot (612)" src="https://github.com/user-attachments/assets/700443e1-466e-45b2-8010-99d044d4e8b7" />


-------------------------------------------------------------------------------------------------
- [x] **BASELINE ABGESCHLOSSEN:** CartPole-DQN funktioniert stabil -> als Referenzmodell behalten
-------------------------------------------------------------------------------------------------
      
- **Langfristig:**
    - Vergleich verschiedener Modelle (CNN vs Transformer)
    - Kombination mti Reinforcement LEarning -> Emotion als zusätzlicher INputfaktor
    - Dokumentation fortführen (Experimente mit Logs + Plots wie heute)

## 2025-10-03
## **Ziel:**
Vergleich zweier DQN-LÄufe (mit und ohne Emotionseinfluss) zur Untersuchung der Auswirkungen emotionaler Modulation auf Lernstabilität, Explorationsverhalten und Leistung. 

**---**

## Versuchsaufbau
- **Baseling:** Standard-DQN (ohne Emotionseinfluss)
- **EMotion-Version:** DQN mit integriertem Emotionsfaktor (EmotionENginge v1)
- **Umgebung:** 'CartPole-v1'
- **Parameter:**ε_eff)
    - ε_decay = 0.999
    - emotion_range = 0.5 - 0.9
    - reward_clipping = aktiv
    - Replay-Buffer: 50k
    - Batchsize: 64

**Zielmetriken:** Return, gleitender Durchschnitt, Emotion_Level, Exploration (ε_eff)

**---**
## Beobachtung 
bild einfügen 

- Der emotionale Agent zeigt ausgeprägtere Explorationsphasen und reagiert adaptiver auf kurzfristige Leistungsschwankungen
- Emotionale Peaks korrelieren mit temporären Leistungsanstiegen
- Der Baseline-Agent bleibt stabiler, verliert aber an Flexibilität im späteren Verlauf  
- Gesamtergebnis: Emotionseinfluss erzeugt höhere Varianz, gleichzeitig jedoch mehr Lernimpulse in stagnierenden Phasen

## Interpretation 
- Emotionale Modulation kann Stagnation im Training reduzieren und Exploration länger aufrechterhalten
- Allerdings erfordert sie ein Feintuning der Kopplung (zu starke Emotion = Überexploration)
- Emotion wirkt als dynamischer Verstärker, aber kein konstanter Performance-Garant
- Potenzial für „EmotionEngine v2“: adaptive Lernrate + gewichtete Emotion-Einbindung  
 









  

