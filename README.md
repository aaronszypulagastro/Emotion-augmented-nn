# Emotion-Augmented-NN
**Experimentelles Projekt**


KI, die mit Emotion, Motivation und Persönlichkeit lernt: 
Dieses Projekt untersucht, wie neuronale Netze effektiver lernen können, wenn Erfahrungen nicht nur technisch, sondern auch **emotional und motivational** bewertet werden.
Die Idee: Menschen lernen schneller aus starken **Gefühlen** (Freude, Frust, Angst) und sind motiviert durch Wettbewerb oder Zusammenarbeit. Warum also nicht auch **KI?**

**Vision**
- Klassische KI speichert **Erfahrungen gleichwertig** → ineffizient.
- Menschen priorisieren Erinnerungen durch Emotion und Motivation.
- Dieses Projekt implementiert eine **Erweiterung für Reinforcement Learning**, bei der:
  -   Emotionen Erfahrungen verstärken oder abschwächen.
  -   Motivation das Verhalten durch imaginäre Gegner oder Ziele beeinflusst.
  -   Persönlichkeit den Lernstil bestimmt (Winner-Mentality, vorsichtig, kooperativ).
  
**Architektur**
Layers im Modell:
- **Input Layer:** Sensorik (Wahrnehmung)
- **Experience Layer:** Gedächtnis (kurz-/langfristig)
- **Emotion Layer:** Freude, Frust, Angst, Neugier
- **Motivation Layer:** Wettbewerb, Kooperation, Selbstverbesserung
- **Personality Layer:** Winner-Mentality, vorsichtig, kooperativ, risikofreudig
- **Integration Layer:** kombiniert alle Einflüsse
- **Decision Layer:** Policy-Netzwerk
- **Output Layer:** konkrete Aktionen

# Projektplan 
[x]**Phase 1 - Setup** 

- [x] Erste Skizzen, Grundstruktur anlegen

[x]**Phase 2 -  Basis_Agent**

- [x]RL-Agent in CartPole-v1 (PyTorch + Gymnasium) 
- [x]Replay Buffer & Training implementieren

[ ]**Phase 3 - Emotion**

- [ ]Emotion-Tagging für Erfahurngen 
- [ ]Gewichtung im Replay BUffer
- [ ]Vergleich: Klassisch vs Emotion

[ ]**Phase 4 - Motivation**

- [ ]"Imaginärer Gegner" als Baseline 
- [ ]Motivation = Differenz eigener Reward - Gegner-Reward

[ ]**Phase 5 - Persönlichkeit**

- [ ]Parameter für Lernstil (Winner, vorsichtig, kooperativ) 
- [ ]Experimente mit verschiedenen Stilen 

[ ]**Phase 6 - Visualisierung**

- [ ]Lernkurvern vergleichen 
- [ ]Unterschiedliche PErsönlichkeiten plotten 
- [ ]Ergebnisse ins README

# Technologien 
- PyTorch (DeepLEarning)
- Python 3.10+
- Gymnasium (Simulation)
- Matplotlib (Visaulisierung)

# Zukunftsideen 
- Integration in Robotik-Simulationen (z. B. PyBullet, Isaac Gym).
- Multi-Agent-Systeme mit kooperativer/kompetitiver Persönlichkeit.
- Transfer auf reale Roboter.













