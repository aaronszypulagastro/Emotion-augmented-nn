# 🚀 Emotion-Augmented Trading Agent

**Revolutionäres Trading-System mit emotionaler Intelligenz für Aktien und Crypto**

## 🎯 Überblick

Dieses Projekt implementiert einen **Emotion-Augmented Trading Agent**, der Rainbow DQN mit einer speziellen Trading Emotion Engine kombiniert. Das System nutzt emotionale Intelligenz, um bessere Trading-Entscheidungen zu treffen und sich an verschiedene Marktbedingungen anzupassen.

## 🧠 Kernkonzepte

### **Emotion-Augmented Neural Networks**
- **8 Trading-Emotionen**: Confident, Cautious, Frustrated, Greedy, Fearful, Optimistic, Pessimistic, Neutral
- **Market Sentiment Analysis**: Trend, Volatility, Momentum, Volume
- **Performance-based Emotion Transitions**: Emotionen passen sich an Trading-Performance an
- **Risk Management**: Emotion-basierte Risikotoleranz und Position Sizing

### **Rainbow DQN Integration**
- **Dueling Network Architecture**: Separate Value und Advantage Streams
- **Prioritized Experience Replay**: Wichtige Experiences werden häufiger gelernt
- **Double DQN**: Reduziert Overestimation Bias
- **Continuous Action Space**: Kontinuierliche Position Sizing

## 📁 Projektstruktur

```
trading_project/
├── environments/
│   └── trading_environment.py      # Trading Environment für Aktien + Crypto
├── agents/
│   ├── trading_emotion_engine.py   # Trading-spezifische Emotion Engine
│   └── emotion_trading_agent.py    # Emotion-Augmented Trading Agent
├── results/                        # Training Results & Analysis
├── train_trading_agent.py          # Training Script
└── README.md                       # Diese Datei
```

## 🚀 Features

### **Multi-Asset Support**
- **Aktien**: AAPL, TSLA, MSFT, GOOGL (via Yahoo Finance)
- **Crypto**: BTC/USD, ETH/USD (via Binance API)
- **Synthetic Data**: Fallback für Offline-Testing

### **Multi-Timeframe Support**
- **5-Minuten**: Für Crypto (hohe Volatilität)
- **15-Minuten**: Für Aktien (stabile Trends)
- **1-Stunde**: Für langfristige Strategien

### **Technische Indikatoren**
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatilitäts-Indikator
- **Volume Analysis**: Volumen-basierte Signale

### **Emotion-Engine Features**
- **Real-time Emotion Updates**: Basierend auf Performance und Marktbedingungen
- **Risk Tolerance Adaptation**: Emotion-basierte Risikobewertung
- **Position Sizing**: Emotion-basierte Position-Größen
- **Market Sentiment Integration**: Trend, Volatilität, Momentum, Volume

## 🎯 Trading-Emotionen

| Emotion | Beschreibung | Risk Modifier | Position Sizing |
|---------|--------------|---------------|-----------------|
| **CONFIDENT** | Gute Performance, steigende Gewinne | 1.2x | 1.3x |
| **CAUTIOUS** | Volatile Märkte, hohe Risiken | 0.7x | 0.6x |
| **FRUSTRATED** | Verluste, schlechte Performance | 0.5x | 0.3x |
| **GREEDY** | Zu viele Gewinne, Risiko-Überschätzung | 1.5x | 1.6x |
| **FEARFUL** | Panik-Verkäufe, Risiko-Unterbewertung | 0.3x | 0.2x |
| **OPTIMISTIC** | Positive Marktausblick | 1.1x | 1.1x |
| **PESSIMISTIC** | Negative Marktausblick | 0.6x | 0.5x |
| **NEUTRAL** | Ausgewogene Marktlage | 1.0x | 1.0x |

## 📊 Performance Metriken

### **Trading Metriken**
- **Total Return**: Gesamt-Rendite
- **Win Rate**: Gewinnrate der Trades
- **Sharpe Ratio**: Risk-Adjusted Returns
- **Maximum Drawdown**: Maximaler Verlust
- **Average Profit per Trade**: Durchschnittlicher Gewinn pro Trade

### **Emotion Metriken**
- **Emotion Stability**: Stabilität der Emotionen
- **Emotion-Performance Correlation**: Korrelation zwischen Emotionen und Performance
- **Risk Adaptation**: Anpassung der Risikotoleranz
- **Market Sentiment Accuracy**: Genauigkeit der Markt-Sentiment-Analyse

## 🚀 Installation & Setup

### **Dependencies**
```bash
pip install torch numpy pandas matplotlib yfinance ccxt gymnasium
```

### **Quick Start**
```python
# Schneller Test
python train_trading_agent.py quick

# Vollständige Experimente
python train_trading_agent.py
```

## 📈 Verwendung

### **1. Environment erstellen**
```python
from environments.trading_environment import TradingEnvironment

env = TradingEnvironment(
    symbol='AAPL',
    timeframe='15m',
    initial_capital=10000.0,
    max_position_size=0.2
)
```

### **2. Agent trainieren**
```python
from agents.emotion_trading_agent import train_emotion_trading_agent

metrics = train_emotion_trading_agent(
    env=env,
    episodes=1000,
    save_interval=100
)
```

### **3. Ergebnisse analysieren**
```python
# Automatische Analyse und Visualisierung
analyze_results(results)
```

## 🎯 Erwartete Verbesserungen

### **Vs. Traditionelle Trading-Systeme**
- **+20-30% Risk-Adjusted Returns** durch emotionale Anpassung
- **-40% Drawdowns** durch besseres Risikomanagement
- **+50% Sharpe Ratio** durch emotionale Diversifikation
- **+25% Win Rate** durch emotionale Markt-Erkennung

### **Vs. Standard DQN**
- **+15% Performance** durch emotionale Kontext-Information
- **+30% Stability** durch emotion-basierte Risikokontrolle
- **+20% Adaptability** durch emotionale Markt-Anpassung

## 🔬 Wissenschaftliche Grundlagen

### **Neuroscience**
- **Amygdala**: Emotion Engine (Gefühle)
- **Prefrontal Cortex**: Attention Mechanisms (Fokus)
- **Cerebellum**: Self-Correction (Fehlerkorrektur)
- **Flow State**: Flow Rewards (optimale Performance)

### **Psychology**
- **Emotional Intelligence**: EQ-basierte Entscheidungen
- **Risk Perception**: Emotion-basierte Risikobewertung
- **Market Psychology**: Herdenverhalten und Sentiment

### **Computer Science**
- **Reinforcement Learning**: Q-Learning mit Emotionen
- **Neural Networks**: Dueling DQN Architecture
- **Experience Replay**: Prioritized Learning

## 🚨 Risikomanagement

### **Technische Sicherheiten**
- **Stopp-Loss**: Automatische Verlustbegrenzung
- **Position Sizing**: Emotion-basierte Position-Größen
- **Diversifikation**: Über verschiedene Märkte
- **Backup-Systeme**: Für System-Ausfälle

### **Emotionale Sicherheiten**
- **Emotion-Thresholds**: Für extreme Zustände
- **Cooling-Off Periods**: Nach großen Verlusten
- **Performance-Limits**: Für emotionale Anpassungen
- **Human Override**: Für kritische Situationen

## 📊 Ergebnisse

### **Test-Environments**
- **AAPL**: Apple Inc. (Aktien)
- **TSLA**: Tesla Inc. (Aktien)
- **BTC/USD**: Bitcoin (Crypto)
- **ETH/USD**: Ethereum (Crypto)

### **Performance-Vergleich**
- **Baseline DQN**: Standard Deep Q-Network
- **Emotion-Augmented**: Unser System
- **Improvement**: Erwartete Verbesserungen

## 🔮 Zukunft

### **Phase 2: Multi-Timeframe**
- **Multi-Timeframe Analysis**: Verschiedene Zeithorizonte
- **Timeframe Fusion**: Kombination von Timeframes
- **Adaptive Timeframes**: Dynamische Timeframe-Anpassung

### **Phase 3: Advanced Features**
- **News Sentiment**: Integration von News-Analyse
- **Social Media**: Twitter/Reddit Sentiment
- **Economic Indicators**: Makroökonomische Daten
- **Real-time Trading**: Live-Trading Integration

## 🤝 Contributing

### **Development**
1. Fork das Repository
2. Erstelle einen Feature Branch
3. Implementiere deine Änderungen
4. Teste gründlich
5. Erstelle einen Pull Request

### **Testing**
- **Unit Tests**: Für alle Komponenten
- **Integration Tests**: Für das gesamte System
- **Performance Tests**: Für verschiedene Märkte
- **Stress Tests**: Für extreme Marktbedingungen

## 📄 Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe LICENSE-Datei für Details.

## 🙏 Danksagungen

- **OpenAI**: Für die Inspiration zu emotionaler KI
- **PyTorch**: Für das Deep Learning Framework
- **Gymnasium**: Für die RL-Environments
- **Yahoo Finance**: Für Aktien-Daten
- **Binance**: Für Crypto-Daten

---

**🚀 Entwickle die Zukunft des emotionalen Trading!** 

**Kontakt**: [Deine Kontaktinformationen]

**Version**: 1.0.0

**Letzte Aktualisierung**: Oktober 2024
