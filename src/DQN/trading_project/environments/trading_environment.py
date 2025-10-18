"""
Trading Environment für Emotion-Augmented Neural Networks
Unterstützt Aktien und Crypto mit Multi-Timeframe Support
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
import ccxt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TradingEnvironment(gym.Env):
    """
    Trading Environment für Aktien und Crypto
    State: [price, volume, indicators, sentiment, portfolio_state]
    Actions: [buy, sell, hold] mit Position Sizing
    """
    
    def __init__(self, 
                 symbol: str,
                 timeframe: str = '5m',
                 initial_capital: float = 10000.0,
                 max_position_size: float = 0.1,
                 transaction_cost: float = 0.001,
                 lookback_window: int = 20):
        
        super().__init__()
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        
        # Portfolio State
        self.capital = initial_capital
        self.position = 0.0  # -1 to 1 (short to long)
        self.position_value = 0.0
        self.portfolio_value = initial_capital
        self.trades = []
        self.current_step = 0
        
        # Data
        self.data = None
        self.current_price = 0.0
        self.price_history = []
        
        # Action and Observation Space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # State: [price_norm, volume_norm, rsi, macd, bb_position, position, portfolio_return]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Lade historische Daten für das Symbol"""
        try:
            if self.symbol in ['BTC/USD', 'ETH/USD']:
                # Crypto Data
                self._load_crypto_data()
            else:
                # Stock Data
                self._load_stock_data()
                
            if self.data is None or len(self.data) < self.lookback_window:
                raise ValueError(f"Nicht genug Daten für {self.symbol}")
                
        except Exception as e:
            print(f"Fehler beim Laden der Daten für {self.symbol}: {e}")
            # Fallback: Generiere synthetische Daten
            self._generate_synthetic_data()
    
    def _load_stock_data(self):
        """Lade Aktien-Daten von Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            
            # Bestimme Period basierend auf Timeframe
            if self.timeframe == '5m':
                period = '5d'
            elif self.timeframe == '15m':
                period = '15d'
            elif self.timeframe == '1h':
                period = '60d'
            else:
                period = '30d'
            
            data = ticker.history(period=period, interval=self.timeframe)
            
            if len(data) > 0:
                self.data = data.reset_index()
                print(f"✅ {len(self.data)} Datenpunkte für {self.symbol} geladen")
            else:
                raise ValueError("Keine Daten erhalten")
                
        except Exception as e:
            print(f"Fehler beim Laden von {self.symbol}: {e}")
            self.data = None
    
    def _load_crypto_data(self):
        """Lade Crypto-Daten von Binance"""
        try:
            exchange = ccxt.binance()
            
            # Konvertiere Timeframe
            timeframe_map = {
                '5m': '5m',
                '15m': '15m', 
                '1h': '1h'
            }
            
            tf = timeframe_map.get(self.timeframe, '5m')
            
            # Lade OHLCV Daten
            ohlcv = exchange.fetch_ohlcv(self.symbol, tf, limit=1000)
            
            if len(ohlcv) > 0:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                self.data = df.reset_index()
                print(f"✅ {len(self.data)} Crypto-Datenpunkte für {self.symbol} geladen")
            else:
                raise ValueError("Keine Crypto-Daten erhalten")
                
        except Exception as e:
            print(f"Fehler beim Laden von Crypto {self.symbol}: {e}")
            self.data = None
    
    def _generate_synthetic_data(self):
        """Generiere synthetische Daten als Fallback"""
        print(f"⚠️ Generiere synthetische Daten für {self.symbol}")
        
        np.random.seed(42)
        n_points = 1000
        
        # Simuliere Preis-Bewegungen
        returns = np.random.normal(0, 0.02, n_points)
        prices = [100.0]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Erstelle DataFrame
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='5min'),
            'open': prices[:-1],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'close': prices[1:],
            'volume': np.random.uniform(1000, 10000, n_points)
        })
        
        self.data = data
        print(f"✅ {len(self.data)} synthetische Datenpunkte generiert")
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Berechne technische Indikatoren"""
        indicators = {}
        
        if len(data) < 20:
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'bb_position': 0.0,
                'volume_ma': data['volume'].mean() if len(data) > 0 else 1000
            }
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        
        # MACD
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        indicators['macd'] = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0.0
        
        # Bollinger Bands Position
        sma20 = data['close'].rolling(window=20).mean()
        std20 = data['close'].rolling(window=20).std()
        bb_upper = sma20 + (std20 * 2)
        bb_lower = sma20 - (std20 * 2)
        
        current_price = data['close'].iloc[-1]
        bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
        if bb_range > 0:
            indicators['bb_position'] = (current_price - bb_lower.iloc[-1]) / bb_range
        else:
            indicators['bb_position'] = 0.5
        
        # Volume MA
        indicators['volume_ma'] = data['volume'].rolling(window=20).mean().iloc[-1]
        
        return indicators
    
    def _get_state(self) -> np.ndarray:
        """Erstelle State-Vektor"""
        if self.current_step < self.lookback_window:
            # Verwende verfügbare Daten
            data_slice = self.data.iloc[:self.current_step + 1]
        else:
            # Verwende Lookback Window
            data_slice = self.data.iloc[self.current_step - self.lookback_window + 1:self.current_step + 1]
        
        if len(data_slice) == 0:
            return np.zeros(7, dtype=np.float32)
        
        # Berechne Indikatoren
        indicators = self._calculate_indicators(data_slice)
        
        # Normalisiere Preis
        current_price = data_slice['close'].iloc[-1]
        price_mean = data_slice['close'].mean()
        price_std = data_slice['close'].std()
        price_norm = (current_price - price_mean) / price_std if price_std > 0 else 0.0
        
        # Normalisiere Volume
        current_volume = data_slice['volume'].iloc[-1]
        volume_norm = (current_volume - indicators['volume_ma']) / indicators['volume_ma'] if indicators['volume_ma'] > 0 else 0.0
        
        # Portfolio Return
        portfolio_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        # State Vector
        state = np.array([
            price_norm,                    # Normalisierter Preis
            volume_norm,                   # Normalisiertes Volume
            (indicators['rsi'] - 50) / 50, # RSI normalisiert (-1 to 1)
            np.tanh(indicators['macd']),   # MACD normalisiert
            indicators['bb_position'] * 2 - 1,  # BB Position (-1 to 1)
            self.position,                 # Aktuelle Position (-1 to 1)
            portfolio_return               # Portfolio Return
        ], dtype=np.float32)
        
        return state
    
    def _execute_trade(self, action: float) -> float:
        """Führe Trade aus und berechne Reward"""
        if self.current_step >= len(self.data) - 1:
            return 0.0
        
        current_price = self.data['close'].iloc[self.current_step]
        next_price = self.data['close'].iloc[self.current_step + 1]
        
        # Bestimme neue Position basierend auf Action
        new_position = np.clip(action, -self.max_position_size, self.max_position_size)
        position_change = new_position - self.position
        
        # Berechne Transaktionskosten
        transaction_cost = abs(position_change) * self.transaction_cost * self.portfolio_value
        
        # Aktualisiere Position
        self.position = new_position
        self.position_value = self.position * self.portfolio_value
        
        # Berechne Portfolio Performance
        price_change = (next_price - current_price) / current_price
        position_return = self.position * price_change
        
        # Berechne neues Portfolio Value
        self.portfolio_value = self.portfolio_value * (1 + position_return) - transaction_cost
        
        # Berechne Reward
        reward = position_return - transaction_cost / self.portfolio_value
        
        # Speichere Trade
        self.trades.append({
            'step': self.current_step,
            'action': action,
            'position': self.position,
            'price': current_price,
            'next_price': next_price,
            'return': position_return,
            'portfolio_value': self.portfolio_value,
            'reward': reward
        })
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Environment Step"""
        action_value = float(action[0])
        
        # Führe Trade aus
        reward = self._execute_trade(action_value)
        
        # Nächster Schritt
        self.current_step += 1
        
        # Prüfe ob Episode beendet
        done = self.current_step >= len(self.data) - 1
        
        # Hole neuen State
        state = self._get_state()
        
        # Info Dictionary
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'current_price': self.data['close'].iloc[self.current_step] if self.current_step < len(self.data) else 0,
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'num_trades': len(self.trades)
        }
        
        return state, reward, done, False, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset Environment"""
        super().reset(seed=seed)
        
        # Reset Portfolio
        self.capital = self.initial_capital
        self.position = 0.0
        self.position_value = 0.0
        self.portfolio_value = self.initial_capital
        self.trades = []
        self.current_step = 0
        
        # Hole initialen State
        state = self._get_state()
        info = {'portfolio_value': self.portfolio_value}
        
        return state, info
    
    def render(self, mode: str = 'human'):
        """Render Environment"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Position: {self.position:.3f}")
            print(f"Total Return: {((self.portfolio_value - self.initial_capital) / self.initial_capital * 100):.2f}%")
            print(f"Trades: {len(self.trades)}")
            print("-" * 40)
    
    def get_performance_metrics(self) -> Dict:
        """Berechne Performance Metriken"""
        if len(self.trades) == 0:
            return {}
        
        returns = [trade['return'] for trade in self.trades]
        
        metrics = {
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'num_trades': len(self.trades),
            'avg_return': np.mean(returns),
            'volatility': np.std(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'win_rate': len([r for r in returns if r > 0]) / len(returns)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Berechne Maximum Drawdown"""
        if len(self.trades) == 0:
            return 0.0
        
        portfolio_values = [self.initial_capital] + [trade['portfolio_value'] for trade in self.trades]
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd


def create_trading_environments() -> Dict[str, TradingEnvironment]:
    """Erstelle verschiedene Trading Environments"""
    environments = {}
    
    # Aktien
    stock_symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL']
    for symbol in stock_symbols:
        try:
            env = TradingEnvironment(
                symbol=symbol,
                timeframe='15m',
                initial_capital=10000.0,
                max_position_size=0.2
            )
            environments[f"stock_{symbol}"] = env
            print(f"✅ {symbol} Environment erstellt")
        except Exception as e:
            print(f"❌ Fehler bei {symbol}: {e}")
    
    # Crypto
    crypto_symbols = ['BTC/USD', 'ETH/USD']
    for symbol in crypto_symbols:
        try:
            env = TradingEnvironment(
                symbol=symbol,
                timeframe='5m',
                initial_capital=10000.0,
                max_position_size=0.3
            )
            environments[f"crypto_{symbol.replace('/', '_')}"] = env
            print(f"✅ {symbol} Environment erstellt")
        except Exception as e:
            print(f"❌ Fehler bei {symbol}: {e}")
    
    return environments


if __name__ == "__main__":
    # Test das Trading Environment
    print("🚀 Teste Trading Environment...")
    
    # Erstelle Environments
    envs = create_trading_environments()
    
    # Teste ein Environment
    if envs:
        env_name = list(envs.keys())[0]
        env = envs[env_name]
        
        print(f"\n📊 Teste {env_name}...")
        
        state, info = env.reset()
        print(f"Initial State Shape: {state.shape}")
        print(f"Initial Portfolio: ${info['portfolio_value']:.2f}")
        
        # Simuliere einige Schritte
        for step in range(10):
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            
            print(f"Step {step}: Action={action[0]:.3f}, Reward={reward:.4f}, Portfolio=${info['portfolio_value']:.2f}")
            
            if done:
                break
        
        # Performance Metrics
        metrics = env.get_performance_metrics()
        print(f"\n📈 Performance Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    print("\n✅ Trading Environment Test abgeschlossen!")
