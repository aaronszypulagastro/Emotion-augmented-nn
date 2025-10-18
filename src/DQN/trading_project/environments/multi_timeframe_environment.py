"""
Multi-Timeframe Trading Environment
Kombiniert verschiedene Zeithorizonte für robustere Trading-Entscheidungen
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
import ccxt
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class MultiTimeframeEnvironment(gym.Env):
    """
    Multi-Timeframe Trading Environment
    Kombiniert 5min, 15min und 1h Daten für bessere Trading-Entscheidungen
    """
    
    def __init__(self, 
                 symbol: str,
                 timeframes: List[str] = ['5m', '15m', '1h'],
                 initial_capital: float = 10000.0,
                 max_position_size: float = 0.1,
                 transaction_cost: float = 0.001,
                 lookback_window: int = 20):
        
        super().__init__()
        
        self.symbol = symbol
        self.timeframes = timeframes
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        
        # Portfolio State
        self.capital = initial_capital
        self.position = 0.0
        self.position_value = 0.0
        self.portfolio_value = initial_capital
        self.trades = []
        self.current_step = 0
        
        # Multi-Timeframe Data
        self.data = {}
        self.current_prices = {}
        self.price_histories = {}
        
        # Load data for all timeframes
        self._load_multi_timeframe_data()
        
        # Determine primary timeframe (shortest for step-by-step trading)
        self.primary_timeframe = min(timeframes, key=lambda x: self._timeframe_to_minutes(x))
        
        # Action and Observation Space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # State: [price_norm, volume_norm, rsi, macd, bb_position, position, portfolio_return] * timeframes
        state_size_per_timeframe = 7
        total_state_size = state_size_per_timeframe * len(timeframes)
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_state_size,), dtype=np.float32
        )
        
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Konvertiere Timeframe zu Minuten"""
        timeframe_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        return timeframe_map.get(timeframe, 5)
    
    def _load_multi_timeframe_data(self):
        """Lade Daten für alle Timeframes"""
        
        for timeframe in self.timeframes:
            try:
                if self.symbol in ['BTC/USD', 'ETH/USD']:
                    # Crypto Data
                    data = self._load_crypto_data(timeframe)
                else:
                    # Stock Data
                    data = self._load_stock_data(timeframe)
                
                if data is not None and len(data) > self.lookback_window:
                    self.data[timeframe] = data
                    self.current_prices[timeframe] = 0.0
                    self.price_histories[timeframe] = []
                    print(f"✅ {timeframe} Daten für {self.symbol} geladen: {len(data)} Punkte")
                else:
                    print(f"⚠️ Nicht genug {timeframe} Daten für {self.symbol}")
                    
            except Exception as e:
                print(f"❌ Fehler beim Laden von {timeframe} Daten für {self.symbol}: {e}")
                # Fallback: Generiere synthetische Daten
                self.data[timeframe] = self._generate_synthetic_data(timeframe)
                self.current_prices[timeframe] = 0.0
                self.price_histories[timeframe] = []
    
    def _load_stock_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """Lade Aktien-Daten für spezifischen Timeframe"""
        try:
            ticker = yf.Ticker(self.symbol)
            
            # Bestimme Period basierend auf Timeframe
            if timeframe == '5m':
                period = '5d'
            elif timeframe == '15m':
                period = '15d'
            elif timeframe == '1h':
                period = '60d'
            else:
                period = '30d'
            
            data = ticker.history(period=period, interval=timeframe)
            
            if len(data) > 0:
                return data.reset_index()
            else:
                return None
                
        except Exception as e:
            print(f"Fehler beim Laden von {self.symbol} ({timeframe}): {e}")
            return None
    
    def _load_crypto_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """Lade Crypto-Daten für spezifischen Timeframe"""
        try:
            exchange = ccxt.binance()
            
            # Lade OHLCV Daten
            ohlcv = exchange.fetch_ohlcv(self.symbol, timeframe, limit=1000)
            
            if len(ohlcv) > 0:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df.reset_index()
            else:
                return None
                
        except Exception as e:
            print(f"Fehler beim Laden von Crypto {self.symbol} ({timeframe}): {e}")
            return None
    
    def _generate_synthetic_data(self, timeframe: str) -> pd.DataFrame:
        """Generiere synthetische Daten als Fallback"""
        print(f"⚠️ Generiere synthetische {timeframe} Daten für {self.symbol}")
        
        np.random.seed(42)
        n_points = 1000
        
        # Simuliere Preis-Bewegungen
        returns = np.random.normal(0, 0.02, n_points)
        prices = [100.0]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Erstelle DataFrame
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_points, freq=timeframe),
            'open': prices[:-1],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'close': prices[1:],
            'volume': np.random.uniform(1000, 10000, n_points)
        })
        
        return data
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Berechne technische Indikatoren für einen Timeframe"""
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
    
    def _get_timeframe_state(self, timeframe: str) -> np.ndarray:
        """Erstelle State-Vektor für einen spezifischen Timeframe"""
        
        if timeframe not in self.data:
            return np.zeros(7, dtype=np.float32)
        
        data = self.data[timeframe]
        
        if self.current_step < self.lookback_window:
            data_slice = data.iloc[:self.current_step + 1]
        else:
            data_slice = data.iloc[self.current_step - self.lookback_window + 1:self.current_step + 1]
        
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
        
        # State Vector für diesen Timeframe
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
    
    def _get_state(self) -> np.ndarray:
        """Erstelle kombinierten State-Vektor aus allen Timeframes"""
        
        combined_state = []
        
        for timeframe in self.timeframes:
            timeframe_state = self._get_timeframe_state(timeframe)
            combined_state.extend(timeframe_state)
        
        return np.array(combined_state, dtype=np.float32)
    
    def _execute_trade(self, action: float) -> float:
        """Führe Trade aus basierend auf Primary Timeframe"""
        
        primary_data = self.data[self.primary_timeframe]
        
        if self.current_step >= len(primary_data) - 1:
            return 0.0
        
        current_price = primary_data['close'].iloc[self.current_step]
        next_price = primary_data['close'].iloc[self.current_step + 1]
        
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
            'reward': reward,
            'timeframe': self.primary_timeframe
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
        primary_data = self.data[self.primary_timeframe]
        done = self.current_step >= len(primary_data) - 1
        
        # Hole neuen State
        state = self._get_state()
        
        # Info Dictionary
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'current_price': primary_data['close'].iloc[self.current_step] if self.current_step < len(primary_data) else 0,
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'num_trades': len(self.trades),
            'primary_timeframe': self.primary_timeframe,
            'timeframes': self.timeframes
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
        info = {
            'portfolio_value': self.portfolio_value,
            'primary_timeframe': self.primary_timeframe,
            'timeframes': self.timeframes
        }
        
        return state, info
    
    def render(self, mode: str = 'human'):
        """Render Environment"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Position: {self.position:.3f}")
            print(f"Total Return: {((self.portfolio_value - self.initial_capital) / self.initial_capital * 100):.2f}%")
            print(f"Trades: {len(self.trades)}")
            print(f"Primary Timeframe: {self.primary_timeframe}")
            print(f"All Timeframes: {self.timeframes}")
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
            'win_rate': len([r for r in returns if r > 0]) / len(returns),
            'timeframes_used': self.timeframes,
            'primary_timeframe': self.primary_timeframe
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
    
    def get_timeframe_analysis(self) -> Dict:
        """Analysiere Performance pro Timeframe"""
        analysis = {}
        
        for timeframe in self.timeframes:
            if timeframe in self.data:
                data = self.data[timeframe]
                analysis[timeframe] = {
                    'data_points': len(data),
                    'price_range': (data['close'].min(), data['close'].max()),
                    'volatility': data['close'].std() / data['close'].mean(),
                    'volume_avg': data['volume'].mean(),
                    'current_price': data['close'].iloc[-1] if len(data) > 0 else 0
                }
        
        return analysis


def create_multi_timeframe_environments() -> Dict[str, MultiTimeframeEnvironment]:
    """Erstelle verschiedene Multi-Timeframe Trading Environments"""
    environments = {}
    
    # Aktien mit Multi-Timeframe
    stock_symbols = ['AAPL', 'TSLA']
    stock_timeframes = [['5m', '15m', '1h']]
    
    for symbol in stock_symbols:
        for timeframes in stock_timeframes:
            try:
                env = MultiTimeframeEnvironment(
                    symbol=symbol,
                    timeframes=timeframes,
                    initial_capital=10000.0,
                    max_position_size=0.2
                )
                env_name = f"stock_{symbol}_{'_'.join(timeframes)}"
                environments[env_name] = env
                print(f"✅ {env_name} Environment erstellt")
            except Exception as e:
                print(f"❌ Fehler bei {symbol} ({timeframes}): {e}")
    
    # Crypto mit Multi-Timeframe
    crypto_symbols = ['BTC/USD', 'ETH/USD']
    crypto_timeframes = [['5m', '15m', '1h']]
    
    for symbol in crypto_symbols:
        for timeframes in crypto_timeframes:
            try:
                env = MultiTimeframeEnvironment(
                    symbol=symbol,
                    timeframes=timeframes,
                    initial_capital=10000.0,
                    max_position_size=0.3
                )
                env_name = f"crypto_{symbol.replace('/', '_')}_{'_'.join(timeframes)}"
                environments[env_name] = env
                print(f"✅ {env_name} Environment erstellt")
            except Exception as e:
                print(f"❌ Fehler bei {symbol} ({timeframes}): {e}")
    
    return environments


if __name__ == "__main__":
    # Test das Multi-Timeframe Environment
    print("🚀 Teste Multi-Timeframe Trading Environment...")
    
    # Erstelle Environments
    envs = create_multi_timeframe_environments()
    
    # Teste ein Environment
    if envs:
        env_name = list(envs.keys())[0]
        env = envs[env_name]
        
        print(f"\n📊 Teste {env_name}...")
        
        state, info = env.reset()
        print(f"State Shape: {state.shape}")
        print(f"Timeframes: {info['timeframes']}")
        print(f"Primary Timeframe: {info['primary_timeframe']}")
        
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
        
        # Timeframe Analysis
        analysis = env.get_timeframe_analysis()
        print(f"\n📊 Timeframe Analysis:")
        for timeframe, data in analysis.items():
            print(f"  {timeframe}: {data['data_points']} points, Volatility: {data['volatility']:.4f}")
    
    print("\n✅ Multi-Timeframe Environment Test abgeschlossen!")
