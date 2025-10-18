"""
Multi-Timeframe Optimized Trading Test
Kombiniert alle Optimierungen mit Multi-Timeframe Support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import unserer optimierten Komponenten
from optimized_trading_system import OptimizedTradingEmotionEngine, OptimizedPaperTradingSystem, TradingEmotion

class MultiTimeframeMarketSimulator:
    """Multi-Timeframe Market Simulator mit realistischen Daten"""
    
    def __init__(self):
        self.assets = {
            'AAPL': {'base_price': 150.0, 'volatility': 0.02, 'trend': 0.001},
            'TSLA': {'base_price': 200.0, 'volatility': 0.04, 'trend': 0.002},
            'BTC/USD': {'base_price': 45000.0, 'volatility': 0.05, 'trend': 0.003},
            'ETH/USD': {'base_price': 3000.0, 'volatility': 0.06, 'trend': 0.004}
        }
        
        # Multi-Timeframe Daten
        self.timeframes = ['5m', '15m', '1h']
        self.current_prices = {}
        self.price_histories = {}
        
        # Initialisiere für alle Assets und Timeframes
        for symbol in self.assets.keys():
            self.current_prices[symbol] = {}
            self.price_histories[symbol] = {}
            for tf in self.timeframes:
                self.current_prices[symbol][tf] = self.assets[symbol]['base_price']
                self.price_histories[symbol][tf] = [self.assets[symbol]['base_price']]
    
    def generate_multi_timeframe_data(self, steps: int = 200) -> Dict[str, Dict[str, List[Dict]]]:
        """Generiere Multi-Timeframe Marktdaten"""
        
        market_data = {}
        
        for symbol in self.assets.keys():
            market_data[symbol] = {}
            
            for tf in self.timeframes:
                market_data[symbol][tf] = []
                
                # Bestimme Timeframe-spezifische Parameter
                tf_multiplier = {'5m': 1.0, '15m': 0.7, '1h': 0.4}[tf]
                tf_steps = int(steps * tf_multiplier)
                
                for step in range(tf_steps):
                    # Generiere Preis-Bewegung basierend auf Timeframe
                    base_volatility = self.assets[symbol]['volatility']
                    base_trend = self.assets[symbol]['trend']
                    
                    # Timeframe-spezifische Anpassungen
                    if tf == '5m':
                        volatility = base_volatility * 1.2  # Höhere Volatilität
                        trend = base_trend * 0.5
                    elif tf == '15m':
                        volatility = base_volatility * 1.0  # Standard
                        trend = base_trend * 1.0
                    else:  # 1h
                        volatility = base_volatility * 0.8  # Niedrigere Volatilität
                        trend = base_trend * 1.5
                    
                    # Generiere Preis-Bewegung
                    trend_component = trend * np.random.normal(0, 1)
                    volatility_component = volatility * np.random.normal(0, 1)
                    
                    price_change = trend_component + volatility_component
                    
                    # Aktualisiere Preis
                    self.current_prices[symbol][tf] *= (1 + price_change)
                    self.price_histories[symbol][tf].append(self.current_prices[symbol][tf])
                    
                    # Berechne Metriken
                    if len(self.price_histories[symbol][tf]) > 1:
                        returns = np.diff(self.price_histories[symbol][tf][-20:]) / self.price_histories[symbol][tf][-20:-1]
                        volatility_metric = np.std(returns) if len(returns) > 1 else volatility
                        volume = np.random.uniform(1000, 10000)
                    else:
                        volatility_metric = volatility
                        volume = 5000
                    
                    # Speichere Marktdaten
                    market_data[symbol][tf].append({
                        'step': step,
                        'price': self.current_prices[symbol][tf],
                        'price_change': price_change,
                        'volatility': volatility_metric,
                        'volume': volume,
                        'timestamp': step,
                        'timeframe': tf
                    })
        
        return market_data

class MultiTimeframeOptimizedTradingTest:
    """Multi-Timeframe Optimized Trading Test"""
    
    def __init__(self):
        self.market_simulator = MultiTimeframeMarketSimulator()
        self.emotion_engine = OptimizedTradingEmotionEngine()
        self.paper_system = OptimizedPaperTradingSystem(
            initial_capital=10000.0,
            commission=0.001,
            slippage=0.0005,
            min_trade_size=100.0,
            max_position_size=0.3
        )
        
        # Multi-Timeframe Tracking
        self.timeframe_weights = {'5m': 0.3, '15m': 0.4, '1h': 0.3}
        self.timeframe_performance = {'5m': [], '15m': [], '1h': []}
        self.attention_weights_history = []
        
        # Erweiterte Trading-Statistiken
        self.trading_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'emotion_changes': 0,
            'trading_frequency': 0.0,
            'timeframe_adaptation': 0.0
        }
        
        self.initial_emotion = self.emotion_engine.current_emotion
        self.emotion_history = []
    
    def run_multi_timeframe_test(self, steps: int = 200) -> Dict:
        """Führe Multi-Timeframe Trading Test durch"""
        
        print(f"🚀 Starte Multi-Timeframe Optimized Trading Test...")
        print(f"Steps: {steps}")
        print(f"Timeframes: {self.market_simulator.timeframes}")
        print(f"Initial Capital: ${self.paper_system.initial_capital:,.2f}")
        print(f"Max Position Size: {self.paper_system.max_position_size*100:.1f}%")
        
        # Generiere Multi-Timeframe Marktdaten
        print(f"\n📊 Generiere Multi-Timeframe Marktdaten...")
        market_data = self.market_simulator.generate_multi_timeframe_data(steps)
        
        # Führe Multi-Timeframe Trading-Simulation durch
        print(f"\n💹 Starte Multi-Timeframe Trading-Simulation...")
        
        for step in range(steps):
            if step % 40 == 0:
                portfolio_value = self.paper_system.get_portfolio_value(self._get_current_prices())
                print(f"   Step {step}/{steps} - Portfolio: ${portfolio_value:,.2f} - Emotion: {self.emotion_engine.current_emotion.value}")
            
            # Wähle zufälliges Asset für diesen Step
            symbol = np.random.choice(list(self.market_simulator.assets.keys()))
            
            # Berechne Multi-Timeframe Analysis
            timeframe_analysis = self._analyze_multi_timeframe_data(market_data[symbol], step)
            
            # Update Emotion Engine basierend auf Multi-Timeframe Daten
            self._update_emotion_engine_multi_timeframe(timeframe_analysis, symbol)
            
            # Bestimme Trading-Entscheidung mit Multi-Timeframe Support
            trading_decision = self._make_multi_timeframe_trading_decision(timeframe_analysis, symbol)
            
            # Führe Trade aus
            if trading_decision['action'] != 'hold':
                self._execute_multi_timeframe_trade(trading_decision, timeframe_analysis, symbol)
            
            # Speichere Emotion-Historie
            self.emotion_history.append(self.emotion_engine.current_emotion.value)
        
        # Berechne finale Ergebnisse
        final_results = self._calculate_multi_timeframe_results()
        
        print(f"\n✅ Multi-Timeframe Trading Test abgeschlossen!")
        return final_results
    
    def _analyze_multi_timeframe_data(self, symbol_data: Dict[str, List[Dict]], step: int) -> Dict:
        """Analysiere Multi-Timeframe Daten"""
        
        analysis = {}
        
        for tf in self.market_simulator.timeframes:
            if tf in symbol_data and step < len(symbol_data[tf]):
                tf_data = symbol_data[tf][step]
                analysis[tf] = {
                    'price': tf_data['price'],
                    'price_change': tf_data['price_change'],
                    'volatility': tf_data['volatility'],
                    'volume': tf_data['volume'],
                    'step': tf_data['step']
                }
            else:
                # Verwende letzte verfügbare Daten
                if tf in symbol_data and len(symbol_data[tf]) > 0:
                    tf_data = symbol_data[tf][-1]
                    analysis[tf] = {
                        'price': tf_data['price'],
                        'price_change': tf_data['price_change'],
                        'volatility': tf_data['volatility'],
                        'volume': tf_data['volume'],
                        'step': tf_data['step']
                    }
                else:
                    # Fallback-Daten
                    analysis[tf] = {
                        'price': 100.0,
                        'price_change': 0.0,
                        'volatility': 0.02,
                        'volume': 5000,
                        'step': step
                    }
        
        # Berechne gewichtete Metriken
        analysis['weighted_price'] = sum(analysis[tf]['price'] * self.timeframe_weights[tf] for tf in self.market_simulator.timeframes)
        analysis['weighted_price_change'] = sum(analysis[tf]['price_change'] * self.timeframe_weights[tf] for tf in self.market_simulator.timeframes)
        analysis['weighted_volatility'] = sum(analysis[tf]['volatility'] * self.timeframe_weights[tf] for tf in self.market_simulator.timeframes)
        analysis['weighted_volume'] = sum(analysis[tf]['volume'] * self.timeframe_weights[tf] for tf in self.market_simulator.timeframes)
        
        return analysis
    
    def _update_emotion_engine_multi_timeframe(self, timeframe_analysis: Dict, symbol: str):
        """Update Emotion Engine mit Multi-Timeframe Daten"""
        
        # Berechne Portfolio-Performance
        current_portfolio_value = self.paper_system.get_portfolio_value(self._get_current_prices())
        portfolio_return = (current_portfolio_value - self.paper_system.initial_capital) / self.paper_system.initial_capital
        
        # Berechne Trade-Performance
        if self.trading_stats['total_trades'] > 0:
            win_rate = self.trading_stats['winning_trades'] / self.trading_stats['total_trades']
        else:
            win_rate = 0.5
        
        # Update Market Sentiment mit gewichteten Multi-Timeframe Daten
        self.emotion_engine.update_market_sentiment(
            price_change=timeframe_analysis['weighted_price_change'],
            volume_change=timeframe_analysis['weighted_volume'] / 5000 - 1,
            volatility=timeframe_analysis['weighted_volatility'],
            trend_strength=timeframe_analysis['weighted_price_change'] * 2
        )
        
        # Update Performance mit optimierter Engine
        self.emotion_engine.update_performance(
            portfolio_return=portfolio_return,
            trade_return=timeframe_analysis['weighted_price_change'],
            drawdown=self.paper_system.max_drawdown,
            win_rate=win_rate
        )
        
        # Track Emotion Changes
        if self.emotion_engine.current_emotion != self.initial_emotion:
            self.trading_stats['emotion_changes'] += 1
            self.initial_emotion = self.emotion_engine.current_emotion
    
    def _make_multi_timeframe_trading_decision(self, timeframe_analysis: Dict, symbol: str) -> Dict:
        """Multi-Timeframe Trading-Entscheidungen"""
        
        enhanced_risk_tolerance = self.emotion_engine.get_enhanced_risk_tolerance()
        enhanced_position_modifier = self.emotion_engine.get_enhanced_position_sizing_modifier()
        current_emotion = self.emotion_engine.current_emotion
        emotion_intensity = self.emotion_engine.emotion_intensity
        
        # Berechne Timeframe-spezifische Signale
        timeframe_signals = {}
        for tf in self.market_simulator.timeframes:
            tf_data = timeframe_analysis[tf]
            
            # Bestimme Signal basierend auf Timeframe
            if tf == '5m':
                # Kurzfristige Signale
                if tf_data['price_change'] > 0.005:
                    timeframe_signals[tf] = 'buy'
                elif tf_data['price_change'] < -0.005:
                    timeframe_signals[tf] = 'sell'
                else:
                    timeframe_signals[tf] = 'hold'
            elif tf == '15m':
                # Mittelfristige Signale
                if tf_data['price_change'] > 0.003:
                    timeframe_signals[tf] = 'buy'
                elif tf_data['price_change'] < -0.003:
                    timeframe_signals[tf] = 'sell'
                else:
                    timeframe_signals[tf] = 'hold'
            else:  # 1h
                # Langfristige Signale
                if tf_data['price_change'] > 0.001:
                    timeframe_signals[tf] = 'buy'
                elif tf_data['price_change'] < -0.001:
                    timeframe_signals[tf] = 'sell'
                else:
                    timeframe_signals[tf] = 'hold'
        
        # Berechne gewichtete Signale
        signal_weights = {'buy': 0, 'sell': 0, 'hold': 0}
        for tf, signal in timeframe_signals.items():
            signal_weights[signal] += self.timeframe_weights[tf]
        
        # Bestimme finales Signal
        if signal_weights['buy'] > signal_weights['sell'] and signal_weights['buy'] > 0.4:
            base_action = 'buy'
        elif signal_weights['sell'] > signal_weights['buy'] and signal_weights['sell'] > 0.4:
            base_action = 'sell'
        else:
            base_action = 'hold'
        
        # Modifiziere basierend auf Emotion
        trade_probability = {
            TradingEmotion.CONFIDENT: 0.9,
            TradingEmotion.CAUTIOUS: 0.4,
            TradingEmotion.FRUSTRATED: 0.2,
            TradingEmotion.GREEDY: 0.95,
            TradingEmotion.FEARFUL: 0.1,
            TradingEmotion.OPTIMISTIC: 0.7,
            TradingEmotion.PESSIMISTIC: 0.3,
            TradingEmotion.NEUTRAL: 0.5
        }.get(current_emotion, 0.5)
        
        # Modifiziere basierend auf Emotion-Intensität
        trade_probability *= emotion_intensity
        
        # Entscheide ob Trade
        if np.random.random() > trade_probability:
            action = 'hold'
        else:
            action = base_action
        
        # Bestimme Trade-Größe mit Multi-Timeframe Modifikatoren
        if action != 'hold':
            base_quantity = 10.0  # Erhöht für Multi-Timeframe
            quantity = base_quantity * enhanced_position_modifier * enhanced_risk_tolerance
            
            # Multi-Timeframe Volatilitäts-Anpassung
            avg_volatility = timeframe_analysis['weighted_volatility']
            if avg_volatility > 0.05:
                quantity *= 0.5
            elif avg_volatility < 0.02:
                quantity *= 1.8
            
            # Timeframe-Konsistenz Bonus
            if len(set(timeframe_signals.values())) == 1:  # Alle Timeframes zeigen gleiches Signal
                quantity *= 1.5
            
            # Emotion-Intensität Anpassung
            quantity *= emotion_intensity
            
            quantity = max(3.0, quantity)  # Erhöhte Mindest-Trade-Größe
        else:
            quantity = 0
        
        # Berechne Attention Weights
        attention_weights = [self.timeframe_weights[tf] for tf in self.market_simulator.timeframes]
        self.attention_weights_history.append(attention_weights.copy())
        
        return {
            'action': action,
            'quantity': quantity,
            'emotion': current_emotion.value,
            'risk_tolerance': enhanced_risk_tolerance,
            'position_modifier': enhanced_position_modifier,
            'emotion_intensity': emotion_intensity,
            'timeframe_signals': timeframe_signals,
            'signal_weights': signal_weights,
            'attention_weights': attention_weights
        }
    
    def _execute_multi_timeframe_trade(self, decision: Dict, timeframe_analysis: Dict, symbol: str):
        """Führe Multi-Timeframe Trade aus"""
        
        if decision['action'] == 'hold':
            return
        
        # Verwende gewichteten Preis für Trade
        trade_price = timeframe_analysis['weighted_price']
        
        # Führe Trade aus
        result = self.paper_system.execute_trade(
            symbol=symbol,
            action=decision['action'],
            quantity=decision['quantity'],
            price=trade_price
        )
        
        if result['success']:
            self.trading_stats['total_trades'] += 1
            
            # Berechne Trade-Ergebnis
            if decision['action'] == 'sell' and symbol in self.paper_system.positions:
                avg_price = self.paper_system.positions[symbol]['avg_price']
                trade_profit = (trade_price - avg_price) * decision['quantity']
                
                if trade_profit > 0:
                    self.trading_stats['winning_trades'] += 1
                else:
                    self.trading_stats['losing_trades'] += 1
                
                self.trading_stats['total_profit'] += trade_profit
                
                # Track Timeframe Performance
                for tf in self.market_simulator.timeframes:
                    self.timeframe_performance[tf].append(trade_profit * self.timeframe_weights[tf])
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Hole aktuelle Preise für Portfolio-Berechnung"""
        current_prices = {}
        for symbol in self.market_simulator.assets.keys():
            # Verwende gewichteten Preis
            weighted_price = sum(
                self.market_simulator.current_prices[symbol][tf] * self.timeframe_weights[tf]
                for tf in self.market_simulator.timeframes
            )
            current_prices[symbol] = weighted_price
        return current_prices
    
    def _calculate_multi_timeframe_results(self) -> Dict:
        """Berechne Multi-Timeframe Ergebnisse"""
        
        final_portfolio_value = self.paper_system.get_portfolio_value(self._get_current_prices())
        total_return = (final_portfolio_value - self.paper_system.initial_capital) / self.paper_system.initial_capital
        
        win_rate = self.trading_stats['winning_trades'] / max(self.trading_stats['total_trades'], 1)
        
        # Berechne Trading-Frequenz
        trading_frequency = self.trading_stats['total_trades'] / len(self.emotion_history) if self.emotion_history else 0
        
        # Berechne durchschnittliche Trade-Größe
        if self.trading_stats['total_trades'] > 0:
            avg_trade_size = sum(trade['trade_value'] for trade in self.paper_system.trades) / self.trading_stats['total_trades']
        else:
            avg_trade_size = 0
        
        # Berechne Sharpe Ratio
        if self.trading_stats['total_trades'] > 0:
            avg_trade_return = self.trading_stats['total_profit'] / self.trading_stats['total_trades']
            sharpe_ratio = avg_trade_return / max(self.paper_system.max_drawdown, 0.01)
        else:
            sharpe_ratio = 0
        
        # Berechne Timeframe Performance
        timeframe_performance = {}
        for tf in self.market_simulator.timeframes:
            if self.timeframe_performance[tf]:
                timeframe_performance[tf] = {
                    'total_contribution': sum(self.timeframe_performance[tf]),
                    'avg_contribution': np.mean(self.timeframe_performance[tf]),
                    'num_contributions': len(self.timeframe_performance[tf])
                }
            else:
                timeframe_performance[tf] = {
                    'total_contribution': 0,
                    'avg_contribution': 0,
                    'num_contributions': 0
                }
        
        # Berechne Attention Weights Analysis
        attention_analysis = {}
        if self.attention_weights_history:
            recent_weights = np.array(self.attention_weights_history[-20:])
            for i, tf in enumerate(self.market_simulator.timeframes):
                attention_analysis[tf] = {
                    'avg_weight': np.mean(recent_weights[:, i]),
                    'std_weight': np.std(recent_weights[:, i]),
                    'max_weight': np.max(recent_weights[:, i]),
                    'min_weight': np.min(recent_weights[:, i])
                }
        
        return {
            'final_portfolio_value': final_portfolio_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_trades': self.trading_stats['total_trades'],
            'winning_trades': self.trading_stats['winning_trades'],
            'losing_trades': self.trading_stats['losing_trades'],
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'total_profit': self.trading_stats['total_profit'],
            'max_drawdown': self.paper_system.max_drawdown,
            'max_drawdown_pct': self.paper_system.max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'emotion_changes': self.trading_stats['emotion_changes'],
            'final_emotion': self.emotion_engine.current_emotion.value,
            'emotion_intensity': self.emotion_engine.emotion_intensity,
            'trading_frequency': trading_frequency,
            'avg_trade_size': avg_trade_size,
            'commission_paid': self.paper_system.total_commission_paid,
            'slippage_cost': self.paper_system.total_slippage_cost,
            'emotion_history': self.emotion_history[-20:],
            'timeframe_performance': timeframe_performance,
            'attention_analysis': attention_analysis,
            'timeframe_weights': self.timeframe_weights
        }

def run_multi_timeframe_optimized_test():
    """Führe Multi-Timeframe Optimized Test durch"""
    
    print("🚀 MULTI-TIMEFRAME OPTIMIZED TRADING TEST")
    print("=" * 60)
    
    assets = ['AAPL', 'TSLA', 'BTC/USD', 'ETH/USD']
    results = {}
    
    for asset in assets:
        print(f"\n📊 Teste {asset} (Multi-Timeframe Optimized)...")
        
        # Erstelle Multi-Timeframe Test für dieses Asset
        test = MultiTimeframeOptimizedTradingTest()
        
        # Führe Multi-Timeframe Test durch
        result = test.run_multi_timeframe_test(steps=150)  # Erhöht für bessere Tests
        
        results[asset] = result
        
        # Zeige Ergebnisse
        print(f"\n📈 {asset} Multi-Timeframe Ergebnisse:")
        print(f"   Total Return: {result['total_return_pct']:.2f}%")
        print(f"   Win Rate: {result['win_rate_pct']:.1f}%")
        print(f"   Total Trades: {result['total_trades']}")
        print(f"   Trading Frequency: {result['trading_frequency']:.3f}")
        print(f"   Max Drawdown: {result['max_drawdown_pct']:.2f}%")
        print(f"   Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"   Final Emotion: {result['final_emotion']}")
        print(f"   Emotion Intensity: {result['emotion_intensity']:.2f}")
        print(f"   Emotion Changes: {result['emotion_changes']}")
        
        # Zeige Timeframe Performance
        print(f"   Timeframe Performance:")
        for tf, perf in result['timeframe_performance'].items():
            print(f"     {tf}: {perf['total_contribution']:.2f} (avg: {perf['avg_contribution']:.2f})")
        
        # Zeige Attention Analysis
        print(f"   Attention Weights:")
        for tf, att in result['attention_analysis'].items():
            print(f"     {tf}: {att['avg_weight']:.3f} ± {att['std_weight']:.3f}")
    
    # Analysiere alle Ergebnisse
    print(f"\n📋 MULTI-TIMEFRAME GESAMTANALYSE")
    print("=" * 50)
    
    # Erstelle Vergleichstabelle
    comparison_data = []
    for asset, result in results.items():
        comparison_data.append({
            'Asset': asset,
            'Return (%)': result['total_return_pct'],
            'Win Rate (%)': result['win_rate_pct'],
            'Total Trades': result['total_trades'],
            'Trading Freq': result['trading_frequency'],
            'Max DD (%)': result['max_drawdown_pct'],
            'Sharpe': result['sharpe_ratio'],
            'Final Emotion': result['final_emotion'],
            'Emotion Intensity': result['emotion_intensity'],
            'Emotion Changes': result['emotion_changes']
        })
    
    df = pd.DataFrame(comparison_data)
    print("\n📊 MULTI-TIMEFRAME PERFORMANCE VERGLEICH:")
    print(df.to_string(index=False, float_format='%.3f'))
    
    # Finde beste Performance
    best_return = df.loc[df['Return (%)'].idxmax()]
    best_winrate = df.loc[df['Win Rate (%)'].idxmax()]
    best_sharpe = df.loc[df['Sharpe'].idxmax()]
    most_active = df.loc[df['Trading Freq'].idxmax()]
    
    print(f"\n🏆 BESTE MULTI-TIMEFRAME PERFORMANCE:")
    print(f"   Höchster Return: {best_return['Asset']} ({best_return['Return (%)']:.2f}%)")
    print(f"   Beste Win Rate: {best_winrate['Asset']} ({best_winrate['Win Rate (%)']:.1f}%)")
    print(f"   Beste Sharpe Ratio: {best_sharpe['Asset']} ({best_sharpe['Sharpe']:.2f})")
    print(f"   Aktivstes Trading: {most_active['Asset']} (Freq: {most_active['Trading Freq']:.3f})")
    
    # Emotion-Analyse
    print(f"\n🧠 MULTI-TIMEFRAME EMOTION-ANALYSE:")
    emotion_counts = df['Final Emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"   {emotion}: {count} Assets")
    
    # Durchschnittliche Performance
    avg_return = df['Return (%)'].mean()
    avg_winrate = df['Win Rate (%)'].mean()
    avg_sharpe = df['Sharpe'].mean()
    avg_trading_freq = df['Trading Freq'].mean()
    avg_emotion_changes = df['Emotion Changes'].mean()
    
    print(f"\n📈 DURCHSCHNITTLICHE MULTI-TIMEFRAME PERFORMANCE:")
    print(f"   Return: {avg_return:.2f}%")
    print(f"   Win Rate: {avg_winrate:.1f}%")
    print(f"   Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"   Trading Frequency: {avg_trading_freq:.3f}")
    print(f"   Emotion Changes: {avg_emotion_changes:.1f}")
    
    return results

if __name__ == "__main__":
    run_multi_timeframe_optimized_test()
