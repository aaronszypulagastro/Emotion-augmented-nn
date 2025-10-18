"""
Optimiertes Trading-System mit verbesserter Performance
Implementiert alle Verbesserungen basierend auf den Test-Ergebnissen
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import unserer Standalone-Komponenten
from standalone_test import StandaloneTradingEmotionEngine, StandalonePaperTradingSystem, TradingEmotion

class OptimizedTradingEmotionEngine(StandaloneTradingEmotionEngine):
    """Optimierte Trading Emotion Engine mit verbesserter Performance"""
    
    def __init__(self, 
                 initial_emotion: TradingEmotion = TradingEmotion.NEUTRAL,
                 learning_rate: float = 0.02,  # Erhöht für schnellere Anpassung
                 emotion_decay: float = 0.98,  # Reduziert für länger anhaltende Emotionen
                 volatility_threshold: float = 0.02,
                 transition_threshold: float = 0.6):  # Niedriger für aktivere Transitions
        
        super().__init__(initial_emotion, learning_rate, emotion_decay, volatility_threshold)
        self.transition_threshold = transition_threshold
        
        # Erweiterte Emotion States mit mehr Granularität
        self.emotion_intensity = 1.0  # Emotion-Intensität (0-2)
        self.emotion_memory = []  # Speichert Emotion-Historie
        self.performance_memory = []  # Speichert Performance-Historie
        
        # Adaptive Parameter
        self.adaptive_learning_rate = learning_rate
        self.adaptive_decay = emotion_decay
        
    def update_performance(self, 
                          portfolio_return: float,
                          trade_return: float,
                          drawdown: float,
                          win_rate: float):
        """Erweiterte Performance-Update mit adaptiven Parametern"""
        
        # Speichere Performance-Historie
        self.performance_memory.append({
            'portfolio_return': portfolio_return,
            'trade_return': trade_return,
            'drawdown': drawdown,
            'win_rate': win_rate,
            'timestamp': len(self.performance_memory)
        })
        
        if len(self.performance_memory) > 100:
            self.performance_memory.pop(0)
        
        # Adaptive Learning Rate basierend auf Performance-Volatilität
        if len(self.performance_memory) > 10:
            recent_returns = [p['trade_return'] for p in self.performance_memory[-10:]]
            performance_volatility = np.std(recent_returns)
            
            # Erhöhe Learning Rate bei hoher Volatilität
            if performance_volatility > 0.02:
                self.adaptive_learning_rate = min(0.05, self.learning_rate * 1.5)
            else:
                self.adaptive_learning_rate = max(0.01, self.learning_rate * 0.8)
        
        # Erweiterte Emotion-Update-Logik
        self._update_emotion_states_optimized(portfolio_return, trade_return, drawdown, win_rate)
        self._transition_emotion_optimized()
        self._decay_emotion_states_optimized()
        
        # Update Emotion-Intensität
        self._update_emotion_intensity()
    
    def _update_emotion_states_optimized(self, portfolio_return: float, trade_return: float, drawdown: float, win_rate: float):
        """Optimierte Emotion State Updates"""
        
        # Berechne Performance-Trend
        if len(self.performance_memory) > 5:
            recent_returns = [p['trade_return'] for p in self.performance_memory[-5:]]
            performance_trend = np.mean(recent_returns)
        else:
            performance_trend = trade_return
        
        # CONFIDENT: Gute Performance mit Trend
        if portfolio_return > 0.005 and win_rate > 0.55 and performance_trend > 0:
            self.emotion_states[TradingEmotion.CONFIDENT] += self.adaptive_learning_rate * 0.4
        else:
            self.emotion_states[TradingEmotion.CONFIDENT] -= self.adaptive_learning_rate * 0.15
        
        # CAUTIOUS: Hohe Volatilität oder Drawdown
        if self.market_sentiment['volatility'] > 0.6 or drawdown > 0.03:
            self.emotion_states[TradingEmotion.CAUTIOUS] += self.adaptive_learning_rate * 0.5
        else:
            self.emotion_states[TradingEmotion.CAUTIOUS] -= self.adaptive_learning_rate * 0.1
        
        # FRUSTRATED: Schlechte Performance mit negativem Trend
        if portfolio_return < -0.005 or win_rate < 0.45 or performance_trend < -0.01:
            self.emotion_states[TradingEmotion.FRUSTRATED] += self.adaptive_learning_rate * 0.5
        else:
            self.emotion_states[TradingEmotion.FRUSTRATED] -= self.adaptive_learning_rate * 0.1
        
        # GREEDY: Sehr gute Performance mit starkem Trend
        if portfolio_return > 0.02 and win_rate > 0.65 and performance_trend > 0.01:
            self.emotion_states[TradingEmotion.GREEDY] += self.adaptive_learning_rate * 0.4
        else:
            self.emotion_states[TradingEmotion.GREEDY] -= self.adaptive_learning_rate * 0.1
        
        # FEARFUL: Große Verluste oder hohe Drawdowns
        if drawdown > 0.08 or portfolio_return < -0.02 or performance_trend < -0.02:
            self.emotion_states[TradingEmotion.FEARFUL] += self.adaptive_learning_rate * 0.5
        else:
            self.emotion_states[TradingEmotion.FEARFUL] -= self.adaptive_learning_rate * 0.1
        
        # OPTIMISTIC: Positive Marktausblick mit guter Performance
        if (self.market_sentiment['trend'] > 0.2 and 
            self.market_sentiment['momentum'] > 0.1 and 
            portfolio_return > 0):
            self.emotion_states[TradingEmotion.OPTIMISTIC] += self.adaptive_learning_rate * 0.3
        else:
            self.emotion_states[TradingEmotion.OPTIMISTIC] -= self.adaptive_learning_rate * 0.1
        
        # PESSIMISTIC: Negative Marktausblick mit schlechter Performance
        if (self.market_sentiment['trend'] < -0.2 and 
            self.market_sentiment['momentum'] < -0.1 and 
            portfolio_return < 0):
            self.emotion_states[TradingEmotion.PESSIMISTIC] += self.adaptive_learning_rate * 0.3
        else:
            self.emotion_states[TradingEmotion.PESSIMISTIC] -= self.adaptive_learning_rate * 0.1
        
        # NEUTRAL: Ausgewogene Bedingungen
        if (abs(portfolio_return) < 0.003 and 
            0.45 < win_rate < 0.55 and 
            self.market_sentiment['volatility'] < 0.4 and
            abs(performance_trend) < 0.005):
            self.emotion_states[TradingEmotion.NEUTRAL] += self.adaptive_learning_rate * 0.3
        else:
            self.emotion_states[TradingEmotion.NEUTRAL] -= self.adaptive_learning_rate * 0.1
        
        # Clamp all values to [0, 1]
        for emotion in self.emotion_states:
            self.emotion_states[emotion] = np.clip(self.emotion_states[emotion], 0, 1)
    
    def _transition_emotion_optimized(self):
        """Optimierte Emotion-Transitions mit niedrigerer Schwelle"""
        
        # Finde Emotion mit höchstem State
        max_emotion = max(self.emotion_states.items(), key=lambda x: x[1])
        
        # Niedrigere Schwelle für aktivere Transitions
        if max_emotion[1] > self.transition_threshold and max_emotion[0] != self.current_emotion:
            # Speichere vorherige Emotion
            self.emotion_memory.append(self.current_emotion)
            if len(self.emotion_memory) > 20:
                self.emotion_memory.pop(0)
            
            # Transition mit Wahrscheinlichkeit basierend auf Intensität
            transition_prob = max_emotion[1] * self.emotion_intensity
            if np.random.random() < transition_prob:
                self.current_emotion = max_emotion[0]
    
    def _decay_emotion_states_optimized(self):
        """Optimierte Emotion State Decay"""
        
        # Adaptive Decay basierend auf Performance
        if len(self.performance_memory) > 5:
            recent_performance = [p['trade_return'] for p in self.performance_memory[-5:]]
            avg_performance = np.mean(recent_performance)
            
            # Reduziere Decay bei guter Performance (Emotionen halten länger)
            if avg_performance > 0.01:
                self.adaptive_decay = min(0.99, self.emotion_decay * 1.1)
            else:
                self.adaptive_decay = max(0.95, self.emotion_decay * 0.9)
        else:
            self.adaptive_decay = self.emotion_decay
        
        # Wende adaptiven Decay an
        for emotion in self.emotion_states:
            self.emotion_states[emotion] *= self.adaptive_decay
    
    def _update_emotion_intensity(self):
        """Update Emotion-Intensität basierend auf Performance"""
        
        if len(self.performance_memory) > 10:
            recent_returns = [p['trade_return'] for p in self.performance_memory[-10:]]
            performance_volatility = np.std(recent_returns)
            
            # Erhöhe Intensität bei hoher Volatilität
            if performance_volatility > 0.03:
                self.emotion_intensity = min(2.0, self.emotion_intensity * 1.1)
            else:
                self.emotion_intensity = max(0.5, self.emotion_intensity * 0.95)
    
    def get_enhanced_risk_tolerance(self) -> float:
        """Erweiterte Risikotoleranz mit Intensität"""
        
        base_tolerance = self.get_risk_tolerance()
        
        # Modifiziere basierend auf Emotion-Intensität
        intensity_modifier = 0.5 + (self.emotion_intensity * 0.5)
        
        # Modifiziere basierend auf Performance-Trend
        if len(self.performance_memory) > 5:
            recent_returns = [p['trade_return'] for p in self.performance_memory[-5:]]
            performance_trend = np.mean(recent_returns)
            
            if performance_trend > 0.01:
                trend_modifier = 1.2
            elif performance_trend < -0.01:
                trend_modifier = 0.8
            else:
                trend_modifier = 1.0
        else:
            trend_modifier = 1.0
        
        enhanced_tolerance = base_tolerance * intensity_modifier * trend_modifier
        return np.clip(enhanced_tolerance, 0.1, 1.0)
    
    def get_enhanced_position_sizing_modifier(self) -> float:
        """Erweiterte Position Sizing mit Intensität"""
        
        base_modifier = self.get_position_sizing_modifier()
        
        # Modifiziere basierend auf Emotion-Intensität
        intensity_modifier = 0.5 + (self.emotion_intensity * 0.5)
        
        # Modifiziere basierend auf Win Rate
        if len(self.performance_memory) > 10:
            recent_win_rates = [p['win_rate'] for p in self.performance_memory[-10:]]
            avg_win_rate = np.mean(recent_win_rates)
            
            if avg_win_rate > 0.6:
                win_rate_modifier = 1.3
            elif avg_win_rate < 0.4:
                win_rate_modifier = 0.7
            else:
                win_rate_modifier = 1.0
        else:
            win_rate_modifier = 1.0
        
        enhanced_modifier = base_modifier * intensity_modifier * win_rate_modifier
        return np.clip(enhanced_modifier, 0.2, 2.0)

class OptimizedPaperTradingSystem(StandalonePaperTradingSystem):
    """Optimiertes Paper Trading System mit verbesserter Performance"""
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 min_trade_size: float = 100.0,
                 max_position_size: float = 0.3):  # Erhöht für mehr Trading
        
        super().__init__(initial_capital, commission, slippage, min_trade_size)
        self.max_position_size = max_position_size
        
        # Erweiterte Trading-Statistiken
        self.daily_pnl = []
        self.trade_sequence = []
        self.position_history = []
        
        # Performance-Tracking
        self.peak_portfolio_value = initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
    def execute_trade(self, 
                     symbol: str, 
                     action: str, 
                     quantity: float, 
                     price: float) -> Dict:
        """Erweiterte Trade-Execution mit verbesserter Logik"""
        
        # Berechne Slippage basierend auf Trade-Größe
        trade_value = quantity * price
        size_based_slippage = min(0.002, self.slippage * (1 + trade_value / 10000))
        
        # Berechne Slippage
        slippage_factor = np.random.normal(0, size_based_slippage)
        if action == 'buy':
            execution_price = price * (1 + abs(slippage_factor))
        else:
            execution_price = price * (1 - abs(slippage_factor))
        
        # Berechne Trade-Wert
        trade_value = quantity * execution_price
        
        # Prüfe Mindest-Trade-Größe
        if trade_value < self.min_trade_size:
            return {
                'success': False,
                'reason': f'Trade value {trade_value:.2f} below minimum {self.min_trade_size}',
                'trade_value': trade_value
            }
        
        # Berechne Kommission
        commission_cost = trade_value * self.commission
        
        # Prüfe verfügbares Kapital
        if action == 'buy':
            total_cost = trade_value + commission_cost
            if total_cost > self.cash:
                return {
                    'success': False,
                    'reason': f'Insufficient cash: {self.cash:.2f} < {total_cost:.2f}',
                    'required': total_cost
                }
        
        # Führe Trade aus
        if action == 'buy':
            self.cash -= (trade_value + commission_cost)
            
            if symbol in self.positions:
                old_shares = self.positions[symbol]['shares']
                old_avg_price = self.positions[symbol]['avg_price']
                new_avg_price = ((old_shares * old_avg_price) + (quantity * execution_price)) / (old_shares + quantity)
                self.positions[symbol]['shares'] += quantity
                self.positions[symbol]['avg_price'] = new_avg_price
            else:
                self.positions[symbol] = {
                    'shares': quantity,
                    'avg_price': execution_price
                }
        
        elif action == 'sell':
            if symbol not in self.positions or self.positions[symbol]['shares'] < quantity:
                return {
                    'success': False,
                    'reason': f'Insufficient shares: {self.positions.get(symbol, {}).get("shares", 0):.2f} < {quantity:.2f}',
                    'available': self.positions.get(symbol, {}).get("shares", 0)
                }
            
            self.cash += (trade_value - commission_cost)
            self.positions[symbol]['shares'] -= quantity
            
            if self.positions[symbol]['shares'] <= 0.001:
                del self.positions[symbol]
        
        # Aktualisiere Statistiken
        self.total_commission_paid += commission_cost
        self.total_slippage_cost += abs(slippage_factor) * trade_value
        
        # Speichere Trade
        trade_record = {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'execution_price': execution_price,
            'trade_value': trade_value,
            'commission': commission_cost,
            'slippage': abs(slippage_factor) * trade_value,
            'cash_after': self.cash
        }
        
        self.trades.append(trade_record)
        self.trade_sequence.append(action)
        
        # Update Performance-Tracking
        self._update_performance_tracking()
        
        return {
            'success': True,
            'trade_record': trade_record,
            'execution_price': execution_price,
            'commission': commission_cost,
            'slippage': abs(slippage_factor) * trade_value
        }
    
    def _update_performance_tracking(self):
        """Update Performance-Tracking"""
        
        # Berechne aktuellen Portfolio-Wert (vereinfacht)
        current_portfolio_value = self.cash + sum(pos['shares'] * pos['avg_price'] for pos in self.positions.values())
        
        # Update Peak und Drawdown
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Speichere Position-Historie
        self.position_history.append({
            'portfolio_value': current_portfolio_value,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'drawdown': self.current_drawdown
        })
        
        if len(self.position_history) > 1000:
            self.position_history.pop(0)

class OptimizedTradingTest:
    """Optimierter Trading Test mit allen Verbesserungen"""
    
    def __init__(self):
        self.market_simulator = None  # Wird in run_test gesetzt
        self.emotion_engine = OptimizedTradingEmotionEngine()
        self.paper_system = OptimizedPaperTradingSystem(
            initial_capital=10000.0,
            commission=0.001,
            slippage=0.0005,
            min_trade_size=100.0,
            max_position_size=0.3
        )
        
        # Erweiterte Trading-Statistiken
        self.trading_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'emotion_changes': 0,
            'trading_frequency': 0.0,
            'avg_trade_size': 0.0,
            'volatility_adaptation': 0.0
        }
        
        self.initial_emotion = self.emotion_engine.current_emotion
        self.emotion_history = []
        
    def run_optimized_test(self, steps: int = 200) -> Dict:  # Erhöht für längere Tests
        """Führe optimierten Trading Test durch"""
        
        print(f"🚀 Starte Optimierten Trading Test...")
        print(f"Steps: {steps}")
        print(f"Initial Capital: ${self.paper_system.initial_capital:,.2f}")
        print(f"Max Position Size: {self.paper_system.max_position_size*100:.1f}%")
        print(f"Transition Threshold: {self.emotion_engine.transition_threshold}")
        
        # Generiere Marktdaten
        print(f"\n📊 Generiere Marktdaten...")
        from realistic_paper_trading_test import RealisticMarketSimulator
        self.market_simulator = RealisticMarketSimulator()
        market_data = self.market_simulator.generate_market_data(steps)
        
        # Führe Trading-Simulation durch
        print(f"\n💹 Starte Optimierte Trading-Simulation...")
        
        for step in range(steps):
            if step % 40 == 0:
                portfolio_value = self.paper_system.get_portfolio_value(self.market_simulator.current_prices)
                print(f"   Step {step}/{steps} - Portfolio: ${portfolio_value:,.2f} - Emotion: {self.emotion_engine.current_emotion.value}")
            
            # Wähle zufälliges Asset für diesen Step
            symbol = np.random.choice(list(self.market_simulator.assets.keys()))
            market_info = market_data[symbol][step]
            
            # Update Emotion Engine basierend auf Marktbedingungen
            self._update_emotion_engine_optimized(market_info, symbol)
            
            # Bestimme Trading-Entscheidung
            trading_decision = self._make_optimized_trading_decision(market_info, symbol)
            
            # Führe Trade aus
            if trading_decision['action'] != 'hold':
                self._execute_optimized_trade(trading_decision, market_info, symbol)
            
            # Speichere Emotion-Historie
            self.emotion_history.append(self.emotion_engine.current_emotion.value)
        
        # Berechne finale Ergebnisse
        final_results = self._calculate_optimized_results()
        
        print(f"\n✅ Optimierter Trading Test abgeschlossen!")
        return final_results
    
    def _update_emotion_engine_optimized(self, market_info: Dict, symbol: str):
        """Optimierte Emotion Engine Updates"""
        
        # Berechne Portfolio-Performance
        current_portfolio_value = self.paper_system.get_portfolio_value(self.market_simulator.current_prices)
        portfolio_return = (current_portfolio_value - self.paper_system.initial_capital) / self.paper_system.initial_capital
        
        # Berechne Trade-Performance
        if self.trading_stats['total_trades'] > 0:
            win_rate = self.trading_stats['winning_trades'] / self.trading_stats['total_trades']
        else:
            win_rate = 0.5
        
        # Update Market Sentiment
        self.emotion_engine.update_market_sentiment(
            price_change=market_info['price_change'],
            volume_change=market_info['volume'] / 5000 - 1,
            volatility=market_info['volatility'],
            trend_strength=market_info['price_change'] * 2
        )
        
        # Update Performance mit optimierter Engine
        self.emotion_engine.update_performance(
            portfolio_return=portfolio_return,
            trade_return=market_info['price_change'],
            drawdown=self.paper_system.max_drawdown,
            win_rate=win_rate
        )
        
        # Track Emotion Changes
        if self.emotion_engine.current_emotion != self.initial_emotion:
            self.trading_stats['emotion_changes'] += 1
            self.initial_emotion = self.emotion_engine.current_emotion
    
    def _make_optimized_trading_decision(self, market_info: Dict, symbol: str) -> Dict:
        """Optimierte Trading-Entscheidungen"""
        
        enhanced_risk_tolerance = self.emotion_engine.get_enhanced_risk_tolerance()
        enhanced_position_modifier = self.emotion_engine.get_enhanced_position_sizing_modifier()
        current_emotion = self.emotion_engine.current_emotion
        emotion_intensity = self.emotion_engine.emotion_intensity
        
        # Erhöhte Trade-Wahrscheinlichkeit basierend auf Emotion
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
            return {'action': 'hold', 'quantity': 0}
        
        # Erweiterte Trading-Logik
        price_change = market_info['price_change']
        volatility = market_info['volatility']
        
        # Emotion-basierte Trading-Logik mit höherer Sensitivität
        if current_emotion in [TradingEmotion.CONFIDENT, TradingEmotion.GREEDY, TradingEmotion.OPTIMISTIC]:
            # Positive Emotionen = Aktiveres Trading
            if price_change > -0.005 or volatility < 0.04:  # Niedrigere Schwelle
                action = 'buy'
            else:
                action = 'hold'
        elif current_emotion in [TradingEmotion.FEARFUL, TradingEmotion.FRUSTRATED, TradingEmotion.PESSIMISTIC]:
            # Negative Emotionen = Defensiveres Trading
            if price_change < 0.005 or volatility > 0.03:  # Niedrigere Schwelle
                action = 'sell'
            else:
                action = 'hold'
        else:
            # Neutrale Emotionen = Balanced Trading
            if price_change > 0.005:
                action = 'buy'
            elif price_change < -0.005:
                action = 'sell'
            else:
                action = 'hold'
        
        # Bestimme Trade-Größe mit erweiterten Modifikatoren
        if action != 'hold':
            base_quantity = 8.0  # Erhöht für mehr Trading
            quantity = base_quantity * enhanced_position_modifier * enhanced_risk_tolerance
            
            # Volatilitäts-Anpassung
            if volatility > 0.05:
                quantity *= 0.6
            elif volatility < 0.02:
                quantity *= 1.8
            
            # Emotion-Intensität Anpassung
            quantity *= emotion_intensity
            
            quantity = max(2.0, quantity)  # Erhöhte Mindest-Trade-Größe
        else:
            quantity = 0
        
        return {
            'action': action,
            'quantity': quantity,
            'emotion': current_emotion.value,
            'risk_tolerance': enhanced_risk_tolerance,
            'position_modifier': enhanced_position_modifier,
            'emotion_intensity': emotion_intensity
        }
    
    def _execute_optimized_trade(self, decision: Dict, market_info: Dict, symbol: str):
        """Optimierte Trade-Execution"""
        
        if decision['action'] == 'hold':
            return
        
        # Führe Trade aus
        result = self.paper_system.execute_trade(
            symbol=symbol,
            action=decision['action'],
            quantity=decision['quantity'],
            price=market_info['price']
        )
        
        if result['success']:
            self.trading_stats['total_trades'] += 1
            
            # Berechne Trade-Ergebnis
            if decision['action'] == 'sell' and symbol in self.paper_system.positions:
                avg_price = self.paper_system.positions[symbol]['avg_price']
                trade_profit = (market_info['price'] - avg_price) * decision['quantity']
                
                if trade_profit > 0:
                    self.trading_stats['winning_trades'] += 1
                else:
                    self.trading_stats['losing_trades'] += 1
                
                self.trading_stats['total_profit'] += trade_profit
    
    def _calculate_optimized_results(self) -> Dict:
        """Berechne optimierte Ergebnisse"""
        
        final_portfolio_value = self.paper_system.get_portfolio_value(self.market_simulator.current_prices)
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
            'emotion_history': self.emotion_history[-20:]  # Letzte 20 Emotionen
        }

def run_optimized_multi_asset_test():
    """Führe optimierten Test für mehrere Assets durch"""
    
    print("🚀 OPTIMIERTER MULTI-ASSET TRADING TEST")
    print("=" * 60)
    
    assets = ['AAPL', 'TSLA', 'BTC/USD', 'ETH/USD']
    results = {}
    
    for asset in assets:
        print(f"\n📊 Teste {asset} (Optimiert)...")
        
        # Erstelle optimierten Test für dieses Asset
        test = OptimizedTradingTest()
        
        # Führe optimierten Test durch
        result = test.run_optimized_test(steps=100)  # Erhöht für bessere Tests
        
        results[asset] = result
        
        # Zeige Ergebnisse
        print(f"\n📈 {asset} Optimierte Ergebnisse:")
        print(f"   Total Return: {result['total_return_pct']:.2f}%")
        print(f"   Win Rate: {result['win_rate_pct']:.1f}%")
        print(f"   Total Trades: {result['total_trades']}")
        print(f"   Trading Frequency: {result['trading_frequency']:.3f}")
        print(f"   Max Drawdown: {result['max_drawdown_pct']:.2f}%")
        print(f"   Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"   Final Emotion: {result['final_emotion']}")
        print(f"   Emotion Intensity: {result['emotion_intensity']:.2f}")
        print(f"   Emotion Changes: {result['emotion_changes']}")
    
    # Analysiere alle Ergebnisse
    print(f"\n📋 OPTIMIERTE GESAMTANALYSE")
    print("=" * 40)
    
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
    print("\n📊 OPTIMIERTE PERFORMANCE VERGLEICH:")
    print(df.to_string(index=False, float_format='%.3f'))
    
    # Finde beste Performance
    best_return = df.loc[df['Return (%)'].idxmax()]
    best_winrate = df.loc[df['Win Rate (%)'].idxmax()]
    best_sharpe = df.loc[df['Sharpe'].idxmax()]
    most_active = df.loc[df['Trading Freq'].idxmax()]
    
    print(f"\n🏆 BESTE OPTIMIERTE PERFORMANCE:")
    print(f"   Höchster Return: {best_return['Asset']} ({best_return['Return (%)']:.2f}%)")
    print(f"   Beste Win Rate: {best_winrate['Asset']} ({best_winrate['Win Rate (%)']:.1f}%)")
    print(f"   Beste Sharpe Ratio: {best_sharpe['Asset']} ({best_sharpe['Sharpe']:.2f})")
    print(f"   Aktivstes Trading: {most_active['Asset']} (Freq: {most_active['Trading Freq']:.3f})")
    
    # Emotion-Analyse
    print(f"\n🧠 OPTIMIERTE EMOTION-ANALYSE:")
    emotion_counts = df['Final Emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"   {emotion}: {count} Assets")
    
    # Durchschnittliche Performance
    avg_return = df['Return (%)'].mean()
    avg_winrate = df['Win Rate (%)'].mean()
    avg_sharpe = df['Sharpe'].mean()
    avg_trading_freq = df['Trading Freq'].mean()
    avg_emotion_changes = df['Emotion Changes'].mean()
    
    print(f"\n📈 DURCHSCHNITTLICHE OPTIMIERTE PERFORMANCE:")
    print(f"   Return: {avg_return:.2f}%")
    print(f"   Win Rate: {avg_winrate:.1f}%")
    print(f"   Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"   Trading Frequency: {avg_trading_freq:.3f}")
    print(f"   Emotion Changes: {avg_emotion_changes:.1f}")
    
    return results

if __name__ == "__main__":
    run_optimized_multi_asset_test()
