"""
Standalone Test ohne externe Dependencies
Testet nur die Kern-Komponenten des Trading-Systems
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class TradingEmotion(Enum):
    """Trading-spezifische Emotionen"""
    CONFIDENT = "confident"
    CAUTIOUS = "cautious"
    FRUSTRATED = "frustrated"
    GREEDY = "greedy"
    FEARFUL = "fearful"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    NEUTRAL = "neutral"

class StandaloneTradingEmotionEngine:
    """Standalone Trading Emotion Engine"""
    
    def __init__(self, 
                 initial_emotion: TradingEmotion = TradingEmotion.NEUTRAL,
                 learning_rate: float = 0.01,
                 emotion_decay: float = 0.95,
                 volatility_threshold: float = 0.02):
        
        self.current_emotion = initial_emotion
        self.learning_rate = learning_rate
        self.emotion_decay = emotion_decay
        self.volatility_threshold = volatility_threshold
        
        # Emotion States (0-1 scale)
        self.emotion_states = {
            TradingEmotion.CONFIDENT: 0.5,
            TradingEmotion.CAUTIOUS: 0.5,
            TradingEmotion.FRUSTRATED: 0.0,
            TradingEmotion.GREEDY: 0.0,
            TradingEmotion.FEARFUL: 0.0,
            TradingEmotion.OPTIMISTIC: 0.5,
            TradingEmotion.PESSIMISTIC: 0.0,
            TradingEmotion.NEUTRAL: 1.0
        }
        
        # Performance History
        self.performance_history = []
        self.volatility_history = []
        
        # Market Sentiment
        self.market_sentiment = {
            'trend': 0.0,
            'volatility': 0.0,
            'momentum': 0.0,
            'volume': 0.0
        }
    
    def update_market_sentiment(self, 
                              price_change: float,
                              volume_change: float,
                              volatility: float,
                              trend_strength: float):
        """Aktualisiere Markt-Sentiment"""
        
        self.market_sentiment['trend'] = np.tanh(price_change * 10)
        self.market_sentiment['volatility'] = np.clip(volatility / self.volatility_threshold, 0, 1)
        self.market_sentiment['momentum'] = np.tanh(trend_strength * 5)
        self.market_sentiment['volume'] = np.clip(volume_change, 0, 1)
        
        self.volatility_history.append(volatility)
        if len(self.volatility_history) > 100:
            self.volatility_history.pop(0)
    
    def update_performance(self, 
                          portfolio_return: float,
                          trade_return: float,
                          drawdown: float,
                          win_rate: float):
        """Aktualisiere Emotion basierend auf Performance"""
        
        self.performance_history.append({
            'portfolio_return': portfolio_return,
            'trade_return': trade_return,
            'drawdown': drawdown,
            'win_rate': win_rate
        })
        
        if len(self.performance_history) > 50:
            self.performance_history.pop(0)
        
        # Berechne Performance-Metriken
        recent_returns = [p['trade_return'] for p in self.performance_history[-10:]]
        avg_return = np.mean(recent_returns) if recent_returns else 0.0
        recent_win_rate = np.mean([p['win_rate'] for p in self.performance_history[-10:]])
        
        # Emotion Transition Logic
        self._update_emotion_states(avg_return, recent_win_rate, drawdown)
        self._transition_emotion()
        self._decay_emotion_states()
    
    def _update_emotion_states(self, avg_return: float, win_rate: float, drawdown: float):
        """Aktualisiere Emotion States"""
        
        # CONFIDENT: Gute Performance
        if avg_return > 0.01 and win_rate > 0.6:
            self.emotion_states[TradingEmotion.CONFIDENT] += self.learning_rate * 0.3
        else:
            self.emotion_states[TradingEmotion.CONFIDENT] -= self.learning_rate * 0.1
        
        # CAUTIOUS: Hohe Volatilität
        if self.market_sentiment['volatility'] > 0.7 or drawdown > 0.05:
            self.emotion_states[TradingEmotion.CAUTIOUS] += self.learning_rate * 0.4
        else:
            self.emotion_states[TradingEmotion.CAUTIOUS] -= self.learning_rate * 0.1
        
        # FRUSTRATED: Schlechte Performance
        if avg_return < -0.01 or win_rate < 0.4:
            self.emotion_states[TradingEmotion.FRUSTRATED] += self.learning_rate * 0.4
        else:
            self.emotion_states[TradingEmotion.FRUSTRATED] -= self.learning_rate * 0.1
        
        # GREEDY: Sehr gute Performance
        if avg_return > 0.03 and win_rate > 0.7:
            self.emotion_states[TradingEmotion.GREEDY] += self.learning_rate * 0.3
        else:
            self.emotion_states[TradingEmotion.GREEDY] -= self.learning_rate * 0.1
        
        # FEARFUL: Große Verluste
        if drawdown > 0.1 or avg_return < -0.02:
            self.emotion_states[TradingEmotion.FEARFUL] += self.learning_rate * 0.4
        else:
            self.emotion_states[TradingEmotion.FEARFUL] -= self.learning_rate * 0.1
        
        # Clamp all values to [0, 1]
        for emotion in self.emotion_states:
            self.emotion_states[emotion] = np.clip(self.emotion_states[emotion], 0, 1)
    
    def _transition_emotion(self):
        """Führe Emotion-Übergang durch"""
        max_emotion = max(self.emotion_states.items(), key=lambda x: x[1])
        
        if max_emotion[1] > 0.7 and max_emotion[0] != self.current_emotion:
            self.current_emotion = max_emotion[0]
    
    def _decay_emotion_states(self):
        """Lasse Emotion States abklingen"""
        for emotion in self.emotion_states:
            self.emotion_states[emotion] *= self.emotion_decay
    
    def get_emotion_vector(self) -> np.ndarray:
        """Erstelle Emotion-Vektor"""
        emotion_vector = np.array([
            self.emotion_states[TradingEmotion.CONFIDENT],
            self.emotion_states[TradingEmotion.CAUTIOUS],
            self.emotion_states[TradingEmotion.FRUSTRATED],
            self.emotion_states[TradingEmotion.GREEDY],
            self.emotion_states[TradingEmotion.FEARFUL],
            self.emotion_states[TradingEmotion.OPTIMISTIC],
            self.emotion_states[TradingEmotion.PESSIMISTIC],
            self.emotion_states[TradingEmotion.NEUTRAL]
        ], dtype=np.float32)
        
        return emotion_vector
    
    def get_risk_tolerance(self) -> float:
        """Berechne Risikotoleranz"""
        risk_tolerances = {
            TradingEmotion.CONFIDENT: 0.8,
            TradingEmotion.CAUTIOUS: 0.4,
            TradingEmotion.FRUSTRATED: 0.2,
            TradingEmotion.GREEDY: 0.9,
            TradingEmotion.FEARFUL: 0.1,
            TradingEmotion.OPTIMISTIC: 0.6,
            TradingEmotion.PESSIMISTIC: 0.3,
            TradingEmotion.NEUTRAL: 0.5
        }
        
        return risk_tolerances.get(self.current_emotion, 0.5)
    
    def get_position_sizing_modifier(self) -> float:
        """Berechne Position Sizing Modifier"""
        sizing_modifiers = {
            TradingEmotion.CONFIDENT: 1.3,
            TradingEmotion.CAUTIOUS: 0.6,
            TradingEmotion.FRUSTRATED: 0.3,
            TradingEmotion.GREEDY: 1.6,
            TradingEmotion.FEARFUL: 0.2,
            TradingEmotion.OPTIMISTIC: 1.1,
            TradingEmotion.PESSIMISTIC: 0.5,
            TradingEmotion.NEUTRAL: 1.0
        }
        
        return sizing_modifiers.get(self.current_emotion, 1.0)

class StandalonePaperTradingSystem:
    """Standalone Paper Trading System"""
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 min_trade_size: float = 100.0):
        
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.min_trade_size = min_trade_size
        
        # Portfolio State
        self.cash = initial_capital
        self.positions = {}
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        self.trades = []
    
    def execute_trade(self, 
                     symbol: str, 
                     action: str, 
                     quantity: float, 
                     price: float) -> Dict:
        """Führe Trade aus"""
        
        # Berechne Slippage
        slippage_factor = np.random.normal(0, self.slippage)
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
        
        return {
            'success': True,
            'trade_record': trade_record,
            'execution_price': execution_price,
            'commission': commission_cost,
            'slippage': abs(slippage_factor) * trade_value
        }
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Berechne Portfolio-Wert"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total_value += position['shares'] * current_prices[symbol]
        
        return total_value
    
    def get_performance_metrics(self) -> Dict:
        """Berechne Performance-Metriken"""
        return {
            'total_trades': len(self.trades),
            'total_commission_paid': self.total_commission_paid,
            'total_slippage_cost': self.total_slippage_cost,
            'final_cash': self.cash,
            'final_positions': self.positions
        }

def test_standalone_emotion_engine():
    """Teste Standalone Emotion Engine"""
    
    print("🧠 Teste Standalone Trading Emotion Engine...")
    
    try:
        emotion_engine = StandaloneTradingEmotionEngine()
        
        print(f"✅ Initial Emotion: {emotion_engine.current_emotion.value}")
        print(f"✅ Risk Tolerance: {emotion_engine.get_risk_tolerance():.2f}")
        
        # Simuliere verschiedene Szenarien
        scenarios = [
            {"name": "Gute Performance", "return": 0.02, "win_rate": 0.7, "drawdown": 0.01},
            {"name": "Schlechte Performance", "return": -0.015, "win_rate": 0.3, "drawdown": 0.05},
            {"name": "Volatile Märkte", "return": 0.005, "win_rate": 0.5, "drawdown": 0.03},
            {"name": "Sehr gute Performance", "return": 0.04, "win_rate": 0.8, "drawdown": 0.005},
            {"name": "Große Verluste", "return": -0.03, "win_rate": 0.2, "drawdown": 0.12}
        ]
        
        for scenario in scenarios:
            print(f"\n--- {scenario['name']} ---")
            
            # Update Market Sentiment
            emotion_engine.update_market_sentiment(
                price_change=scenario['return'],
                volume_change=0.1,
                volatility=0.02,
                trend_strength=scenario['return'] * 2
            )
            
            # Update Performance
            emotion_engine.update_performance(
                portfolio_return=scenario['return'],
                trade_return=scenario['return'],
                drawdown=scenario['drawdown'],
                win_rate=scenario['win_rate']
            )
            
            print(f"Emotion: {emotion_engine.current_emotion.value}")
            print(f"Risk Tolerance: {emotion_engine.get_risk_tolerance():.2f}")
            print(f"Position Sizing: {emotion_engine.get_position_sizing_modifier():.2f}")
            
            # Zeige Emotion Vector
            emotion_vector = emotion_engine.get_emotion_vector()
            print(f"Emotion Vector: {emotion_vector}")
        
        print("\n✅ Standalone Trading Emotion Engine Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Fehler bei Standalone Emotion Engine Test: {e}")
        return False

def test_standalone_paper_trading():
    """Teste Standalone Paper Trading System"""
    
    print("\n📊 Teste Standalone Paper Trading System...")
    
    try:
        paper_system = StandalonePaperTradingSystem(
            initial_capital=10000.0,
            commission=0.001,
            slippage=0.0005,
            min_trade_size=100.0
        )
        
        print(f"✅ Initial Capital: ${paper_system.initial_capital:,.2f}")
        print(f"✅ Commission: {paper_system.commission*100:.1f}%")
        print(f"✅ Slippage: {paper_system.slippage*100:.2f}%")
        
        # Simuliere Trades
        print("\n--- Simuliere Trades ---")
        
        # Kauf
        result1 = paper_system.execute_trade(
            symbol='AAPL',
            action='buy',
            quantity=10.0,
            price=150.0
        )
        
        if result1['success']:
            print(f"✅ Kauf erfolgreich: {result1['trade_record']['quantity']} AAPL @ ${result1['execution_price']:.2f}")
            print(f"   Commission: ${result1['commission']:.2f}")
            print(f"   Cash nach Trade: ${result1['trade_record']['cash_after']:,.2f}")
        else:
            print(f"❌ Kauf fehlgeschlagen: {result1['reason']}")
        
        # Verkauf
        result2 = paper_system.execute_trade(
            symbol='AAPL',
            action='sell',
            quantity=5.0,
            price=155.0
        )
        
        if result2['success']:
            print(f"✅ Verkauf erfolgreich: {result2['trade_record']['quantity']} AAPL @ ${result2['execution_price']:.2f}")
            print(f"   Commission: ${result2['commission']:.2f}")
            print(f"   Cash nach Trade: ${result2['trade_record']['cash_after']:,.2f}")
        else:
            print(f"❌ Verkauf fehlgeschlagen: {result2['reason']}")
        
        # Performance Metrics
        metrics = paper_system.get_performance_metrics()
        print(f"\n📈 Performance Metrics:")
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Total Commission Paid: ${metrics['total_commission_paid']:.2f}")
        print(f"   Total Slippage Cost: ${metrics['total_slippage_cost']:.2f}")
        print(f"   Final Cash: ${metrics['final_cash']:,.2f}")
        print(f"   Final Positions: {metrics['final_positions']}")
        
        print("\n✅ Standalone Paper Trading System Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Fehler bei Standalone Paper Trading Test: {e}")
        return False

def test_emotion_paper_trading_integration():
    """Teste Integration von Emotion Engine und Paper Trading"""
    
    print("\n🔗 Teste Emotion Engine + Paper Trading Integration...")
    
    try:
        # Erstelle beide Systeme
        emotion_engine = StandaloneTradingEmotionEngine()
        paper_system = StandalonePaperTradingSystem()
        
        print(f"✅ Beide Systeme erstellt")
        
        # Simuliere Trading-Session
        print("\n--- Simuliere Trading-Session ---")
        
        # Verschiedene Marktbedingungen
        market_conditions = [
            {"price": 150.0, "return": 0.02, "volatility": 0.01},
            {"price": 155.0, "return": 0.01, "volatility": 0.02},
            {"price": 152.0, "return": -0.01, "volatility": 0.03},
            {"price": 158.0, "return": 0.03, "volatility": 0.015},
            {"price": 160.0, "return": 0.01, "volatility": 0.01}
        ]
        
        for i, condition in enumerate(market_conditions):
            print(f"\n--- Trading Step {i+1} ---")
            print(f"Market: Price=${condition['price']:.2f}, Return={condition['return']*100:.1f}%, Vol={condition['volatility']*100:.1f}%")
            
            # Update Emotion Engine
            emotion_engine.update_market_sentiment(
                price_change=condition['return'],
                volume_change=0.1,
                volatility=condition['volatility'],
                trend_strength=condition['return'] * 2
            )
            
            emotion_engine.update_performance(
                portfolio_return=condition['return'],
                trade_return=condition['return'],
                drawdown=0.02,
                win_rate=0.6
            )
            
            # Bestimme Trade basierend auf Emotion
            risk_tolerance = emotion_engine.get_risk_tolerance()
            position_modifier = emotion_engine.get_position_sizing_modifier()
            
            print(f"Emotion: {emotion_engine.current_emotion.value}")
            print(f"Risk Tolerance: {risk_tolerance:.2f}")
            print(f"Position Modifier: {position_modifier:.2f}")
            
            # Führe Trade aus basierend auf Emotion
            if risk_tolerance > 0.6:  # Hohe Risikotoleranz = Trade
                if condition['return'] > 0:  # Positive Returns = Kauf
                    quantity = 5.0 * position_modifier
                    result = paper_system.execute_trade(
                        symbol='AAPL',
                        action='buy',
                        quantity=quantity,
                        price=condition['price']
                    )
                    
                    if result['success']:
                        print(f"✅ Kauf: {quantity:.1f} AAPL @ ${result['execution_price']:.2f}")
                    else:
                        print(f"❌ Kauf fehlgeschlagen: {result['reason']}")
                
                elif condition['return'] < -0.01:  # Negative Returns = Verkauf
                    if 'AAPL' in paper_system.positions:
                        quantity = 2.0 * position_modifier
                        result = paper_system.execute_trade(
                            symbol='AAPL',
                            action='sell',
                            quantity=quantity,
                            price=condition['price']
                        )
                        
                        if result['success']:
                            print(f"✅ Verkauf: {quantity:.1f} AAPL @ ${result['execution_price']:.2f}")
                        else:
                            print(f"❌ Verkauf fehlgeschlagen: {result['reason']}")
            else:
                print(f"⏸️ Niedrige Risikotoleranz - kein Trade")
        
        # Finale Performance
        metrics = paper_system.get_performance_metrics()
        portfolio_value = paper_system.get_portfolio_value({'AAPL': 160.0})
        
        print(f"\n📈 FINALE PERFORMANCE:")
        print(f"   Portfolio Value: ${portfolio_value:,.2f}")
        print(f"   Total Return: {((portfolio_value - 10000) / 10000 * 100):.2f}%")
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Commission Paid: ${metrics['total_commission_paid']:.2f}")
        print(f"   Final Emotion: {emotion_engine.current_emotion.value}")
        
        print("\n✅ Emotion Engine + Paper Trading Integration Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Fehler bei Integration Test: {e}")
        return False

def run_standalone_comprehensive_test():
    """Führe umfassenden Standalone Test durch"""
    
    print("🚀 UMFASSENDER STANDALONE TRADING-SYSTEM TEST")
    print("=" * 60)
    
    test_results = {}
    
    # Teste alle Komponenten
    test_results['emotion_engine'] = test_standalone_emotion_engine()
    test_results['paper_trading'] = test_standalone_paper_trading()
    test_results['integration'] = test_emotion_paper_trading_integration()
    
    # Zusammenfassung
    print("\n📋 STANDALONE TEST ZUSAMMENFASSUNG")
    print("=" * 40)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for component, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{component}: {status}")
    
    print(f"\n🎯 GESAMTERGEBNIS: {passed_tests}/{total_tests} Tests bestanden")
    
    if passed_tests == total_tests:
        print("🎉 ALLE STANDALONE TESTS ERFOLGREICH!")
        print("🚀 Das Trading-System ist bereit für echte Tests!")
    else:
        print("⚠️ Einige Tests fehlgeschlagen. Überprüfe die Fehler.")
    
    return test_results

if __name__ == "__main__":
    run_standalone_comprehensive_test()
