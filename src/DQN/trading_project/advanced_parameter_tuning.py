"""
Advanced Parameter Tuning System
Optimiert alle Parameter für maximale Performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
import json
from datetime import datetime
import itertools
warnings.filterwarnings('ignore')

# Import unserer optimierten Komponenten
from optimized_trading_system import OptimizedTradingEmotionEngine, OptimizedPaperTradingSystem, TradingEmotion

class AdvancedParameterTuner:
    """Advanced Parameter Tuning System"""
    
    def __init__(self):
        self.parameter_ranges = {
            # Emotion Engine Parameters
            'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05],
            'emotion_decay': [0.95, 0.97, 0.99, 0.995, 0.999],
            'transition_threshold': [0.3, 0.4, 0.5, 0.6, 0.7],
            'emotion_intensity': [0.8, 0.9, 1.0, 1.1, 1.2],
            
            # Trading Parameters
            'base_quantity': [3.0, 4.0, 5.0, 6.0, 7.0],
            'volatility_multiplier': [0.3, 0.5, 0.7, 0.9, 1.1],
            'trend_multiplier': [0.5, 0.7, 0.9, 1.1, 1.3],
            
            # Risk Parameters
            'max_risk_per_trade': [0.01, 0.015, 0.02, 0.025, 0.03],
            'max_daily_loss': [0.03, 0.04, 0.05, 0.06, 0.07],
            'target_daily_return': [0.005, 0.01, 0.015, 0.02, 0.025]
        }
        
        self.best_parameters = {}
        self.optimization_results = []
        
    def run_parameter_optimization(self, num_tests: int = 50) -> Dict:
        """Führe Parameter-Optimierung durch"""
        
        print(f"🔧 ADVANCED PARAMETER TUNING GESTARTET")
        print(f"Anzahl Tests: {num_tests}")
        print(f"Parameter-Kombinationen: {len(list(itertools.product(*self.parameter_ranges.values())))}")
        
        # Generiere zufällige Parameter-Kombinationen
        parameter_combinations = self._generate_parameter_combinations(num_tests)
        
        best_performance = -float('inf')
        best_params = None
        
        for i, params in enumerate(parameter_combinations):
            print(f"\n🧪 Test {i+1}/{num_tests}")
            print(f"Parameter: {params}")
            
            # Führe Test mit diesen Parametern durch
            performance = self._test_parameter_combination(params)
            
            # Speichere Ergebnis
            result = {
                'test_id': i+1,
                'parameters': params,
                'performance': performance
            }
            self.optimization_results.append(result)
            
            # Prüfe ob beste Performance
            if performance['total_return'] > best_performance:
                best_performance = performance['total_return']
                best_params = params
                print(f"   🎯 NEUE BESTE PERFORMANCE: {performance['total_return']*100:.2f}%")
            else:
                print(f"   📊 Performance: {performance['total_return']*100:.2f}%")
        
        # Speichere beste Parameter
        self.best_parameters = best_params
        
        print(f"\n✅ PARAMETER OPTIMIERUNG ABGESCHLOSSEN")
        print(f"Beste Performance: {best_performance*100:.2f}%")
        print(f"Beste Parameter: {best_params}")
        
        return {
            'best_parameters': best_params,
            'best_performance': best_performance,
            'all_results': self.optimization_results
        }
    
    def _generate_parameter_combinations(self, num_tests: int) -> List[Dict]:
        """Generiere Parameter-Kombinationen"""
        
        combinations = []
        
        for _ in range(num_tests):
            params = {}
            for param_name, param_range in self.parameter_ranges.items():
                params[param_name] = np.random.choice(param_range)
            combinations.append(params)
        
        return combinations
    
    def _test_parameter_combination(self, params: Dict) -> Dict:
        """Teste Parameter-Kombination"""
        
        # Erstelle System mit diesen Parametern
        emotion_engine = OptimizedTradingEmotionEngine(
            learning_rate=params['learning_rate'],
            emotion_decay=params['emotion_decay'],
            transition_threshold=params['transition_threshold']
        )
        
        paper_system = OptimizedPaperTradingSystem(
            initial_capital=10000.0,
            commission=0.001,
            slippage=0.0005,
            min_trade_size=100.0,
            max_position_size=0.25
        )
        
        # Führe Trading-Session durch
        performance = self._run_trading_session(emotion_engine, paper_system, params)
        
        return performance
    
    def _run_trading_session(self, emotion_engine, paper_system, params: Dict) -> Dict:
        """Führe Trading-Session durch"""
        
        # Generiere Marktdaten
        market_data = self._generate_optimization_market_data(100)
        
        # Trading Loop
        for step in range(100):
            # Wähle Asset
            symbol = np.random.choice(['AAPL', 'TSLA', 'BTC/USD', 'ETH/USD'])
            market_info = market_data[symbol][step]
            
            # Update System
            self._update_optimization_system(emotion_engine, paper_system, market_info, symbol)
            
            # Trading-Entscheidung
            trading_decision = self._make_optimization_trading_decision(
                emotion_engine, market_info, symbol, params
            )
            
            # Führe Trade aus
            if trading_decision['action'] != 'hold':
                self._execute_optimization_trade(
                    paper_system, trading_decision, market_info, symbol
                )
        
        # Berechne Performance
        final_portfolio_value = paper_system.get_portfolio_value(self._get_current_prices())
        total_return = (final_portfolio_value - 10000.0) / 10000.0
        
        # Berechne zusätzliche Metriken
        total_trades = len(paper_system.trades)
        if total_trades > 0:
            winning_trades = len([t for t in paper_system.trades if t['action'] == 'sell'])
            win_rate = winning_trades / total_trades
        else:
            win_rate = 0.0
        
        max_drawdown = paper_system.max_drawdown
        
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'final_portfolio_value': final_portfolio_value
        }
    
    def _generate_optimization_market_data(self, steps: int) -> Dict[str, List[Dict]]:
        """Generiere Marktdaten für Optimierung"""
        
        assets = ['AAPL', 'TSLA', 'BTC/USD', 'ETH/USD']
        market_data = {}
        
        for symbol in assets:
            market_data[symbol] = []
            
            # Asset-spezifische Parameter
            base_price = {'AAPL': 150.0, 'TSLA': 200.0, 'BTC/USD': 45000.0, 'ETH/USD': 3000.0}[symbol]
            volatility = {'AAPL': 0.02, 'TSLA': 0.04, 'BTC/USD': 0.05, 'ETH/USD': 0.06}[symbol]
            trend = {'AAPL': 0.001, 'TSLA': 0.002, 'BTC/USD': 0.003, 'ETH/USD': 0.004}[symbol]
            
            current_price = base_price
            
            for step in range(steps):
                # Generiere realistische Preis-Bewegung
                trend_component = trend * np.random.normal(0, 1)
                volatility_component = volatility * np.random.normal(0, 1)
                
                price_change = trend_component + volatility_component
                current_price *= (1 + price_change)
                
                # Berechne Metriken
                volume = np.random.uniform(1000, 10000)
                
                market_data[symbol].append({
                    'step': step,
                    'price': current_price,
                    'price_change': price_change,
                    'volatility': volatility,
                    'volume': volume
                })
        
        return market_data
    
    def _update_optimization_system(self, emotion_engine, paper_system, market_info: Dict, symbol: str):
        """Update System für Optimierung"""
        
        # Berechne Portfolio-Performance
        current_portfolio_value = paper_system.get_portfolio_value(self._get_current_prices())
        portfolio_return = (current_portfolio_value - 10000.0) / 10000.0
        
        # Berechne Trade-Performance
        total_trades = len(paper_system.trades)
        if total_trades > 0:
            winning_trades = len([t for t in paper_system.trades if t['action'] == 'sell'])
            win_rate = winning_trades / total_trades
        else:
            win_rate = 0.5
        
        # Update Emotion Engine
        emotion_engine.update_market_sentiment(
            price_change=market_info['price_change'],
            volume_change=market_info['volume'] / 5000 - 1,
            volatility=market_info['volatility'],
            trend_strength=market_info['price_change'] * 2
        )
        
        emotion_engine.update_performance(
            portfolio_return=portfolio_return,
            trade_return=market_info['price_change'],
            drawdown=paper_system.max_drawdown,
            win_rate=win_rate
        )
    
    def _make_optimization_trading_decision(self, emotion_engine, market_info: Dict, symbol: str, params: Dict) -> Dict:
        """Trading-Entscheidung für Optimierung"""
        
        enhanced_risk_tolerance = emotion_engine.get_enhanced_risk_tolerance()
        enhanced_position_modifier = emotion_engine.get_enhanced_position_sizing_modifier()
        current_emotion = emotion_engine.current_emotion
        emotion_intensity = emotion_engine.emotion_intensity
        
        # Verwende optimierte Parameter
        base_quantity = params['base_quantity']
        volatility_multiplier = params['volatility_multiplier']
        trend_multiplier = params['trend_multiplier']
        
        # Trading-Logik
        price_change = market_info['price_change']
        volatility = market_info['volatility']
        
        # Emotion-basierte Trading-Logik
        if current_emotion in [TradingEmotion.CONFIDENT, TradingEmotion.GREEDY, TradingEmotion.OPTIMISTIC]:
            if price_change > -0.003 and volatility < 0.04:
                action = 'buy'
            else:
                action = 'hold'
        elif current_emotion in [TradingEmotion.FEARFUL, TradingEmotion.FRUSTRATED, TradingEmotion.PESSIMISTIC]:
            if price_change < 0.003 or volatility > 0.03:
                action = 'sell'
            else:
                action = 'hold'
        else:
            if price_change > 0.003:
                action = 'buy'
            elif price_change < -0.003:
                action = 'sell'
            else:
                action = 'hold'
        
        # Bestimme Trade-Größe mit optimierten Parametern
        if action != 'hold':
            quantity = base_quantity * enhanced_position_modifier * enhanced_risk_tolerance
            
            # Volatilitäts-Anpassung mit optimiertem Multiplier
            if volatility > 0.05:
                quantity *= volatility_multiplier
            elif volatility < 0.02:
                quantity *= (2 - volatility_multiplier)
            
            # Trend-Anpassung mit optimiertem Multiplier
            if price_change > 0:
                quantity *= trend_multiplier
            else:
                quantity *= (2 - trend_multiplier)
            
            # Emotion-Intensität Anpassung
            quantity *= emotion_intensity
            
            quantity = max(1.0, quantity)
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
    
    def _execute_optimization_trade(self, paper_system, decision: Dict, market_info: Dict, symbol: str):
        """Führe Trade für Optimierung aus"""
        
        if decision['action'] == 'hold':
            return
        
        # Führe Trade aus
        paper_system.execute_trade(
            symbol=symbol,
            action=decision['action'],
            quantity=decision['quantity'],
            price=market_info['price']
        )
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Hole aktuelle Preise"""
        return {'AAPL': 150.0, 'TSLA': 200.0, 'BTC/USD': 45000.0, 'ETH/USD': 3000.0}
    
    def save_optimization_results(self, filename: str = 'optimization_results.json'):
        """Speichere Optimierungsergebnisse"""
        
        results = {
            'best_parameters': self.best_parameters,
            'optimization_results': self.optimization_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"📁 Optimierungsergebnisse gespeichert: {filename}")
    
    def load_optimization_results(self, filename: str = 'optimization_results.json') -> Dict:
        """Lade Optimierungsergebnisse"""
        
        try:
            with open(filename, 'r') as f:
                results = json.load(f)
            
            self.best_parameters = results['best_parameters']
            self.optimization_results = results['optimization_results']
            
            print(f"📁 Optimierungsergebnisse geladen: {filename}")
            return results
        except FileNotFoundError:
            print(f"❌ Datei nicht gefunden: {filename}")
            return {}

def run_advanced_parameter_tuning():
    """Führe Advanced Parameter Tuning durch"""
    
    print("🔧 ADVANCED PARAMETER TUNING SYSTEM")
    print("=" * 60)
    
    # Erstelle Parameter Tuner
    tuner = AdvancedParameterTuner()
    
    # Führe Optimierung durch
    results = tuner.run_parameter_optimization(num_tests=30)
    
    # Zeige Ergebnisse
    print(f"\n📈 OPTIMIERUNGSERGEBNISSE:")
    print(f"   Beste Performance: {results['best_performance']*100:.2f}%")
    print(f"   Beste Parameter:")
    for param, value in results['best_parameters'].items():
        print(f"     {param}: {value}")
    
    # Speichere Ergebnisse
    tuner.save_optimization_results()
    
    # Zeige Top 5 Ergebnisse
    print(f"\n🏆 TOP 5 PERFORMANCE:")
    sorted_results = sorted(results['all_results'], key=lambda x: x['performance']['total_return'], reverse=True)
    
    for i, result in enumerate(sorted_results[:5]):
        perf = result['performance']
        print(f"   {i+1}. Return: {perf['total_return']*100:.2f}% | Trades: {perf['total_trades']} | Win Rate: {perf['win_rate']*100:.1f}%")
    
    return results

if __name__ == "__main__":
    run_advanced_parameter_tuning()
