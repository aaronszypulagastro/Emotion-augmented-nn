"""
Optimized Trading System V2
Mit den besten gefundenen Parametern
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Import unserer optimierten Komponenten
from optimized_trading_system import OptimizedTradingEmotionEngine, OptimizedPaperTradingSystem, TradingEmotion

class OptimizedTradingSystemV2:
    """Optimized Trading System V2 mit besten Parametern"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        
        # BESTE PARAMETER AUS OPTIMIERUNG
        self.emotion_engine = OptimizedTradingEmotionEngine(
            learning_rate=0.05,        # Beste Performance
            emotion_decay=0.995,       # Lange anhaltende Emotionen
            transition_threshold=0.6   # Aktive Transitions
        )
        
        self.paper_system = OptimizedPaperTradingSystem(
            initial_capital=initial_capital,
            commission=0.001,
            slippage=0.0005,
            min_trade_size=100.0,
            max_position_size=0.25
        )
        
        # Optimierte Trading-Parameter
        self.base_quantity = 3.0
        self.volatility_multiplier = 0.9
        self.trend_multiplier = 0.9
        self.emotion_intensity = 1.0
        
        # Risk Management
        self.max_risk_per_trade = 0.025  # 2.5%
        self.max_daily_loss = 0.05       # 5%
        self.target_daily_return = 0.015 # 1.5%
        
        # Performance Tracking
        self.session_start_time = datetime.now()
        self.daily_pnl = 0.0
        self.trading_enabled = True
        
    def run_optimized_session(self, steps: int = 300) -> Dict:
        """Führe optimierte Trading-Session durch"""
        
        print(f"🚀 OPTIMIZED TRADING SYSTEM V2")
        print(f"Zeit: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Steps: {steps}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Optimierte Parameter aktiviert!")
        
        # Generiere Marktdaten
        print(f"\n📊 Generiere Marktdaten...")
        market_data = self._generate_optimized_market_data(steps)
        
        # Optimized Trading Loop
        print(f"\n💹 Starte optimiertes Trading...")
        
        for step in range(steps):
            if step % 50 == 0:
                portfolio_value = self.paper_system.get_portfolio_value(self._get_current_prices())
                print(f"   Step {step}/{steps} - Portfolio: ${portfolio_value:,.2f} - Emotion: {self.emotion_engine.current_emotion.value}")
            
            # Prüfe Trading-Status
            if not self._check_optimized_trading_conditions():
                print(f"   ⏸️ Trading pausiert bei Step {step}")
                continue
            
            # Wähle Asset
            symbol = self._select_optimized_asset()
            market_info = market_data[symbol][step]
            
            # Update System
            self._update_optimized_system(market_info, symbol)
            
            # Trading-Entscheidung mit optimierten Parametern
            trading_decision = self._make_optimized_trading_decision(market_info, symbol)
            
            # Führe Trade aus
            if trading_decision['action'] != 'hold':
                self._execute_optimized_trade(trading_decision, market_info, symbol)
            
            # Update Monitoring
            self._update_optimized_monitoring()
        
        # Finale Ergebnisse
        final_results = self._calculate_optimized_results()
        
        print(f"\n✅ Optimierte Session abgeschlossen!")
        return final_results
    
    def _generate_optimized_market_data(self, steps: int) -> Dict[str, List[Dict]]:
        """Generiere optimierte Marktdaten"""
        
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
                    'volume': volume,
                    'timestamp': step
                })
        
        return market_data
    
    def _check_optimized_trading_conditions(self) -> bool:
        """Prüfe optimierte Trading-Bedingungen"""
        
        if not self.trading_enabled:
            return False
        
        # Prüfe Daily Loss Limit
        current_portfolio = self.paper_system.get_portfolio_value(self._get_current_prices())
        daily_return = (current_portfolio - self.initial_capital) / self.initial_capital
        
        if daily_return < -self.max_daily_loss:
            print(f"   🚨 Daily Loss Limit erreicht: {daily_return*100:.2f}%")
            self.trading_enabled = False
            return False
        
        # Prüfe Target Return
        if daily_return > self.target_daily_return:
            print(f"   🎯 Target Return erreicht: {daily_return*100:.2f}%")
            # Trading fortsetzen, aber konservativer
        
        return True
    
    def _select_optimized_asset(self) -> str:
        """Wähle Asset mit optimierter Logik"""
        
        assets = ['AAPL', 'TSLA', 'BTC/USD', 'ETH/USD']
        
        # Wähle basierend auf Emotion und Performance
        current_emotion = self.emotion_engine.current_emotion
        
        if current_emotion in [TradingEmotion.CONFIDENT, TradingEmotion.GREEDY]:
            # Risikoreichere Assets
            return np.random.choice(['TSLA', 'BTC/USD', 'ETH/USD'], p=[0.3, 0.4, 0.3])
        elif current_emotion in [TradingEmotion.FEARFUL, TradingEmotion.FRUSTRATED]:
            # Sicherere Assets
            return 'AAPL'
        else:
            # Balanced
            return np.random.choice(assets)
    
    def _update_optimized_system(self, market_info: Dict, symbol: str):
        """Update optimiertes System"""
        
        # Berechne Portfolio-Performance
        current_portfolio_value = self.paper_system.get_portfolio_value(self._get_current_prices())
        portfolio_return = (current_portfolio_value - self.initial_capital) / self.initial_capital
        
        # Berechne Trade-Performance
        total_trades = len(self.paper_system.trades)
        if total_trades > 0:
            winning_trades = len([t for t in self.paper_system.trades if t['action'] == 'sell'])
            win_rate = winning_trades / total_trades
        else:
            win_rate = 0.5
        
        # Update Emotion Engine mit optimierten Parametern
        self.emotion_engine.update_market_sentiment(
            price_change=market_info['price_change'],
            volume_change=market_info['volume'] / 5000 - 1,
            volatility=market_info['volatility'],
            trend_strength=market_info['price_change'] * 2
        )
        
        self.emotion_engine.update_performance(
            portfolio_return=portfolio_return,
            trade_return=market_info['price_change'],
            drawdown=self.paper_system.max_drawdown,
            win_rate=win_rate
        )
    
    def _make_optimized_trading_decision(self, market_info: Dict, symbol: str) -> Dict:
        """Optimierte Trading-Entscheidung"""
        
        enhanced_risk_tolerance = self.emotion_engine.get_enhanced_risk_tolerance()
        enhanced_position_modifier = self.emotion_engine.get_enhanced_position_sizing_modifier()
        current_emotion = self.emotion_engine.current_emotion
        
        # OPTIMIERTE PARAMETER
        base_quantity = self.base_quantity
        volatility_multiplier = self.volatility_multiplier
        trend_multiplier = self.trend_multiplier
        emotion_intensity = self.emotion_intensity
        
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
        
        # Bestimme Trade-Größe mit OPTIMIERTEN Parametern
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
            
            # Risk Management
            quantity = min(quantity, self.max_risk_per_trade * 10000 / market_info['price'])
            
            quantity = max(1.0, quantity)
        else:
            quantity = 0
        
        return {
            'action': action,
            'quantity': quantity,
            'emotion': current_emotion.value,
            'risk_tolerance': enhanced_risk_tolerance,
            'position_modifier': enhanced_position_modifier,
            'emotion_intensity': emotion_intensity,
            'optimized_params': {
                'base_quantity': base_quantity,
                'volatility_multiplier': volatility_multiplier,
                'trend_multiplier': trend_multiplier,
                'emotion_intensity': emotion_intensity
            }
        }
    
    def _execute_optimized_trade(self, decision: Dict, market_info: Dict, symbol: str):
        """Führe optimierten Trade aus"""
        
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
            print(f"   💹 {decision['action'].upper()}: {decision['quantity']:.1f} {symbol} @ ${market_info['price']:.2f}")
    
    def _update_optimized_monitoring(self):
        """Update optimiertes Monitoring"""
        
        current_portfolio = self.paper_system.get_portfolio_value(self._get_current_prices())
        self.daily_pnl = (current_portfolio - self.initial_capital) / self.initial_capital
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Hole aktuelle Preise"""
        return {'AAPL': 150.0, 'TSLA': 200.0, 'BTC/USD': 45000.0, 'ETH/USD': 3000.0}
    
    def _calculate_optimized_results(self) -> Dict:
        """Berechne optimierte Ergebnisse"""
        
        final_portfolio_value = self.paper_system.get_portfolio_value(self._get_current_prices())
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital
        
        # Performance Metrics
        total_trades = len(self.paper_system.trades)
        if total_trades > 0:
            winning_trades = len([t for t in self.paper_system.trades if t['action'] == 'sell'])
            win_rate = winning_trades / total_trades
        else:
            win_rate = 0.0
        
        # Risk Metrics
        max_drawdown = self.paper_system.max_drawdown
        daily_return = self.daily_pnl
        
        # Session Metrics
        session_duration = (datetime.now() - self.session_start_time).total_seconds() / 60  # Minuten
        
        return {
            'final_portfolio_value': final_portfolio_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'daily_return_pct': daily_return * 100,
            'total_trades': total_trades,
            'win_rate_pct': win_rate * 100,
            'max_drawdown_pct': max_drawdown * 100,
            'session_duration_min': session_duration,
            'final_emotion': self.emotion_engine.current_emotion.value,
            'emotion_intensity': self.emotion_engine.emotion_intensity,
            'trading_enabled': self.trading_enabled,
            'optimized_parameters': {
                'base_quantity': self.base_quantity,
                'volatility_multiplier': self.volatility_multiplier,
                'trend_multiplier': self.trend_multiplier,
                'emotion_intensity': self.emotion_intensity,
                'max_risk_per_trade': self.max_risk_per_trade,
                'max_daily_loss': self.max_daily_loss,
                'target_daily_return': self.target_daily_return
            }
        }

def run_optimized_trading_system_v2():
    """Führe Optimized Trading System V2 durch"""
    
    print("🚀 OPTIMIZED TRADING SYSTEM V2")
    print("=" * 60)
    
    # Erstelle optimiertes System
    optimized_system = OptimizedTradingSystemV2(initial_capital=10000.0)
    
    # Führe optimierte Session durch
    results = optimized_system.run_optimized_session(steps=300)
    
    # Zeige Ergebnisse
    print(f"\n📈 OPTIMIERTE SESSION ERGEBNISSE:")
    print(f"   Final Portfolio: ${results['final_portfolio_value']:,.2f}")
    print(f"   Total Return: {results['total_return_pct']:.2f}%")
    print(f"   Daily Return: {results['daily_return_pct']:.2f}%")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   Win Rate: {results['win_rate_pct']:.1f}%")
    print(f"   Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"   Session Duration: {results['session_duration_min']:.1f} min")
    print(f"   Final Emotion: {results['final_emotion']}")
    print(f"   Emotion Intensity: {results['emotion_intensity']:.2f}")
    print(f"   Trading Enabled: {results['trading_enabled']}")
    
    # Optimierte Parameter
    opt_params = results['optimized_parameters']
    print(f"\n🔧 OPTIMIERTE PARAMETER:")
    print(f"   Base Quantity: {opt_params['base_quantity']}")
    print(f"   Volatility Multiplier: {opt_params['volatility_multiplier']}")
    print(f"   Trend Multiplier: {opt_params['trend_multiplier']}")
    print(f"   Emotion Intensity: {opt_params['emotion_intensity']}")
    print(f"   Max Risk per Trade: {opt_params['max_risk_per_trade']*100:.1f}%")
    print(f"   Max Daily Loss: {opt_params['max_daily_loss']*100:.1f}%")
    print(f"   Target Daily Return: {opt_params['target_daily_return']*100:.1f}%")
    
    return results

if __name__ == "__main__":
    run_optimized_trading_system_v2()
