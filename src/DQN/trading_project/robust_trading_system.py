"""
Robust Trading System
Mit verbessertem Risk Management und Recovery-Mechanismen
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

class RobustTradingSystem:
    """Robust Trading System mit verbessertem Risk Management"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        
        # KONSERVATIVE PARAMETER für Robustheit
        self.emotion_engine = OptimizedTradingEmotionEngine(
            learning_rate=0.02,        # Konservativer
            emotion_decay=0.98,        # Schnellere Anpassung
            transition_threshold=0.7   # Weniger aktive Transitions
        )
        
        self.paper_system = OptimizedPaperTradingSystem(
            initial_capital=initial_capital,
            commission=0.001,
            slippage=0.0005,
            min_trade_size=100.0,
            max_position_size=0.15  # Konservativer
        )
        
        # ROBUSTE Trading-Parameter
        self.base_quantity = 2.0  # Konservativer
        self.volatility_multiplier = 0.7  # Weniger aggressiv
        self.trend_multiplier = 0.8  # Konservativer
        self.emotion_intensity = 0.8  # Weniger intensiv
        
        # VERBESSERTES Risk Management
        self.max_risk_per_trade = 0.015  # 1.5% (konservativer)
        self.max_daily_loss = 0.08       # 8% (höher für Recovery)
        self.target_daily_return = 0.008 # 0.8% (realistischer)
        self.stop_loss_threshold = 0.03  # 3% Stop-Loss
        self.recovery_threshold = 0.02   # 2% Recovery-Trigger
        
        # Recovery System
        self.recovery_mode = False
        self.recovery_trades = 0
        self.max_recovery_trades = 5
        
        # Performance Tracking
        self.session_start_time = datetime.now()
        self.daily_pnl = 0.0
        self.trading_enabled = True
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        
    def run_robust_session(self, steps: int = 300) -> Dict:
        """Führe robuste Trading-Session durch"""
        
        print(f"🛡️ ROBUST TRADING SYSTEM")
        print(f"Zeit: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Steps: {steps}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Robuste Parameter aktiviert!")
        print(f"Max Risk per Trade: {self.max_risk_per_trade*100:.1f}%")
        print(f"Max Daily Loss: {self.max_daily_loss*100:.1f}%")
        print(f"Stop-Loss: {self.stop_loss_threshold*100:.1f}%")
        
        # Generiere Marktdaten
        print(f"\n📊 Generiere Marktdaten...")
        market_data = self._generate_robust_market_data(steps)
        
        # Robust Trading Loop
        print(f"\n💹 Starte robustes Trading...")
        
        for step in range(steps):
            if step % 50 == 0:
                portfolio_value = self.paper_system.get_portfolio_value(self._get_current_prices())
                print(f"   Step {step}/{steps} - Portfolio: ${portfolio_value:,.2f} - Emotion: {self.emotion_engine.current_emotion.value} - Recovery: {self.recovery_mode}")
            
            # Prüfe Trading-Status
            if not self._check_robust_trading_conditions():
                print(f"   ⏸️ Trading pausiert bei Step {step}")
                continue
            
            # Wähle Asset
            symbol = self._select_robust_asset()
            market_info = market_data[symbol][step]
            
            # Update System
            self._update_robust_system(market_info, symbol)
            
            # Trading-Entscheidung mit robusten Parametern
            trading_decision = self._make_robust_trading_decision(market_info, symbol)
            
            # Führe Trade aus
            if trading_decision['action'] != 'hold':
                self._execute_robust_trade(trading_decision, market_info, symbol)
            
            # Update Monitoring
            self._update_robust_monitoring()
        
        # Finale Ergebnisse
        final_results = self._calculate_robust_results()
        
        print(f"\n✅ Robuste Session abgeschlossen!")
        return final_results
    
    def _generate_robust_market_data(self, steps: int) -> Dict[str, List[Dict]]:
        """Generiere robuste Marktdaten"""
        
        assets = ['AAPL', 'TSLA', 'BTC/USD', 'ETH/USD']
        market_data = {}
        
        for symbol in assets:
            market_data[symbol] = []
            
            # Asset-spezifische Parameter
            base_price = {'AAPL': 150.0, 'TSLA': 200.0, 'BTC/USD': 45000.0, 'ETH/USD': 3000.0}[symbol]
            volatility = {'AAPL': 0.015, 'TSLA': 0.03, 'BTC/USD': 0.04, 'ETH/USD': 0.05}[symbol]  # Weniger volatil
            trend = {'AAPL': 0.0005, 'TSLA': 0.001, 'BTC/USD': 0.002, 'ETH/USD': 0.003}[symbol]  # Weniger Trend
            
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
    
    def _check_robust_trading_conditions(self) -> bool:
        """Prüfe robuste Trading-Bedingungen"""
        
        if not self.trading_enabled:
            return False
        
        # Prüfe Daily Loss Limit
        current_portfolio = self.paper_system.get_portfolio_value(self._get_current_prices())
        daily_return = (current_portfolio - self.initial_capital) / self.initial_capital
        
        if daily_return < -self.max_daily_loss:
            print(f"   🚨 Daily Loss Limit erreicht: {daily_return*100:.2f}%")
            self.trading_enabled = False
            return False
        
        # Prüfe Stop-Loss
        if daily_return < -self.stop_loss_threshold:
            print(f"   🛑 Stop-Loss erreicht: {daily_return*100:.2f}%")
            self.trading_enabled = False
            return False
        
        # Prüfe Consecutive Losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            print(f"   ⚠️ Zu viele aufeinanderfolgende Verluste: {self.consecutive_losses}")
            self.trading_enabled = False
            return False
        
        # Prüfe Recovery Mode
        if self.recovery_mode and self.recovery_trades >= self.max_recovery_trades:
            print(f"   🔄 Recovery Mode beendet: {self.recovery_trades} Trades")
            self.recovery_mode = False
            self.recovery_trades = 0
        
        # Prüfe Target Return
        if daily_return > self.target_daily_return:
            print(f"   🎯 Target Return erreicht: {daily_return*100:.2f}%")
            # Trading fortsetzen, aber konservativer
        
        return True
    
    def _select_robust_asset(self) -> str:
        """Wähle Asset mit robuster Logik"""
        
        assets = ['AAPL', 'TSLA', 'BTC/USD', 'ETH/USD']
        
        # In Recovery Mode: Nur sichere Assets
        if self.recovery_mode:
            return 'AAPL'  # Sicherstes Asset
        
        # Wähle basierend auf Emotion und Performance
        current_emotion = self.emotion_engine.current_emotion
        
        if current_emotion in [TradingEmotion.CONFIDENT, TradingEmotion.GREEDY]:
            # Risikoreichere Assets
            return np.random.choice(['TSLA', 'BTC/USD', 'ETH/USD'], p=[0.4, 0.3, 0.3])
        elif current_emotion in [TradingEmotion.FEARFUL, TradingEmotion.FRUSTRATED]:
            # Sicherere Assets
            return 'AAPL'
        else:
            # Balanced
            return np.random.choice(assets, p=[0.4, 0.2, 0.2, 0.2])  # AAPL bevorzugt
    
    def _update_robust_system(self, market_info: Dict, symbol: str):
        """Update robustes System"""
        
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
        
        # Update Emotion Engine mit robusten Parametern
        self.emotion_engine.update_market_sentiment(
            price_change=market_info['price_change'],
            volume_change=market_info['volume'] / 5000 - 1,
            volatility=market_info['volatility'],
            trend_strength=market_info['price_change'] * 1.5  # Weniger intensiv
        )
        
        self.emotion_engine.update_performance(
            portfolio_return=portfolio_return,
            trade_return=market_info['price_change'],
            drawdown=self.paper_system.max_drawdown,
            win_rate=win_rate
        )
        
        # Recovery Mode Logic
        if portfolio_return < -self.recovery_threshold and not self.recovery_mode:
            self.recovery_mode = True
            print(f"   🔄 Recovery Mode aktiviert bei {portfolio_return*100:.2f}%")
        elif portfolio_return > 0 and self.recovery_mode:
            self.recovery_mode = False
            self.recovery_trades = 0
            print(f"   ✅ Recovery Mode beendet bei {portfolio_return*100:.2f}%")
    
    def _make_robust_trading_decision(self, market_info: Dict, symbol: str) -> Dict:
        """Robuste Trading-Entscheidung"""
        
        enhanced_risk_tolerance = self.emotion_engine.get_enhanced_risk_tolerance()
        enhanced_position_modifier = self.emotion_engine.get_enhanced_position_sizing_modifier()
        current_emotion = self.emotion_engine.current_emotion
        
        # ROBUSTE PARAMETER
        base_quantity = self.base_quantity
        volatility_multiplier = self.volatility_multiplier
        trend_multiplier = self.trend_multiplier
        emotion_intensity = self.emotion_intensity
        
        # Recovery Mode: Konservativer
        if self.recovery_mode:
            base_quantity *= 0.5
            enhanced_risk_tolerance *= 0.5
            enhanced_position_modifier *= 0.5
        
        # Trading-Logik
        price_change = market_info['price_change']
        volatility = market_info['volatility']
        
        # Emotion-basierte Trading-Logik (konservativer)
        if current_emotion in [TradingEmotion.CONFIDENT, TradingEmotion.GREEDY]:
            if price_change > -0.002 and volatility < 0.03:  # Strengere Bedingungen
                action = 'buy'
            else:
                action = 'hold'
        elif current_emotion in [TradingEmotion.FEARFUL, TradingEmotion.FRUSTRATED]:
            if price_change < 0.002 or volatility > 0.025:  # Strengere Bedingungen
                action = 'sell'
            else:
                action = 'hold'
        else:
            if price_change > 0.002:  # Strengere Bedingungen
                action = 'buy'
            elif price_change < -0.002:  # Strengere Bedingungen
                action = 'sell'
            else:
                action = 'hold'
        
        # Bestimme Trade-Größe mit ROBUSTEN Parametern
        if action != 'hold':
            quantity = base_quantity * enhanced_position_modifier * enhanced_risk_tolerance
            
            # Volatilitäts-Anpassung mit robustem Multiplier
            if volatility > 0.04:
                quantity *= volatility_multiplier
            elif volatility < 0.015:
                quantity *= (2 - volatility_multiplier)
            
            # Trend-Anpassung mit robustem Multiplier
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
            'recovery_mode': self.recovery_mode,
            'robust_params': {
                'base_quantity': base_quantity,
                'volatility_multiplier': volatility_multiplier,
                'trend_multiplier': trend_multiplier,
                'emotion_intensity': emotion_intensity
            }
        }
    
    def _execute_robust_trade(self, decision: Dict, market_info: Dict, symbol: str):
        """Führe robusten Trade aus"""
        
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
            
            # Update Recovery Trades
            if self.recovery_mode:
                self.recovery_trades += 1
            
            # Update Consecutive Losses
            if decision['action'] == 'sell':
                self.consecutive_losses = 0  # Reset bei Verkauf
            else:
                self.consecutive_losses += 1  # Increment bei Kauf
    
    def _update_robust_monitoring(self):
        """Update robustes Monitoring"""
        
        current_portfolio = self.paper_system.get_portfolio_value(self._get_current_prices())
        self.daily_pnl = (current_portfolio - self.initial_capital) / self.initial_capital
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Hole aktuelle Preise"""
        return {'AAPL': 150.0, 'TSLA': 200.0, 'BTC/USD': 45000.0, 'ETH/USD': 3000.0}
    
    def _calculate_robust_results(self) -> Dict:
        """Berechne robuste Ergebnisse"""
        
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
            'recovery_mode': self.recovery_mode,
            'recovery_trades': self.recovery_trades,
            'consecutive_losses': self.consecutive_losses,
            'robust_parameters': {
                'base_quantity': self.base_quantity,
                'volatility_multiplier': self.volatility_multiplier,
                'trend_multiplier': self.trend_multiplier,
                'emotion_intensity': self.emotion_intensity,
                'max_risk_per_trade': self.max_risk_per_trade,
                'max_daily_loss': self.max_daily_loss,
                'target_daily_return': self.target_daily_return,
                'stop_loss_threshold': self.stop_loss_threshold,
                'recovery_threshold': self.recovery_threshold
            }
        }

def run_robust_trading_system():
    """Führe Robust Trading System durch"""
    
    print("🛡️ ROBUST TRADING SYSTEM")
    print("=" * 60)
    
    # Erstelle robustes System
    robust_system = RobustTradingSystem(initial_capital=10000.0)
    
    # Führe robuste Session durch
    results = robust_system.run_robust_session(steps=300)
    
    # Zeige Ergebnisse
    print(f"\n📈 ROBUSTE SESSION ERGEBNISSE:")
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
    print(f"   Recovery Mode: {results['recovery_mode']}")
    print(f"   Recovery Trades: {results['recovery_trades']}")
    print(f"   Consecutive Losses: {results['consecutive_losses']}")
    
    # Robuste Parameter
    robust_params = results['robust_parameters']
    print(f"\n🛡️ ROBUSTE PARAMETER:")
    print(f"   Base Quantity: {robust_params['base_quantity']}")
    print(f"   Volatility Multiplier: {robust_params['volatility_multiplier']}")
    print(f"   Trend Multiplier: {robust_params['trend_multiplier']}")
    print(f"   Emotion Intensity: {robust_params['emotion_intensity']}")
    print(f"   Max Risk per Trade: {robust_params['max_risk_per_trade']*100:.1f}%")
    print(f"   Max Daily Loss: {robust_params['max_daily_loss']*100:.1f}%")
    print(f"   Target Daily Return: {robust_params['target_daily_return']*100:.1f}%")
    print(f"   Stop-Loss: {robust_params['stop_loss_threshold']*100:.1f}%")
    print(f"   Recovery Threshold: {robust_params['recovery_threshold']*100:.1f}%")
    
    return results

if __name__ == "__main__":
    run_robust_trading_system()
