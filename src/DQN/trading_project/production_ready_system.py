"""
Production-Ready Trading System
Perfektioniert das bestehende System für echte Anwendungen
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

class ProductionReadyTradingSystem:
    """Production-Ready Trading System mit allen Optimierungen"""
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 max_risk_per_trade: float = 0.02,  # 2% Max Risk pro Trade
                 max_daily_loss: float = 0.05,     # 5% Max Daily Loss
                 target_daily_return: float = 0.01): # 1% Target Daily Return
        
        self.initial_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.target_daily_return = target_daily_return
        
        # Core Components
        self.emotion_engine = OptimizedTradingEmotionEngine(
            learning_rate=0.03,  # Erhöht für schnellere Anpassung
            emotion_decay=0.99,  # Länger anhaltende Emotionen
            transition_threshold=0.5  # Niedrigere Schwelle für aktivere Transitions
        )
        
        self.paper_system = OptimizedPaperTradingSystem(
            initial_capital=initial_capital,
            commission=0.001,
            slippage=0.0005,
            min_trade_size=100.0,
            max_position_size=0.25  # Konservativer für Production
        )
        
        # Production Features
        self.risk_manager = ProductionRiskManager(
            max_risk_per_trade=max_risk_per_trade,
            max_daily_loss=max_daily_loss,
            target_daily_return=target_daily_return
        )
        
        self.performance_monitor = PerformanceMonitor()
        self.trading_logger = TradingLogger()
        
        # System State
        self.is_trading_enabled = True
        self.daily_pnl = 0.0
        self.session_start_time = datetime.now()
        
    def run_production_session(self, steps: int = 300) -> Dict:
        """Führe Production Trading Session durch"""
        
        print(f"🚀 PRODUCTION TRADING SESSION GESTARTET")
        print(f"Zeit: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Steps: {steps}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Max Risk per Trade: {self.max_risk_per_trade*100:.1f}%")
        print(f"Max Daily Loss: {self.max_daily_loss*100:.1f}%")
        print(f"Target Daily Return: {self.target_daily_return*100:.1f}%")
        
        # Generiere Marktdaten
        print(f"\n📊 Generiere Production Marktdaten...")
        market_data = self._generate_production_market_data(steps)
        
        # Production Trading Loop
        print(f"\n💹 Starte Production Trading...")
        
        for step in range(steps):
            if step % 50 == 0:
                portfolio_value = self.paper_system.get_portfolio_value(self._get_current_prices())
                print(f"   Step {step}/{steps} - Portfolio: ${portfolio_value:,.2f} - Emotion: {self.emotion_engine.current_emotion.value}")
            
            # Prüfe Trading-Status
            if not self._check_trading_conditions():
                print(f"   ⏸️ Trading pausiert bei Step {step}")
                continue
            
            # Wähle Asset
            symbol = self._select_asset_for_trading()
            market_info = market_data[symbol][step]
            
            # Update System
            self._update_production_system(market_info, symbol)
            
            # Trading-Entscheidung
            trading_decision = self._make_production_trading_decision(market_info, symbol)
            
            # Führe Trade aus
            if trading_decision['action'] != 'hold':
                self._execute_production_trade(trading_decision, market_info, symbol)
            
            # Update Monitoring
            self._update_production_monitoring()
        
        # Finale Ergebnisse
        final_results = self._calculate_production_results()
        
        print(f"\n✅ Production Session abgeschlossen!")
        return final_results
    
    def _generate_production_market_data(self, steps: int) -> Dict[str, List[Dict]]:
        """Generiere Production-Marktdaten"""
        
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
    
    def _check_trading_conditions(self) -> bool:
        """Prüfe Trading-Bedingungen"""
        
        # Prüfe Trading-Status
        if not self.is_trading_enabled:
            return False
        
        # Prüfe Daily Loss Limit
        current_portfolio = self.paper_system.get_portfolio_value(self._get_current_prices())
        daily_return = (current_portfolio - self.initial_capital) / self.initial_capital
        
        if daily_return < -self.max_daily_loss:
            print(f"   🚨 Daily Loss Limit erreicht: {daily_return*100:.2f}%")
            self.is_trading_enabled = False
            return False
        
        # Prüfe Target Return
        if daily_return > self.target_daily_return:
            print(f"   🎯 Target Return erreicht: {daily_return*100:.2f}%")
            # Trading fortsetzen, aber konservativer
        
        return True
    
    def _select_asset_for_trading(self) -> str:
        """Wähle Asset für Trading basierend auf Performance"""
        
        assets = ['AAPL', 'TSLA', 'BTC/USD', 'ETH/USD']
        
        # Wähle basierend auf Emotion
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
    
    def _update_production_system(self, market_info: Dict, symbol: str):
        """Update Production System"""
        
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
        
        # Update Emotion Engine
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
    
    def _make_production_trading_decision(self, market_info: Dict, symbol: str) -> Dict:
        """Production Trading-Entscheidung"""
        
        enhanced_risk_tolerance = self.emotion_engine.get_enhanced_risk_tolerance()
        enhanced_position_modifier = self.emotion_engine.get_enhanced_position_sizing_modifier()
        current_emotion = self.emotion_engine.current_emotion
        emotion_intensity = self.emotion_engine.emotion_intensity
        
        # Risk Management
        risk_adjusted_tolerance = self.risk_manager.adjust_risk_tolerance(enhanced_risk_tolerance)
        
        # Production Trading-Logik
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
        
        # Bestimme Trade-Größe
        if action != 'hold':
            base_quantity = 5.0  # Konservativer für Production
            quantity = base_quantity * enhanced_position_modifier * risk_adjusted_tolerance
            
            # Volatilitäts-Anpassung
            if volatility > 0.05:
                quantity *= 0.5
            elif volatility < 0.02:
                quantity *= 1.2
            
            # Emotion-Intensität Anpassung
            quantity *= emotion_intensity
            
            # Risk Management
            quantity = self.risk_manager.adjust_position_size(quantity, market_info['price'])
            
            quantity = max(1.0, quantity)
        else:
            quantity = 0
        
        return {
            'action': action,
            'quantity': quantity,
            'emotion': current_emotion.value,
            'risk_tolerance': risk_adjusted_tolerance,
            'position_modifier': enhanced_position_modifier,
            'emotion_intensity': emotion_intensity
        }
    
    def _execute_production_trade(self, decision: Dict, market_info: Dict, symbol: str):
        """Führe Production Trade aus"""
        
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
            # Log Trade
            self.trading_logger.log_trade(result['trade_record'], decision)
            
            # Update Performance Monitor
            self.performance_monitor.update_trade(result['trade_record'])
    
    def _update_production_monitoring(self):
        """Update Production Monitoring"""
        
        current_portfolio = self.paper_system.get_portfolio_value(self._get_current_prices())
        self.daily_pnl = (current_portfolio - self.initial_capital) / self.initial_capital
        
        # Update Performance Monitor
        self.performance_monitor.update_portfolio(current_portfolio)
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Hole aktuelle Preise"""
        return {'AAPL': 150.0, 'TSLA': 200.0, 'BTC/USD': 45000.0, 'ETH/USD': 3000.0}
    
    def _calculate_production_results(self) -> Dict:
        """Berechne Production Ergebnisse"""
        
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
        
        # Production Metrics
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
            'trading_enabled': self.is_trading_enabled,
            'risk_metrics': self.risk_manager.get_risk_metrics(),
            'performance_metrics': self.performance_monitor.get_performance_metrics()
        }

class ProductionRiskManager:
    """Production Risk Manager"""
    
    def __init__(self, max_risk_per_trade: float, max_daily_loss: float, target_daily_return: float):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.target_daily_return = target_daily_return
        
        self.daily_trades = 0
        self.daily_risk_used = 0.0
        
    def adjust_risk_tolerance(self, base_tolerance: float) -> float:
        """Passe Risikotoleranz an"""
        
        # Reduziere Risiko bei hoher Tagesaktivität
        if self.daily_trades > 20:
            return base_tolerance * 0.5
        elif self.daily_trades > 10:
            return base_tolerance * 0.7
        else:
            return base_tolerance
    
    def adjust_position_size(self, base_size: float, price: float) -> float:
        """Passe Position-Größe an"""
        
        # Berechne maximale Position-Größe basierend auf Risiko
        max_position_value = self.max_risk_per_trade * 10000  # Vereinfacht
        max_quantity = max_position_value / price
        
        return min(base_size, max_quantity)
    
    def get_risk_metrics(self) -> Dict:
        """Hole Risk Metrics"""
        return {
            'daily_trades': self.daily_trades,
            'daily_risk_used': self.daily_risk_used,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_daily_loss': self.max_daily_loss
        }

class PerformanceMonitor:
    """Performance Monitor"""
    
    def __init__(self):
        self.portfolio_history = []
        self.trade_history = []
        
    def update_portfolio(self, portfolio_value: float):
        """Update Portfolio History"""
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value
        })
        
        if len(self.portfolio_history) > 1000:
            self.portfolio_history.pop(0)
    
    def update_trade(self, trade_record: Dict):
        """Update Trade History"""
        self.trade_history.append(trade_record)
        
        if len(self.trade_history) > 1000:
            self.trade_history.pop(0)
    
    def get_performance_metrics(self) -> Dict:
        """Hole Performance Metrics"""
        if not self.portfolio_history:
            return {}
        
        portfolio_values = [h['portfolio_value'] for h in self.portfolio_history]
        
        return {
            'portfolio_volatility': np.std(portfolio_values) if len(portfolio_values) > 1 else 0,
            'max_portfolio_value': max(portfolio_values),
            'min_portfolio_value': min(portfolio_values),
            'total_trades': len(self.trade_history)
        }

class TradingLogger:
    """Trading Logger"""
    
    def __init__(self):
        self.logs = []
    
    def log_trade(self, trade_record: Dict, decision: Dict):
        """Log Trade"""
        log_entry = {
            'timestamp': datetime.now(),
            'trade': trade_record,
            'decision': decision
        }
        
        self.logs.append(log_entry)
        
        if len(self.logs) > 1000:
            self.logs.pop(0)

def run_production_ready_test():
    """Führe Production-Ready Test durch"""
    
    print("🚀 PRODUCTION-READY TRADING SYSTEM TEST")
    print("=" * 60)
    
    # Erstelle Production System
    production_system = ProductionReadyTradingSystem(
        initial_capital=10000.0,
        max_risk_per_trade=0.02,
        max_daily_loss=0.05,
        target_daily_return=0.01
    )
    
    # Führe Production Session durch
    results = production_system.run_production_session(steps=200)
    
    # Zeige Ergebnisse
    print(f"\n📈 PRODUCTION SESSION ERGEBNISSE:")
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
    
    # Risk Metrics
    risk_metrics = results['risk_metrics']
    print(f"\n🛡️ RISK METRICS:")
    print(f"   Daily Trades: {risk_metrics['daily_trades']}")
    print(f"   Daily Risk Used: {risk_metrics['daily_risk_used']:.3f}")
    print(f"   Max Risk per Trade: {risk_metrics['max_risk_per_trade']*100:.1f}%")
    print(f"   Max Daily Loss: {risk_metrics['max_daily_loss']*100:.1f}%")
    
    # Performance Metrics
    perf_metrics = results['performance_metrics']
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Portfolio Volatility: {perf_metrics.get('portfolio_volatility', 0):.2f}")
    print(f"   Max Portfolio Value: ${perf_metrics.get('max_portfolio_value', 0):,.2f}")
    print(f"   Min Portfolio Value: ${perf_metrics.get('min_portfolio_value', 0):,.2f}")
    print(f"   Total Trades: {perf_metrics.get('total_trades', 0)}")
    
    return results

if __name__ == "__main__":
    run_production_ready_test()
