"""
Realistic Paper Trading Test mit synthetischen Marktdaten
Simuliert echte Marktbedingungen für AAPL, TSLA, BTC/USD, ETH/USD
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import unserer Standalone-Komponenten
from standalone_test import StandaloneTradingEmotionEngine, StandalonePaperTradingSystem, TradingEmotion

class RealisticMarketSimulator:
    """Simuliert realistische Marktdaten für verschiedene Assets"""
    
    def __init__(self):
        self.assets = {
            'AAPL': {'base_price': 150.0, 'volatility': 0.02, 'trend': 0.001},
            'TSLA': {'base_price': 200.0, 'volatility': 0.04, 'trend': 0.002},
            'BTC/USD': {'base_price': 45000.0, 'volatility': 0.05, 'trend': 0.003},
            'ETH/USD': {'base_price': 3000.0, 'volatility': 0.06, 'trend': 0.004}
        }
        
        self.current_prices = {symbol: data['base_price'] for symbol, data in self.assets.items()}
        self.price_history = {symbol: [data['base_price']] for symbol, data in self.assets.items()}
        
    def generate_market_data(self, steps: int = 100) -> Dict[str, List[Dict]]:
        """Generiere realistische Marktdaten"""
        
        market_data = {symbol: [] for symbol in self.assets.keys()}
        
        for step in range(steps):
            for symbol, data in self.assets.items():
                # Generiere Preis-Bewegung
                trend_component = data['trend'] * np.random.normal(0, 1)
                volatility_component = data['volatility'] * np.random.normal(0, 1)
                
                # Kombiniere Trend und Volatilität
                price_change = trend_component + volatility_component
                
                # Aktualisiere Preis
                self.current_prices[symbol] *= (1 + price_change)
                self.price_history[symbol].append(self.current_prices[symbol])
                
                # Berechne Metriken
                if len(self.price_history[symbol]) > 1:
                    returns = np.diff(self.price_history[symbol][-20:]) / self.price_history[symbol][-20:-1]
                    volatility = np.std(returns) if len(returns) > 1 else 0.02
                    volume = np.random.uniform(1000, 10000)
                else:
                    volatility = data['volatility']
                    volume = 5000
                
                # Speichere Marktdaten
                market_data[symbol].append({
                    'step': step,
                    'price': self.current_prices[symbol],
                    'price_change': price_change,
                    'volatility': volatility,
                    'volume': volume,
                    'timestamp': step
                })
        
        return market_data

class RealisticPaperTradingTest:
    """Realistischer Paper Trading Test"""
    
    def __init__(self):
        self.market_simulator = RealisticMarketSimulator()
        self.emotion_engine = StandaloneTradingEmotionEngine()
        self.paper_system = StandalonePaperTradingSystem(
            initial_capital=10000.0,
            commission=0.001,
            slippage=0.0005,
            min_trade_size=100.0
        )
        
        # Trading-Statistiken
        self.trading_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'emotion_changes': 0
        }
        
        self.initial_emotion = self.emotion_engine.current_emotion
        
    def run_realistic_test(self, steps: int = 100) -> Dict:
        """Führe realistischen Paper Trading Test durch"""
        
        print(f"🚀 Starte Realistic Paper Trading Test...")
        print(f"Steps: {steps}")
        print(f"Initial Capital: ${self.paper_system.initial_capital:,.2f}")
        print(f"Assets: {list(self.market_simulator.assets.keys())}")
        
        # Generiere Marktdaten
        print(f"\n📊 Generiere Marktdaten...")
        market_data = self.market_simulator.generate_market_data(steps)
        
        # Führe Trading-Simulation durch
        print(f"\n💹 Starte Trading-Simulation...")
        
        for step in range(steps):
            if step % 20 == 0:
                print(f"   Step {step}/{steps} - Portfolio: ${self.paper_system.get_portfolio_value(self.market_simulator.current_prices):,.2f}")
            
            # Wähle zufälliges Asset für diesen Step
            symbol = np.random.choice(list(self.market_simulator.assets.keys()))
            market_info = market_data[symbol][step]
            
            # Update Emotion Engine basierend auf Marktbedingungen
            self._update_emotion_engine(market_info, symbol)
            
            # Bestimme Trading-Entscheidung
            trading_decision = self._make_trading_decision(market_info, symbol)
            
            # Führe Trade aus
            if trading_decision['action'] != 'hold':
                self._execute_trade(trading_decision, market_info, symbol)
        
        # Berechne finale Ergebnisse
        final_results = self._calculate_final_results()
        
        print(f"\n✅ Realistic Paper Trading Test abgeschlossen!")
        return final_results
    
    def _update_emotion_engine(self, market_info: Dict, symbol: str):
        """Aktualisiere Emotion Engine basierend auf Marktbedingungen"""
        
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
            volume_change=market_info['volume'] / 5000 - 1,  # Normalisiere Volume
            volatility=market_info['volatility'],
            trend_strength=market_info['price_change'] * 2
        )
        
        # Update Performance
        self.emotion_engine.update_performance(
            portfolio_return=portfolio_return,
            trade_return=market_info['price_change'],
            drawdown=self.trading_stats['max_drawdown'],
            win_rate=win_rate
        )
        
        # Track Emotion Changes
        if self.emotion_engine.current_emotion != self.initial_emotion:
            self.trading_stats['emotion_changes'] += 1
            self.initial_emotion = self.emotion_engine.current_emotion
    
    def _make_trading_decision(self, market_info: Dict, symbol: str) -> Dict:
        """Treffe Trading-Entscheidung basierend auf Emotion und Marktbedingungen"""
        
        risk_tolerance = self.emotion_engine.get_risk_tolerance()
        position_modifier = self.emotion_engine.get_position_sizing_modifier()
        current_emotion = self.emotion_engine.current_emotion
        
        # Bestimme Trade-Wahrscheinlichkeit basierend auf Emotion
        trade_probability = {
            TradingEmotion.CONFIDENT: 0.8,
            TradingEmotion.CAUTIOUS: 0.3,
            TradingEmotion.FRUSTRATED: 0.1,
            TradingEmotion.GREEDY: 0.9,
            TradingEmotion.FEARFUL: 0.05,
            TradingEmotion.OPTIMISTIC: 0.6,
            TradingEmotion.PESSIMISTIC: 0.2,
            TradingEmotion.NEUTRAL: 0.4
        }.get(current_emotion, 0.4)
        
        # Entscheide ob Trade
        if np.random.random() > trade_probability:
            return {'action': 'hold', 'quantity': 0}
        
        # Bestimme Trade-Richtung basierend auf Marktbedingungen und Emotion
        price_change = market_info['price_change']
        volatility = market_info['volatility']
        
        # Emotion-basierte Trading-Logik
        if current_emotion in [TradingEmotion.CONFIDENT, TradingEmotion.GREEDY, TradingEmotion.OPTIMISTIC]:
            # Positive Emotionen = Tendenz zu Käufen
            if price_change > 0 or volatility < 0.03:
                action = 'buy'
            else:
                action = 'hold'
        elif current_emotion in [TradingEmotion.FEARFUL, TradingEmotion.FRUSTRATED, TradingEmotion.PESSIMISTIC]:
            # Negative Emotionen = Tendenz zu Verkäufen
            if price_change < 0 or volatility > 0.04:
                action = 'sell'
            else:
                action = 'hold'
        else:
            # Neutrale Emotionen = Balanced Trading
            if price_change > 0.01:
                action = 'buy'
            elif price_change < -0.01:
                action = 'sell'
            else:
                action = 'hold'
        
        # Bestimme Trade-Größe
        if action != 'hold':
            base_quantity = 5.0  # Basis-Trade-Größe
            quantity = base_quantity * position_modifier * risk_tolerance
            
            # Anpassung basierend auf Volatilität
            if volatility > 0.05:  # Hohe Volatilität = kleinere Trades
                quantity *= 0.5
            elif volatility < 0.02:  # Niedrige Volatilität = größere Trades
                quantity *= 1.5
            
            quantity = max(1.0, quantity)  # Mindest-Trade-Größe
        else:
            quantity = 0
        
        return {
            'action': action,
            'quantity': quantity,
            'emotion': current_emotion.value,
            'risk_tolerance': risk_tolerance,
            'position_modifier': position_modifier
        }
    
    def _execute_trade(self, decision: Dict, market_info: Dict, symbol: str):
        """Führe Trade aus"""
        
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
                
                # Update Max Drawdown
                current_portfolio = self.paper_system.get_portfolio_value(self.market_simulator.current_prices)
                if current_portfolio < self.paper_system.initial_capital:
                    drawdown = (self.paper_system.initial_capital - current_portfolio) / self.paper_system.initial_capital
                    self.trading_stats['max_drawdown'] = max(self.trading_stats['max_drawdown'], drawdown)
    
    def _calculate_final_results(self) -> Dict:
        """Berechne finale Ergebnisse"""
        
        final_portfolio_value = self.paper_system.get_portfolio_value(self.market_simulator.current_prices)
        total_return = (final_portfolio_value - self.paper_system.initial_capital) / self.paper_system.initial_capital
        
        win_rate = self.trading_stats['winning_trades'] / max(self.trading_stats['total_trades'], 1)
        
        # Berechne Sharpe Ratio (vereinfacht)
        if self.trading_stats['total_trades'] > 0:
            avg_trade_return = self.trading_stats['total_profit'] / self.trading_stats['total_trades']
            sharpe_ratio = avg_trade_return / max(self.trading_stats['max_drawdown'], 0.01)
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
            'max_drawdown': self.trading_stats['max_drawdown'],
            'max_drawdown_pct': self.trading_stats['max_drawdown'] * 100,
            'sharpe_ratio': sharpe_ratio,
            'emotion_changes': self.trading_stats['emotion_changes'],
            'final_emotion': self.emotion_engine.current_emotion.value,
            'commission_paid': self.paper_system.total_commission_paid,
            'slippage_cost': self.paper_system.total_slippage_cost
        }

def run_multi_asset_realistic_test():
    """Führe realistischen Test für mehrere Assets durch"""
    
    print("🚀 MULTI-ASSET REALISTIC PAPER TRADING TEST")
    print("=" * 60)
    
    assets = ['AAPL', 'TSLA', 'BTC/USD', 'ETH/USD']
    results = {}
    
    for asset in assets:
        print(f"\n📊 Teste {asset}...")
        
        # Erstelle Test für dieses Asset
        test = RealisticPaperTradingTest()
        
        # Führe Test durch
        result = test.run_realistic_test(steps=50)  # Reduziert für schnelleren Test
        
        results[asset] = result
        
        # Zeige Ergebnisse
        print(f"\n📈 {asset} Ergebnisse:")
        print(f"   Total Return: {result['total_return_pct']:.2f}%")
        print(f"   Win Rate: {result['win_rate_pct']:.1f}%")
        print(f"   Total Trades: {result['total_trades']}")
        print(f"   Max Drawdown: {result['max_drawdown_pct']:.2f}%")
        print(f"   Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"   Final Emotion: {result['final_emotion']}")
        print(f"   Emotion Changes: {result['emotion_changes']}")
    
    # Analysiere alle Ergebnisse
    print(f"\n📋 GESAMTANALYSE")
    print("=" * 30)
    
    # Erstelle Vergleichstabelle
    comparison_data = []
    for asset, result in results.items():
        comparison_data.append({
            'Asset': asset,
            'Return (%)': result['total_return_pct'],
            'Win Rate (%)': result['win_rate_pct'],
            'Total Trades': result['total_trades'],
            'Max DD (%)': result['max_drawdown_pct'],
            'Sharpe': result['sharpe_ratio'],
            'Final Emotion': result['final_emotion'],
            'Emotion Changes': result['emotion_changes']
        })
    
    df = pd.DataFrame(comparison_data)
    print("\n📊 PERFORMANCE VERGLEICH:")
    print(df.to_string(index=False, float_format='%.2f'))
    
    # Finde beste Performance
    best_return = df.loc[df['Return (%)'].idxmax()]
    best_winrate = df.loc[df['Win Rate (%)'].idxmax()]
    best_sharpe = df.loc[df['Sharpe'].idxmax()]
    
    print(f"\n🏆 BESTE PERFORMANCE:")
    print(f"   Höchster Return: {best_return['Asset']} ({best_return['Return (%)']:.2f}%)")
    print(f"   Beste Win Rate: {best_winrate['Asset']} ({best_winrate['Win Rate (%)']:.1f}%)")
    print(f"   Beste Sharpe Ratio: {best_sharpe['Asset']} ({best_sharpe['Sharpe']:.2f})")
    
    # Emotion-Analyse
    print(f"\n🧠 EMOTION-ANALYSE:")
    emotion_counts = df['Final Emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"   {emotion}: {count} Assets")
    
    # Durchschnittliche Performance
    avg_return = df['Return (%)'].mean()
    avg_winrate = df['Win Rate (%)'].mean()
    avg_sharpe = df['Sharpe'].mean()
    
    print(f"\n📈 DURCHSCHNITTLICHE PERFORMANCE:")
    print(f"   Return: {avg_return:.2f}%")
    print(f"   Win Rate: {avg_winrate:.1f}%")
    print(f"   Sharpe Ratio: {avg_sharpe:.2f}")
    
    return results

if __name__ == "__main__":
    run_multi_asset_realistic_test()
