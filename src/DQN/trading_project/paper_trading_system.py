"""
Paper Trading System für Emotion-Augmented Trading Agent
Simuliert reales Trading ohne echtes Geld zu riskieren
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
import json
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Import unserer Trading-Komponenten
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environments.trading_environment import TradingEnvironment, create_trading_environments
from environments.multi_timeframe_environment import MultiTimeframeEnvironment, create_multi_timeframe_environments
from agents.emotion_trading_agent import EmotionTradingAgent, train_emotion_trading_agent
from agents.multi_timeframe_agent import MultiTimeframeTradingAgent, train_multi_timeframe_agent

class PaperTradingSystem:
    """
    Paper Trading System für realistische Trading-Simulation
    """
    
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
        self.positions = {}  # {symbol: {'shares': float, 'avg_price': float}}
        self.portfolio_value = initial_capital
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        
        # Trading History
        self.trades = []
        self.daily_returns = []
        self.portfolio_history = []
        
        # Performance Metrics
        self.start_date = datetime.now()
        self.current_date = self.start_date
        
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Berechne aktuellen Portfolio-Wert"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total_value += position['shares'] * current_prices[symbol]
        
        return total_value
    
    def execute_trade(self, 
                     symbol: str, 
                     action: str, 
                     quantity: float, 
                     price: float,
                     timestamp: datetime = None) -> Dict:
        """Führe Trade aus"""
        
        if timestamp is None:
            timestamp = self.current_date
        
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
            # Kauf
            self.cash -= (trade_value + commission_cost)
            
            if symbol in self.positions:
                # Durchschnittspreis berechnen
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
            # Verkauf
            if symbol not in self.positions or self.positions[symbol]['shares'] < quantity:
                return {
                    'success': False,
                    'reason': f'Insufficient shares: {self.positions.get(symbol, {}).get("shares", 0):.2f} < {quantity:.2f}',
                    'available': self.positions.get(symbol, {}).get("shares", 0)
                }
            
            self.cash += (trade_value - commission_cost)
            self.positions[symbol]['shares'] -= quantity
            
            # Entferne Position wenn leer
            if self.positions[symbol]['shares'] <= 0.001:
                del self.positions[symbol]
        
        # Aktualisiere Statistiken
        self.total_commission_paid += commission_cost
        self.total_slippage_cost += abs(slippage_factor) * trade_value
        
        # Speichere Trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'execution_price': execution_price,
            'trade_value': trade_value,
            'commission': commission_cost,
            'slippage': abs(slippage_factor) * trade_value,
            'cash_after': self.cash,
            'portfolio_value': self.get_portfolio_value({symbol: execution_price})
        }
        
        self.trades.append(trade_record)
        
        return {
            'success': True,
            'trade_record': trade_record,
            'execution_price': execution_price,
            'commission': commission_cost,
            'slippage': abs(slippage_factor) * trade_value
        }
    
    def update_portfolio_history(self, current_prices: Dict[str, float]):
        """Aktualisiere Portfolio-Historie"""
        portfolio_value = self.get_portfolio_value(current_prices)
        self.portfolio_history.append({
            'date': self.current_date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions': self.positions.copy()
        })
        
        # Berechne tägliche Returns
        if len(self.portfolio_history) > 1:
            prev_value = self.portfolio_history[-2]['portfolio_value']
            daily_return = (portfolio_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
    
    def get_performance_metrics(self) -> Dict:
        """Berechne Performance-Metriken"""
        if not self.portfolio_history:
            return {}
        
        portfolio_values = [h['portfolio_value'] for h in self.portfolio_history]
        returns = np.array(self.daily_returns) if self.daily_returns else np.array([0])
        
        # Basis-Metriken
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (365 / len(portfolio_values)) - 1
        
        # Risiko-Metriken
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Trading-Metriken
        winning_trades = len([t for t in self.trades if t['action'] == 'sell' and 
                            t['execution_price'] > self.positions.get(t['symbol'], {}).get('avg_price', 0)])
        total_trades = len([t for t in self.trades if t['action'] == 'sell'])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_commission_paid': self.total_commission_paid,
            'total_slippage_cost': self.total_slippage_cost,
            'final_portfolio_value': portfolio_values[-1],
            'final_cash': self.cash,
            'final_positions': self.positions
        }
    
    def reset(self):
        """Reset Paper Trading System"""
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_value = self.initial_capital
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        self.trades = []
        self.daily_returns = []
        self.portfolio_history = []
        self.start_date = datetime.now()
        self.current_date = self.start_date

class PaperTradingBacktest:
    """
    Backtest System für Paper Trading
    """
    
    def __init__(self, 
                 paper_trading_system: PaperTradingSystem,
                 start_date: str = "2024-01-01",
                 end_date: str = "2024-12-31"):
        
        self.paper_system = paper_trading_system
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.current_date = self.start_date
        
        # Backtest Results
        self.backtest_results = {}
        self.comparison_results = {}
        
    def run_backtest(self, 
                    agent, 
                    environment, 
                    symbol: str,
                    episodes: int = 100) -> Dict:
        """Führe Backtest durch"""
        
        print(f"🚀 Starte Paper Trading Backtest für {symbol}...")
        print(f"Zeitraum: {self.start_date.date()} bis {self.end_date.date()}")
        print(f"Initial Capital: ${self.paper_system.initial_capital:,.2f}")
        
        # Reset Paper Trading System
        self.paper_system.reset()
        
        # Training des Agents
        print(f"📚 Trainiere Agent...")
        training_metrics = train_emotion_trading_agent(
            env=environment,
            episodes=episodes,
            save_interval=episodes // 4
        )
        
        # Backtest Simulation
        print(f"📊 Führe Backtest durch...")
        
        state, info = environment.reset()
        done = False
        step = 0
        
        while not done and step < 1000:  # Limit für Backtest
            # Agent Action
            action = agent.select_action(state, training=False)
            
            # Environment Step
            next_state, reward, done, truncated, info = environment.step(np.array([action]))
            
            # Paper Trading Execution
            current_price = info.get('current_price', 100.0)
            
            # Bestimme Trade basierend auf Action
            if abs(action) > 0.1:  # Trade-Schwelle
                if action > 0:
                    # Kauf
                    trade_value = abs(action) * self.paper_system.cash * 0.1  # 10% des Kapitals
                    quantity = trade_value / current_price
                    
                    result = self.paper_system.execute_trade(
                        symbol=symbol,
                        action='buy',
                        quantity=quantity,
                        price=current_price,
                        timestamp=self.current_date
                    )
                    
                else:
                    # Verkauf
                    if symbol in self.paper_system.positions:
                        quantity = abs(action) * self.paper_system.positions[symbol]['shares'] * 0.1
                        
                        result = self.paper_system.execute_trade(
                            symbol=symbol,
                            action='sell',
                            quantity=quantity,
                            price=current_price,
                            timestamp=self.current_date
                        )
            
            # Update Portfolio History
            self.paper_system.update_portfolio_history({symbol: current_price})
            
            # Nächster Schritt
            state = next_state
            step += 1
            self.current_date += timedelta(minutes=15)  # 15-Minuten Schritte
        
        # Berechne Backtest-Ergebnisse
        performance_metrics = self.paper_system.get_performance_metrics()
        
        backtest_result = {
            'symbol': symbol,
            'training_metrics': training_metrics,
            'paper_trading_metrics': performance_metrics,
            'num_trades': len(self.paper_system.trades),
            'backtest_duration': step,
            'start_date': self.start_date,
            'end_date': self.current_date
        }
        
        self.backtest_results[symbol] = backtest_result
        
        print(f"✅ Backtest für {symbol} abgeschlossen")
        print(f"   Total Return: {performance_metrics['total_return']*100:.2f}%")
        print(f"   Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {performance_metrics['max_drawdown']*100:.2f}%")
        print(f"   Win Rate: {performance_metrics['win_rate']*100:.1f}%")
        print(f"   Total Trades: {performance_metrics['total_trades']}")
        
        return backtest_result
    
    def run_multi_symbol_backtest(self, 
                                 symbols: List[str],
                                 environments: Dict,
                                 episodes_per_symbol: int = 100) -> Dict:
        """Führe Backtest für mehrere Symbole durch"""
        
        print(f"🚀 Starte Multi-Symbol Paper Trading Backtest...")
        print(f"Symbole: {symbols}")
        print(f"Episodes pro Symbol: {episodes_per_symbol}")
        
        all_results = {}
        
        for symbol in symbols:
            if symbol in environments:
                try:
                    # Erstelle Agent für dieses Symbol
                    env = environments[symbol]
                    state_size = env.observation_space.shape[0]
                    
                    agent = EmotionTradingAgent(
                        state_size=state_size,
                        action_size=1,
                        learning_rate=1e-4,
                        epsilon_decay=0.995
                    )
                    
                    # Führe Backtest durch
                    result = self.run_backtest(agent, env, symbol, episodes_per_symbol)
                    all_results[symbol] = result
                    
                except Exception as e:
                    print(f"❌ Fehler bei {symbol}: {e}")
                    continue
        
        # Analysiere alle Ergebnisse
        self.analyze_multi_symbol_results(all_results)
        
        return all_results
    
    def analyze_multi_symbol_results(self, results: Dict):
        """Analysiere Multi-Symbol Backtest-Ergebnisse"""
        
        print(f"\n📈 MULTI-SYMBOL BACKTEST ANALYSE")
        print("=" * 50)
        
        if not results:
            print("❌ Keine Ergebnisse zu analysieren!")
            return
        
        # Erstelle Vergleichstabelle
        comparison_data = []
        
        for symbol, result in results.items():
            paper_metrics = result['paper_trading_metrics']
            training_metrics = result['training_metrics']
            
            comparison_data.append({
                'Symbol': symbol,
                'Paper Return (%)': paper_metrics['total_return'] * 100,
                'Training Return (%)': training_metrics['total_return'] * 100,
                'Paper Sharpe': paper_metrics['sharpe_ratio'],
                'Paper Max DD (%)': paper_metrics['max_drawdown'] * 100,
                'Paper Win Rate (%)': paper_metrics['win_rate'] * 100,
                'Paper Trades': paper_metrics['total_trades'],
                'Commission Paid': paper_metrics['total_commission_paid'],
                'Slippage Cost': paper_metrics['total_slippage_cost']
            })
        
        # Erstelle DataFrame
        df = pd.DataFrame(comparison_data)
        
        print("\n📊 PAPER TRADING PERFORMANCE VERGLEICH:")
        print(df.to_string(index=False, float_format='%.2f'))
        
        # Finde beste Performance
        best_return = df.loc[df['Paper Return (%)'].idxmax()]
        best_sharpe = df.loc[df['Paper Sharpe'].idxmax()]
        lowest_drawdown = df.loc[df['Paper Max DD (%)'].idxmin()]
        
        print(f"\n🏆 BESTE PAPER TRADING PERFORMANCE:")
        print(f"   Höchster Return: {best_return['Symbol']} ({best_return['Paper Return (%)']:.2f}%)")
        print(f"   Beste Sharpe Ratio: {best_sharpe['Symbol']} ({best_sharpe['Paper Sharpe']:.2f})")
        print(f"   Niedrigster Drawdown: {lowest_drawdown['Symbol']} ({lowest_drawdown['Paper Max DD (%)']:.2f}%)")
        
        # Speichere Ergebnisse
        self.save_backtest_results(results, df)
        
        # Erstelle Visualisierungen
        self.create_backtest_visualizations(results, df)
    
    def save_backtest_results(self, results: Dict, df: pd.DataFrame):
        """Speichere Backtest-Ergebnisse"""
        
        # Erstelle results Ordner
        os.makedirs("trading_project/results", exist_ok=True)
        
        # Speichere Vergleichstabelle
        df.to_csv("trading_project/results/paper_trading_backtest_comparison.csv", index=False)
        
        # Speichere detaillierte Ergebnisse
        with open("trading_project/results/paper_trading_detailed_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Paper Trading Backtest Ergebnisse gespeichert in trading_project/results/")
    
    def create_backtest_visualizations(self, results: Dict, df: pd.DataFrame):
        """Erstelle Backtest-Visualisierungen"""
        
        try:
            # Erstelle Figure mit Subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Paper Trading Backtest - Performance Analysis', fontsize=16)
            
            # 1. Paper vs Training Returns
            x = np.arange(len(df))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, df['Paper Return (%)'], width, label='Paper Trading', alpha=0.8)
            axes[0, 0].bar(x + width/2, df['Training Return (%)'], width, label='Training', alpha=0.8)
            axes[0, 0].set_xlabel('Symbols')
            axes[0, 0].set_ylabel('Return (%)')
            axes[0, 0].set_title('Paper Trading vs Training Returns')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(df['Symbol'])
            axes[0, 0].legend()
            
            # 2. Sharpe Ratio
            axes[0, 1].bar(df['Symbol'], df['Paper Sharpe'])
            axes[0, 1].set_title('Paper Trading Sharpe Ratio')
            axes[0, 1].set_ylabel('Sharpe Ratio')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. Max Drawdown
            axes[1, 0].bar(df['Symbol'], df['Paper Max DD (%)'])
            axes[1, 0].set_title('Paper Trading Max Drawdown')
            axes[1, 0].set_ylabel('Max Drawdown (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. Win Rate
            axes[1, 1].bar(df['Symbol'], df['Paper Win Rate (%)'])
            axes[1, 1].set_title('Paper Trading Win Rate')
            axes[1, 1].set_ylabel('Win Rate (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('trading_project/results/paper_trading_backtest_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"📊 Paper Trading Visualisierungen gespeichert in trading_project/results/paper_trading_backtest_analysis.png")
            
        except Exception as e:
            print(f"⚠️ Fehler beim Erstellen der Paper Trading Visualisierungen: {e}")


def run_paper_trading_tests():
    """
    Führe Paper Trading Tests für alle Symbole durch
    """
    
    print("🚀 Starte Paper Trading Tests...")
    
    # Erstelle Paper Trading System
    paper_system = PaperTradingSystem(
        initial_capital=10000.0,
        commission=0.001,  # 0.1% Kommission
        slippage=0.0005,   # 0.05% Slippage
        min_trade_size=100.0
    )
    
    # Erstelle Backtest System
    backtest = PaperTradingBacktest(
        paper_system=paper_system,
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    # Erstelle Environments
    print("\n📊 Erstelle Trading Environments...")
    envs = create_trading_environments()
    
    if not envs:
        print("❌ Keine Environments erstellt!")
        return
    
    # Wähle Test-Symbole
    test_symbols = ['AAPL', 'TSLA', 'BTC/USD', 'ETH/USD']
    available_envs = {symbol: env for symbol, env in envs.items() 
                     if any(test_symbol in symbol for test_symbol in test_symbols)}
    
    print(f"✅ {len(available_envs)} Environments für Paper Trading Tests verfügbar")
    
    # Führe Multi-Symbol Backtest durch
    results = backtest.run_multi_symbol_backtest(
        symbols=list(available_envs.keys()),
        environments=available_envs,
        episodes_per_symbol=100
    )
    
    return results

def run_multi_timeframe_paper_trading():
    """
    Führe Paper Trading Tests mit Multi-Timeframe System durch
    """
    
    print("🚀 Starte Multi-Timeframe Paper Trading Tests...")
    
    # Erstelle Paper Trading System
    paper_system = PaperTradingSystem(
        initial_capital=10000.0,
        commission=0.001,
        slippage=0.0005,
        min_trade_size=100.0
    )
    
    # Erstelle Backtest System
    backtest = PaperTradingBacktest(
        paper_system=paper_system,
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    # Erstelle Multi-Timeframe Environments
    print("\n📊 Erstelle Multi-Timeframe Trading Environments...")
    envs = create_multi_timeframe_environments()
    
    if not envs:
        print("❌ Keine Multi-Timeframe Environments erstellt!")
        return
    
    # Wähle Test-Symbole
    test_symbols = ['AAPL', 'TSLA', 'BTC/USD', 'ETH/USD']
    available_envs = {symbol: env for symbol, env in envs.items() 
                     if any(test_symbol in symbol for test_symbol in test_symbols)}
    
    print(f"✅ {len(available_envs)} Multi-Timeframe Environments für Paper Trading Tests verfügbar")
    
    # Führe Multi-Symbol Backtest durch
    results = backtest.run_multi_symbol_backtest(
        symbols=list(available_envs.keys()),
        environments=available_envs,
        episodes_per_symbol=100
    )
    
    return results

if __name__ == "__main__":
    print("🎯 Paper Trading System für Emotion-Augmented Trading Agent")
    print("=" * 60)
    
    # Wähle Test-Modus
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "multi":
        # Multi-Timeframe Paper Trading
        run_multi_timeframe_paper_trading()
    else:
        # Standard Paper Trading
        run_paper_trading_tests()
    
    print("\n✅ Paper Trading Tests abgeschlossen!")
