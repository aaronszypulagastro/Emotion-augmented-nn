"""
Live Trading Preparation System
Vorbereitung für echte Trading-Anwendungen mit kleinen Beträgen
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
import json
from datetime import datetime, timedelta
import time
warnings.filterwarnings('ignore')

# Import unserer robusten Komponenten
from robust_trading_system import RobustTradingSystem

class LiveTradingPreparation:
    """Live Trading Preparation System"""
    
    def __init__(self, test_capital: float = 1000.0):
        self.test_capital = test_capital
        
        # Erstelle robustes System für Live Trading
        self.robust_system = RobustTradingSystem(initial_capital=test_capital)
        
        # Live Trading Features
        self.live_trading_enabled = False
        self.test_mode = True
        self.safety_checks = True
        
        # Performance Tracking
        self.daily_results = []
        self.weekly_results = []
        self.monthly_results = []
        
        # Risk Limits für Live Trading
        self.max_daily_loss_live = 0.05  # 5% für Live Trading
        self.max_weekly_loss_live = 0.15  # 15% für Live Trading
        self.max_monthly_loss_live = 0.30  # 30% für Live Trading
        
        # Trading Limits
        self.max_trades_per_day = 10
        self.max_trades_per_week = 50
        self.daily_trade_count = 0
        self.weekly_trade_count = 0
        
        # Monitoring
        self.start_time = datetime.now()
        self.last_reset_time = datetime.now()
        
    def run_live_trading_preparation(self, days: int = 7) -> Dict:
        """Führe Live Trading Vorbereitung durch"""
        
        print(f"🚀 LIVE TRADING PREPARATION")
        print(f"Startzeit: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test-Kapital: ${self.test_capital:,.2f}")
        print(f"Test-Dauer: {days} Tage")
        print(f"Max Daily Loss: {self.max_daily_loss_live*100:.1f}%")
        print(f"Max Weekly Loss: {self.max_weekly_loss_live*100:.1f}%")
        print(f"Max Trades per Day: {self.max_trades_per_day}")
        
        # Simuliere Live Trading über mehrere Tage
        print(f"\n📅 Starte Live Trading Simulation...")
        
        for day in range(days):
            print(f"\n--- TAG {day+1}/{days} ---")
            day_result = self._simulate_trading_day(day)
            self.daily_results.append(day_result)
            
            # Prüfe Weekly Limits
            if (day + 1) % 7 == 0:
                weekly_result = self._calculate_weekly_result()
                self.weekly_results.append(weekly_result)
                print(f"📊 WOCHE {(day+1)//7} ABGESCHLOSSEN:")
                print(f"   Weekly Return: {weekly_result['weekly_return_pct']:.2f}%")
                print(f"   Weekly Trades: {weekly_result['weekly_trades']}")
                print(f"   Weekly Win Rate: {weekly_result['weekly_win_rate_pct']:.1f}%")
                
                # Reset Weekly Counters
                self.weekly_trade_count = 0
            
            # Prüfe Safety Limits
            if not self._check_safety_limits():
                print(f"🚨 SAFETY LIMITS ERREICHT - Trading gestoppt!")
                break
        
        # Finale Ergebnisse
        final_results = self._calculate_final_results()
        
        print(f"\n✅ Live Trading Preparation abgeschlossen!")
        return final_results
    
    def _simulate_trading_day(self, day: int) -> Dict:
        """Simuliere einen Trading-Tag"""
        
        print(f"🌅 Tag {day+1} - Trading Session startet...")
        
        # Reset Daily Counters
        self.daily_trade_count = 0
        daily_start_capital = self.robust_system.paper_system.get_portfolio_value(self.robust_system._get_current_prices())
        
        # Simuliere Trading-Session (100 Steps = 1 Tag)
        session_result = self.robust_system.run_robust_session(steps=100)
        
        # Berechne Daily Results
        daily_end_capital = session_result['final_portfolio_value']
        daily_return = (daily_end_capital - daily_start_capital) / daily_start_capital
        
        daily_result = {
            'day': day + 1,
            'start_capital': daily_start_capital,
            'end_capital': daily_end_capital,
            'daily_return': daily_return,
            'daily_return_pct': daily_return * 100,
            'daily_trades': session_result['total_trades'],
            'daily_win_rate_pct': session_result['win_rate_pct'],
            'max_drawdown_pct': session_result['max_drawdown_pct'],
            'final_emotion': session_result['final_emotion'],
            'trading_enabled': session_result['trading_enabled'],
            'recovery_mode': session_result['recovery_mode']
        }
        
        print(f"📊 TAG {day+1} ERGEBNISSE:")
        print(f"   Start Capital: ${daily_start_capital:,.2f}")
        print(f"   End Capital: ${daily_end_capital:,.2f}")
        print(f"   Daily Return: {daily_return*100:.2f}%")
        print(f"   Daily Trades: {session_result['total_trades']}")
        print(f"   Win Rate: {session_result['win_rate_pct']:.1f}%")
        print(f"   Max Drawdown: {session_result['max_drawdown_pct']:.2f}%")
        print(f"   Final Emotion: {session_result['final_emotion']}")
        print(f"   Trading Enabled: {session_result['trading_enabled']}")
        
        return daily_result
    
    def _calculate_weekly_result(self) -> Dict:
        """Berechne Wochen-Ergebnis"""
        
        # Hole letzte 7 Tage
        recent_days = self.daily_results[-7:]
        
        weekly_start_capital = recent_days[0]['start_capital']
        weekly_end_capital = recent_days[-1]['end_capital']
        weekly_return = (weekly_end_capital - weekly_start_capital) / weekly_start_capital
        
        weekly_trades = sum(day['daily_trades'] for day in recent_days)
        weekly_win_rate = np.mean([day['daily_win_rate_pct'] for day in recent_days])
        weekly_max_drawdown = max(day['max_drawdown_pct'] for day in recent_days)
        
        return {
            'week': len(self.weekly_results) + 1,
            'weekly_start_capital': weekly_start_capital,
            'weekly_end_capital': weekly_end_capital,
            'weekly_return': weekly_return,
            'weekly_return_pct': weekly_return * 100,
            'weekly_trades': weekly_trades,
            'weekly_win_rate_pct': weekly_win_rate,
            'weekly_max_drawdown_pct': weekly_max_drawdown
        }
    
    def _check_safety_limits(self) -> bool:
        """Prüfe Safety Limits"""
        
        # Prüfe Daily Loss
        if self.daily_results:
            last_day = self.daily_results[-1]
            if last_day['daily_return'] < -self.max_daily_loss_live:
                print(f"🚨 Daily Loss Limit erreicht: {last_day['daily_return']*100:.2f}%")
                return False
        
        # Prüfe Weekly Loss
        if self.weekly_results:
            last_week = self.weekly_results[-1]
            if last_week['weekly_return'] < -self.max_weekly_loss_live:
                print(f"🚨 Weekly Loss Limit erreicht: {last_week['weekly_return']*100:.2f}%")
                return False
        
        # Prüfe Monthly Loss
        if len(self.daily_results) >= 30:
            monthly_start = self.daily_results[-30]['start_capital']
            monthly_end = self.daily_results[-1]['end_capital']
            monthly_return = (monthly_end - monthly_start) / monthly_start
            
            if monthly_return < -self.max_monthly_loss_live:
                print(f"🚨 Monthly Loss Limit erreicht: {monthly_return*100:.2f}%")
                return False
        
        return True
    
    def _calculate_final_results(self) -> Dict:
        """Berechne finale Ergebnisse"""
        
        if not self.daily_results:
            return {}
        
        # Gesamt-Performance
        total_start_capital = self.daily_results[0]['start_capital']
        total_end_capital = self.daily_results[-1]['end_capital']
        total_return = (total_end_capital - total_start_capital) / total_start_capital
        
        # Statistiken
        total_trades = sum(day['daily_trades'] for day in self.daily_results)
        avg_daily_return = np.mean([day['daily_return'] for day in self.daily_results])
        avg_win_rate = np.mean([day['daily_win_rate_pct'] for day in self.daily_results])
        max_drawdown = max(day['max_drawdown_pct'] for day in self.daily_results)
        
        # Trading Days
        trading_days = len(self.daily_results)
        profitable_days = len([day for day in self.daily_results if day['daily_return'] > 0])
        profitable_day_rate = profitable_days / trading_days if trading_days > 0 else 0
        
        # Risk Metrics
        daily_returns = [day['daily_return'] for day in self.daily_results]
        volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0
        sharpe_ratio = avg_daily_return / volatility if volatility > 0 else 0
        
        # Live Trading Readiness
        live_trading_ready = self._assess_live_trading_readiness()
        
        return {
            'total_start_capital': total_start_capital,
            'total_end_capital': total_end_capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_trades': total_trades,
            'trading_days': trading_days,
            'profitable_days': profitable_days,
            'profitable_day_rate_pct': profitable_day_rate * 100,
            'avg_daily_return_pct': avg_daily_return * 100,
            'avg_win_rate_pct': avg_win_rate,
            'max_drawdown_pct': max_drawdown,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'live_trading_ready': live_trading_ready,
            'safety_checks_passed': self._check_safety_limits(),
            'daily_results': self.daily_results,
            'weekly_results': self.weekly_results
        }
    
    def _assess_live_trading_readiness(self) -> Dict:
        """Bewerte Live Trading Bereitschaft"""
        
        if not self.daily_results:
            return {'ready': False, 'reasons': ['Keine Daten verfügbar']}
        
        # Kriterien für Live Trading Bereitschaft
        criteria = {
            'positive_return': self.daily_results[-1]['end_capital'] > self.daily_results[0]['start_capital'],
            'low_volatility': np.std([day['daily_return'] for day in self.daily_results]) < 0.05,
            'consistent_trading': len([day for day in self.daily_results if day['daily_trades'] > 0]) > len(self.daily_results) * 0.5,
            'risk_management': max(day['max_drawdown_pct'] for day in self.daily_results) < 10.0,
            'emotion_stability': len(set(day['final_emotion'] for day in self.daily_results)) <= 4
        }
        
        passed_criteria = sum(criteria.values())
        total_criteria = len(criteria)
        
        readiness_score = passed_criteria / total_criteria
        
        reasons = []
        if not criteria['positive_return']:
            reasons.append('Negative Gesamt-Performance')
        if not criteria['low_volatility']:
            reasons.append('Hohe Volatilität')
        if not criteria['consistent_trading']:
            reasons.append('Inkonsistentes Trading')
        if not criteria['risk_management']:
            reasons.append('Schwaches Risk Management')
        if not criteria['emotion_stability']:
            reasons.append('Instabile Emotionen')
        
        return {
            'ready': readiness_score >= 0.8,
            'readiness_score': readiness_score,
            'passed_criteria': passed_criteria,
            'total_criteria': total_criteria,
            'reasons': reasons,
            'criteria_details': criteria
        }
    
    def generate_live_trading_report(self, results: Dict) -> str:
        """Generiere Live Trading Report"""
        
        report = f"""
🚀 LIVE TRADING PREPARATION REPORT
=====================================

📊 GESAMTPERFORMANCE:
   Start Kapital: ${results['total_start_capital']:,.2f}
   End Kapital: ${results['total_end_capital']:,.2f}
   Gesamt Return: {results['total_return_pct']:.2f}%
   Trading Tage: {results['trading_days']}
   Profitable Tage: {results['profitable_days']} ({results['profitable_day_rate_pct']:.1f}%)
   Gesamt Trades: {results['total_trades']}

📈 PERFORMANCE METRIKEN:
   Durchschnittlicher Tages-Return: {results['avg_daily_return_pct']:.2f}%
   Durchschnittliche Win Rate: {results['avg_win_rate_pct']:.1f}%
   Max Drawdown: {results['max_drawdown_pct']:.2f}%
   Volatilität: {results['volatility_pct']:.2f}%
   Sharpe Ratio: {results['sharpe_ratio']:.2f}

🛡️ RISK MANAGEMENT:
   Safety Checks: {'✅ PASSED' if results['safety_checks_passed'] else '❌ FAILED'}
   Max Daily Loss: {self.max_daily_loss_live*100:.1f}%
   Max Weekly Loss: {self.max_weekly_loss_live*100:.1f}%
   Max Monthly Loss: {self.max_monthly_loss_live*100:.1f}%

🎯 LIVE TRADING BEREITSCHAFT:
   Bereitschaft: {'✅ READY' if results['live_trading_ready']['ready'] else '❌ NOT READY'}
   Score: {results['live_trading_ready']['readiness_score']*100:.1f}%
   Erfüllte Kriterien: {results['live_trading_ready']['passed_criteria']}/{results['live_trading_ready']['total_criteria']}
   
   Kriterien Details:
   - Positive Performance: {'✅' if results['live_trading_ready']['criteria_details']['positive_return'] else '❌'}
   - Niedrige Volatilität: {'✅' if results['live_trading_ready']['criteria_details']['low_volatility'] else '❌'}
   - Konsistentes Trading: {'✅' if results['live_trading_ready']['criteria_details']['consistent_trading'] else '❌'}
   - Risk Management: {'✅' if results['live_trading_ready']['criteria_details']['risk_management'] else '❌'}
   - Emotionale Stabilität: {'✅' if results['live_trading_ready']['criteria_details']['emotion_stability'] else '❌'}

⚠️ VERBESSERUNGSBEREICHE:
"""
        
        for reason in results['live_trading_ready']['reasons']:
            report += f"   - {reason}\n"
        
        report += f"""
💡 EMPFEHLUNGEN:
   {'✅ System ist bereit für Live Trading mit kleinen Beträgen' if results['live_trading_ready']['ready'] else '❌ System benötigt weitere Optimierung vor Live Trading'}
   
   Nächste Schritte:
   {'1. Starte mit $100-500 Test-Beträgen' if results['live_trading_ready']['ready'] else '1. Optimiere identifizierte Schwachstellen'}
   {'2. Überwache Performance kontinuierlich' if results['live_trading_ready']['ready'] else '2. Führe weitere Tests durch'}
   {'3. Erhöhe Beträge schrittweise' if results['live_trading_ready']['ready'] else '3. Verbessere Risk Management'}
   
=====================================
Report generiert: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report

def run_live_trading_preparation():
    """Führe Live Trading Preparation durch"""
    
    print("🚀 LIVE TRADING PREPARATION SYSTEM")
    print("=" * 60)
    
    # Erstelle Live Trading Preparation System
    live_prep = LiveTradingPreparation(test_capital=1000.0)
    
    # Führe Live Trading Preparation durch
    results = live_prep.run_live_trading_preparation(days=7)
    
    # Generiere Report
    report = live_prep.generate_live_trading_report(results)
    print(report)
    
    # Speichere Report
    with open('trading_project/live_trading_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📁 Live Trading Report gespeichert: trading_project/live_trading_report.txt")
    
    return results

if __name__ == "__main__":
    run_live_trading_preparation()
