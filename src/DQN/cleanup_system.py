"""
System Cleanup Script
Bereinigt temporäre Dateien und optimiert das System
"""

import os
import shutil
import glob
import sys
from pathlib import Path

def cleanup_temporary_files():
    """Bereinige temporäre Dateien"""
    
    print("🧹 SYSTEM CLEANUP GESTARTET")
    print("=" * 50)
    
    # Definiere Datei-Muster für Cleanup
    cleanup_patterns = [
        "**/*.pyc",
        "**/__pycache__",
        "**/*.tmp",
        "**/*.log",
        "**/*.cache",
        "**/*.pth",
        "**/*.pt",
        "**/*.checkpoint",
        "**/temp_*",
        "**/tmp_*"
    ]
    
    total_cleaned = 0
    total_size_freed = 0
    
    for pattern in cleanup_patterns:
        print(f"\n🔍 Suche nach: {pattern}")
        
        # Finde Dateien/Ordner
        matches = glob.glob(pattern, recursive=True)
        
        if matches:
            print(f"   Gefunden: {len(matches)} Dateien/Ordner")
            
            for match in matches:
                try:
                    if os.path.isfile(match):
                        # Datei löschen
                        size = os.path.getsize(match)
                        os.remove(match)
                        total_cleaned += 1
                        total_size_freed += size
                        print(f"   ✅ Gelöscht: {match} ({size} bytes)")
                        
                    elif os.path.isdir(match):
                        # Ordner löschen
                        size = get_folder_size(match)
                        shutil.rmtree(match)
                        total_cleaned += 1
                        total_size_freed += size
                        print(f"   ✅ Gelöscht: {match}/ ({size} bytes)")
                        
                except Exception as e:
                    print(f"   ❌ Fehler bei {match}: {e}")
        else:
            print(f"   ℹ️ Keine Dateien gefunden")
    
    print(f"\n📊 CLEANUP ERGEBNISSE:")
    print(f"   Gelöschte Dateien/Ordner: {total_cleaned}")
    print(f"   Freigegebener Speicher: {total_size_freed / 1024 / 1024:.2f} MB")
    
    return total_cleaned, total_size_freed

def get_folder_size(folder_path):
    """Berechne Ordner-Größe"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception:
        pass
    return total_size

def cleanup_trading_project():
    """Bereinige Trading Project spezifisch"""
    
    print(f"\n💹 TRADING PROJECT CLEANUP:")
    
    trading_path = Path("trading_project")
    if not trading_path.exists():
        print("   ℹ️ Trading Project Ordner nicht gefunden")
        return
    
    # Bereinige Trading Project
    cleanup_items = [
        "*.tmp",
        "*.log", 
        "*.cache",
        "temp_*",
        "tmp_*",
        "__pycache__",
        "*.pyc"
    ]
    
    cleaned = 0
    for item in cleanup_items:
        matches = list(trading_path.rglob(item))
        for match in matches:
            try:
                if match.is_file():
                    match.unlink()
                    cleaned += 1
                elif match.is_dir():
                    shutil.rmtree(match)
                    cleaned += 1
            except Exception as e:
                print(f"   ❌ Fehler bei {match}: {e}")
    
    print(f"   ✅ Trading Project bereinigt: {cleaned} Dateien/Ordner")

def create_gitignore():
    """Erstelle/aktualisiere .gitignore"""
    
    print(f"\n📁 GITIGNORE AKTUALISIERUNG:")
    
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pth
*.pt
*.checkpoint

# Trading Project
trading_project/temp_*
trading_project/tmp_*
trading_project/*.log
trading_project/*.cache
trading_project/results/temp_*

# Temporary files
*.tmp
*.temp
*.log
*.cache

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Results (optional - uncomment if you don't want to track results)
# results/
# trading_project/results/
"""
    
    try:
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        print("   ✅ .gitignore aktualisiert")
    except Exception as e:
        print(f"   ❌ Fehler beim Erstellen von .gitignore: {e}")

def analyze_memory_usage():
    """Analysiere Memory-Verbrauch"""
    
    print(f"\n🧠 MEMORY ANALYSE:")
    
    # Python-Prozesse
    try:
        import psutil
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            if 'python' in proc.info['name'].lower():
                python_processes.append(proc)
        
        if python_processes:
            print(f"   Python-Prozesse gefunden: {len(python_processes)}")
            total_memory = sum(proc.info['memory_info'].rss for proc in python_processes)
            print(f"   Gesamter Python Memory: {total_memory / 1024 / 1024:.2f} MB")
        else:
            print("   ℹ️ Keine Python-Prozesse gefunden")
            
    except ImportError:
        print("   ℹ️ psutil nicht verfügbar - installiere mit: pip install psutil")
    except Exception as e:
        print(f"   ❌ Fehler bei Memory-Analyse: {e}")

def main():
    """Hauptfunktion"""
    
    print("🚀 SYSTEM CLEANUP & OPTIMIERUNG")
    print("=" * 60)
    
    # 1. Temporäre Dateien bereinigen
    cleaned_files, freed_size = cleanup_temporary_files()
    
    # 2. Trading Project bereinigen
    cleanup_trading_project()
    
    # 3. .gitignore aktualisieren
    create_gitignore()
    
    # 4. Memory-Analyse
    analyze_memory_usage()
    
    print(f"\n✅ CLEANUP ABGESCHLOSSEN!")
    print(f"   Gelöschte Dateien: {cleaned_files}")
    print(f"   Freigegebener Speicher: {freed_size / 1024 / 1024:.2f} MB")
    
    print(f"\n💡 EMPFEHLUNGEN:")
    print(f"   1. System neu starten für vollständige Memory-Freigabe")
    print(f"   2. .gitignore verwenden um temporäre Dateien zu ignorieren")
    print(f"   3. Regelmäßige Cleanups durchführen")
    print(f"   4. Trading-Systeme mit weniger Memory verwenden")

if __name__ == "__main__":
    main()
