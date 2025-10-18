# Projekt Cleanup Plan

## 🗂️ DATEIEN ZU ARCHIVIEREN/LÖSCHEN

### **CORE (Veraltet):**
```
❌ emotion_engine.py                     → haben emotion_engine_fixed.py
❌ adaptive_zone_predictor.py            → haben adaptive_zone_predictor_v2.py
❌ meta_optimizer.py                     → haben meta_optimizer_v2.py
❌ winner_mindset_regulator.py           → FAILED approach
❌ phase7_integration_manager.py         → Phase 7 veraltet
❌ adaptive_configuration_manager.py     → nicht mehr verwendet
❌ auto_tuner.py                         → nicht mehr verwendet
❌ bayesian_hyperparameter_optimizer.py  → nicht mehr verwendet
❌ emotion_curriculum_learning.py        → nicht mehr verwendet
❌ emotion_predictive_regulation_unit.py → nicht mehr verwendet
❌ multi_objective_optimizer.py          → nicht mehr verwendet
❌ performance_stability_analyzer.py     → nicht mehr verwendet
❌ reward_zone_analyzer.py               → nicht mehr verwendet
❌ self_regulation_controller.py         → nicht mehr verwendet
❌ zone_transition_engine.py             → nicht mehr verwendet
```

### **TRAINING (Veraltet):**
```
❌ train_acrobot_winner_mindset.py       → FAILED approach
❌ train_lunarlander_winner_mindset.py   → FAILED approach
❌ train_test1_vanilla_plus_emotion.py   → alte Tests
❌ train_test2_vanilla_plus_emotion_plus_bdh.py → alte Tests
❌ train_test3_vanilla_plus_fixed_emotion.py → alte Tests
❌ train_test4_emotion_for_exploration.py → alte Tests
❌ train_finetuning.py                   → Phase 7
❌ train.py                              → veraltet
```

### **ANALYSIS (Veraltet):**
```
❌ adaptive_feedback_analyzer.py         → Phase 7
❌ analyze_results.py                    → generisch, nicht mehr verwendet
❌ compare_results.py                    → ersetzt durch compare_multi_environment.py
❌ plot_winner_mindset.py                → FAILED approach
❌ policy_surface_analyzer.py            → Phase 7
❌ zone_response_map.py                  → Phase 7
❌ emotion_td_eta_trends.py              → spezifisch, nicht mehr verwendet
❌ summary_dashboard.py                  → ersetzt
```

### **DOCS (Veraltet):**
```
❌ PHASE_8_MASTER_STATUS.md              → veraltet
❌ PHASE_8_FINAL_SUMMARY.md              → veraltet  
❌ PHASE_8_1_COMPETITIVE_LEARNING.md     → veraltet
❌ CURRENT_STATUS.md                     → veraltet
❌ AGENT_IMPROVEMENT_ROADMAP.md          → veraltet
❌ NEXT_LEVEL_STRATEGY.md                → veraltet
❌ COMPLETE_SYSTEM_ARCHITECTURE.md       → veraltet
```

---

## ✅ DATEIEN ZU BEHALTEN (AKTIV)

### **CORE (Aktiv):**
```
✅ competitive_emotion_engine.py         → CURRENT approach
✅ infrastructure_profile.py             → CURRENT approach
✅ rainbow_dqn_agent.py                  → NEW agent!
✅ prioritized_replay_buffer.py          → Rainbow component
✅ dueling_network.py                    → Rainbow component
✅ live_infrastructure_adapter.py        → Future feature
✅ meta_performance_predictor.py         → Useful utility
✅ emotion_engine_fixed.py               → Baseline emotion
✅ adaptive_zone_predictor_v2.py         → Useful utility
✅ meta_optimizer_v2.py                  → Useful utility
```

### **TRAINING (Aktiv):**
```
✅ agent.py                              → Base agent
✅ train_rainbow_universal.py            → NEW! Universal script
✅ train_competitive_selfplay.py         → Current approach
✅ train_competitive_optimized.py        → Optimized version
✅ train_regional_infrastructure.py      → Regional testing
✅ train_acrobot_regional.py             → Environment-specific
✅ train_lunarlander_regional.py         → Environment-specific
✅ train_vanilla_dqn.py                  → Baseline
✅ train_vanilla_acrobot.py              → Baseline
```

### **ANALYSIS (Aktiv):**
```
✅ quick_analysis.py                     → Primary analysis tool
✅ compare_multi_environment.py          → Multi-env comparison
✅ compare_all_systems.py                → System comparison
✅ statistical_analysis.py               → Statistical tests
✅ visualize_competitive.py              → Competitive viz
✅ visualize_regional_comparison.py      → Regional viz
✅ create_final_report.py                → Report generation
✅ monitor_competitive.py                → Live monitoring
✅ monitor_regional_training.py          → Live monitoring
✅ plot_utils.py                         → Utilities
```

### **DOCS (Aktiv):**
```
✅ README.md                             → Main readme
✅ CONTRIBUTING.md                       → Contributing guide
✅ SYSTEMATIC_TESTING_PLAN.md            → NEW! Testing plan
✅ TODAY_ACHIEVEMENTS.md                 → Progress tracking
✅ REGIONAL_INFRASTRUCTURE_QUICKSTART.md → Quickstart guide
✅ PAPER_OUTLINE.md                      → Paper planning
✅ requirements.txt                      → Dependencies
```

---

## 📦 ARCHIVIERUNG STRATEGIE

```
legacy_archive/
├── core_legacy/          → Alte core Module
├── training_legacy/      → Alte training Scripts
├── analysis_legacy/      → Alte analysis Scripts
└── docs_legacy/          → Alte Dokumentation
```

