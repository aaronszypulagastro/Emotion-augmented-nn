# self_regulation_controller.py
# Koordiniert EmotionEngine (reaktiv), ZTE (feed-forward), MetaOptimizer (reflektierend) und EPRU (prädiktiv)

from dataclasses import dataclass
import numpy as np
from typing import Optional, Dict, Any

@dataclass
class SRCConfig:
    # Gewichte der vier Regelkreise (werden adaptiv feinjustiert)
    w_emotion: float = 1.0      # reaktiv (apply_zone_response)
    w_zte: float = 1.0          # vorausschauend (ZoneTransitionEngine)
    w_meta: float = 1.0         # reflektierend (MetaOptimizer)
    w_epru: float = 0.8         # prädiktiv (EmotionPredictiveRegulationUnit)
    # Sicherheitsgrenzen
    eta_min: float = 1e-5
    eta_max: float = 7e-3
    gain_min: float = 0.8
    gain_max: float = 1.6
    # De-conflict: wie stark wir ZTE vs. Meta mischen
    blend: float = 0.6          # 0..1 (0=Meta dominiert, 1=ZTE dominiert)
    # Sanfte Glättung der finalen η-Änderung
    eta_smoothing: float = 0.3  # 0..1 (höher = glatter)
    # EPRU-spezifische Parameter
    epru_confidence_threshold: float = 0.7
    epru_intervention_strength: float = 0.3

class SelfRegulationController:
    """
    Orchestriert 4 Ebenen:
      1) EmotionEngine.apply_zone_response(...)   [reaktiv]
      2) ZoneTransitionEngine.update/apply(...)   [vorausschauend]
      3) MetaOptimizer.update(...)                [reflektierend]
      4) EmotionPredictiveRegulationUnit(...)     [prädiktiv]
    Gibt eine final regulierte η zurück und passt gain der EmotionEngine an.
    """

    def __init__(self, emotion_engine, zte, meta_opt, epru=None, cfg: SRCConfig = SRCConfig()):
        self.emotion = emotion_engine
        self.zte = zte
        self.meta = meta_opt
        self.epru = epru  # EmotionPredictiveRegulationUnit
        self.cfg = cfg
        self._eta_prev = None
        self.last_zone_pred = None
        self.last_trend = 0.0
        # Zonen-Hysterese
        self._last_zone = None
        self._zone_dwell = 0
        self._zone_min_dwell = 5  # Mindestverweildauer in Episoden

    def step(self, reward_ep: float, td_error_ep: float, eta: float) -> float:
        """
        Wird 1x am Episodenende aufgerufen.
        - Aktualisiert zuerst reaktiv die Emotion (Zone-Response).
        - Holt dann ZTE-Vorhersage und Meta-Feedback.
        - Mischt beide zu einer konsistenten Anpassung von η & gain.
        """
        # 1) Reaktive Zone-Response (leichte, stetige Regulierung)
        #    -> Hält System in praktikablen Zonen (kein harter Eingriff)
        if hasattr(self.emotion, "apply_zone_response"):
            self.emotion.apply_zone_response(self.emotion.value, td_error_ep)

        # 2) Vorausschauende ZTE-Korrektur (Feed-Forward)
        zone_pred = self.zte.update(self.emotion.value, td_error_ep, eta)
        if zone_pred is not None:
            # ZTE wirkt direkt auf emotion.gain / interne Ziele
            self.zte.apply_to_emotion_engine(self.emotion)
        # Zonen-Hysterese anwenden
        pred = zone_pred or "neutral"
        if self._last_zone is None:
            self._last_zone = pred
            self._zone_dwell = 1
        else:
            if pred == self._last_zone:
                self._zone_dwell += 1
            else:
                if self._zone_dwell >= self._zone_min_dwell:
                    self._last_zone = pred
                    self._zone_dwell = 1
                else:
                    pred = self._last_zone
                    self._zone_dwell += 1

        self.last_zone_pred = pred

        # 3) EPRU - Prädiktive η-Regelung (Phase 6.0)
        epru_adjustment = 0.0
        epru_confidence = 0.0
        epru_debug = {}
        if self.epru is not None:
            # EPRU-Historie aktualisieren
            self.epru.update_history(self.emotion.value, td_error_ep, eta, reward_ep)
            
            # Antizipative η-Anpassung berechnen
            epru_adjustment, epru_confidence, epru_debug = self.epru.get_anticipatory_eta_adjustment(
                self.emotion.value, td_error_ep, eta
            )
            
            # EPRU-Training (nur bei ausreichender Historie)
            if hasattr(self, '_episode_count'):
                self.epru.train_step(self._episode_count)

        # 4) Meta-Optimizer (Reflexion über Reward-Trend)
        meta_fb = self.meta.update(reward_ep, eta, self.emotion.gain, self.emotion.value)
        if meta_fb is not None:
            self.last_trend = float(meta_fb.get("trend", 0.0))

        # --- De-conflict & Fusion ---
        # Wir formen aus ZTE-Signal (qualitativ) und Meta-Signal (quantitativ) eine gemeinsame η-Anpassung.
        eta_zte = eta
        if zone_pred == "exploration_soon":
            # etwas mehr Lernen, leicht geringerer Gain
            eta_zte = eta * 1.12
            self.emotion.gain = float(np.clip(self.emotion.gain * 0.98, self.cfg.gain_min, self.cfg.gain_max))
        elif zone_pred == "stabilization_soon":
            # etwas konservativer lernen, leicht höherer Gain (Belohnungen konsolidieren)
            eta_zte = eta * 0.92
            self.emotion.gain = float(np.clip(self.emotion.gain * 1.02, self.cfg.gain_min, self.cfg.gain_max))

        eta_meta = eta
        if meta_fb is not None:
            eta_meta = float(meta_fb["eta"])
            self.emotion.gain = float(np.clip(meta_fb["gain"], self.cfg.gain_min, self.cfg.gain_max))

        # Mischung: ZTE (feed-forward) bekommt default mehr Gewicht (cfg.blend ~0.6).
        # Meta wirkt als Korrektur (trendbasiert).
        # EPRU wirkt als prädiktive Komponente (nur bei hoher Confidence).
        eta_mixed = self.cfg.blend * eta_zte + (1.0 - self.cfg.blend) * eta_meta
        
        # EPRU-Integration: Prädiktive Anpassung nur bei hoher Confidence
        if epru_confidence >= self.cfg.epru_confidence_threshold:
            epru_weight = self.cfg.w_epru * epru_confidence
            eta_mixed = eta_mixed + epru_weight * epru_adjustment

        # Glättung gegenüber Vorwert (verhindert Sprünge)
        if self._eta_prev is None:
            eta_final = eta_mixed
        else:
            eta_final = (1 - self.cfg.eta_smoothing) * eta_mixed + self.cfg.eta_smoothing * self._eta_prev

        # Clamp
        eta_final = float(np.clip(eta_final, self.cfg.eta_min, self.cfg.eta_max))
        self._eta_prev = eta_final
        return eta_final
    
    def set_episode_count(self, episode: int):
        """Setzt die aktuelle Episode für EPRU-Training."""
        self._episode_count = episode
    
    def get_epru_debug_info(self) -> Dict[str, Any]:
        """Gibt EPRU-Debug-Informationen zurück."""
        if self.epru is not None:
            return self.epru.get_debug_info()
        return {}

    # optional: kompaktes Statusobjekt für Logging
    def status(self):
        return {
            "zone_pred": self.last_zone_pred,
            "trend": self.last_trend,
            "gain": getattr(self.emotion, "gain", None)
        }
