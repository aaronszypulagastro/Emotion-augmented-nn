# self_regulation_controller.py
# Koordiniert EmotionEngine (reaktiv), ZTE (feed-forward) und MetaOptimizer (reflektierend)

from dataclasses import dataclass
import numpy as np

@dataclass
class SRCConfig:
    # Gewichte der drei Regelkreise (werden adaptiv feinjustiert)
    w_emotion: float = 1.0      # reaktiv (apply_zone_response)
    w_zte: float = 1.0          # vorausschauend (ZoneTransitionEngine)
    w_meta: float = 1.0         # reflektierend (MetaOptimizer)
    # Sicherheitsgrenzen
    eta_min: float = 1e-5
    eta_max: float = 7e-3
    gain_min: float = 0.8
    gain_max: float = 1.6
    # De-conflict: wie stark wir ZTE vs. Meta mischen
    blend: float = 0.6          # 0..1 (0=Meta dominiert, 1=ZTE dominiert)
    # Sanfte Glättung der finalen η-Änderung
    eta_smoothing: float = 0.3  # 0..1 (höher = glatter)

class SelfRegulationController:
    """
    Orchestriert 3 Ebenen:
      1) EmotionEngine.apply_zone_response(...)   [reaktiv]
      2) ZoneTransitionEngine.update/apply(...)   [vorausschauend]
      3) MetaOptimizer.update(...)                [reflektierend]
    Gibt eine final regulierte η zurück und passt gain der EmotionEngine an.
    """

    def __init__(self, emotion_engine, zte, meta_opt, cfg: SRCConfig = SRCConfig()):
        self.emotion = emotion_engine
        self.zte = zte
        self.meta = meta_opt
        self.cfg = cfg
        self._eta_prev = None
        self.last_zone_pred = None
        self.last_trend = 0.0

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
        self.last_zone_pred = zone_pred

        # 3) Meta-Optimizer (Reflexion über Reward-Trend)
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
        eta_mixed = self.cfg.blend * eta_zte + (1.0 - self.cfg.blend) * eta_meta

        # Glättung gegenüber Vorwert (verhindert Sprünge)
        if self._eta_prev is None:
            eta_final = eta_mixed
        else:
            eta_final = (1 - self.cfg.eta_smoothing) * eta_mixed + self.cfg.eta_smoothing * self._eta_prev

        # Clamp
        eta_final = float(np.clip(eta_final, self.cfg.eta_min, self.cfg.eta_max))
        self._eta_prev = eta_final
        return eta_final

    # optional: kompaktes Statusobjekt für Logging
    def status(self):
        return {
            "zone_pred": self.last_zone_pred,
            "trend": self.last_trend,
            "gain": getattr(self.emotion, "gain", None)
        }
