import numpy as np

class AutoTuner:
    """
    Phase 5.7 – η-Reaktivität Kalibrator
    Passt Meta-Sensitivität dynamisch an in den ersten Episoden.
    """
    def __init__(self, target_corr=-0.3, window=20):
        self.target_corr = target_corr
        self.window = window
        self.buffer_td = []
        self.buffer_eta = []
        self.active = True

    def record(self, td_error, eta):
        if not self.active:
            return
        self.buffer_td.append(td_error)
        self.buffer_eta.append(eta)
        if len(self.buffer_td) > self.window:
            self.buffer_td.pop(0)
            self.buffer_eta.pop(0)

    def tune(self, meta_opt):
        if not self.active or len(self.buffer_td) < self.window:
            return meta_opt.lr, None

        corr = np.corrcoef(self.buffer_td, self.buffer_eta)[0,1]
        diff = corr - self.target_corr

        # Lernrate anpassen, um Ziel-Korrelation zu erreichen
        new_lr = np.clip(meta_opt.lr * (1 + 0.5 * diff), 0.05, 0.2)

        # Wenn Ziel erreicht, Routine deaktivieren
        if abs(diff) < 0.05:
            self.active = False

        return new_lr, corr
