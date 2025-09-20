import numpy as np

class PolityEnv:
    """
    政策シミュレーション（層反応つき）
    state = [eco, budget, sat_y, sat_s, sat_b] ∈ [-1, 1]^5
    actions = 0:減税, 1:増税, 2:教育投資, 3:環境規制, 4:景気刺激
    """
    def __init__(self, seed=42, episode_len=20):
        self.rng = np.random.default_rng(seed)
        self.episode_len = episode_len
        self.t = 0

        # 人口比（正規化しておく）
        self.wY, self.wS, self.wB = 0.35, 0.35, 0.30

        # 係数（調整しやすいように一箇所に集約）
        self.K = {
            "tax_cut":     dict(eco=+0.08, budget=-0.12, y=+0.00, s=+0.05, b=+0.10),
            "tax_raise":   dict(eco=-0.06, budget=+0.12, y=-0.05, s=-0.05, b=-0.08),
            "edu_invest":  dict(eco=+0.03, budget=-0.15, y=+0.12, s=+0.00, b=+0.00),
            "env_reg":     dict(eco=-0.04, budget=+0.00, y=+0.05, s=+0.05, b=-0.12),
            "stimulus":    dict(eco=+0.10, budget=-0.18, y=+0.05, s=+0.03, b=+0.03),
        }

        # 罰則係数
        self.lmb_budget = 0.25
        self.lmb_eco    = 0.10

        self.low  = np.array([-1, -1, -1, -1, -1], dtype=np.float32)
        self.high = np.array([+1, +1, +1, +1, +1], dtype=np.float32)
        self.reset()

    def reset(self):
        self.t = 0
        # 初期は少しバラつかせる
        eco    = self.rng.normal(0.0, 0.1)
        budget = self.rng.normal(0.0, 0.1)
        sat_y  = self.rng.normal(0.0, 0.1)
        sat_s  = self.rng.normal(0.0, 0.1)
        sat_b  = self.rng.normal(0.0, 0.1)
        self.state = np.clip(np.array([eco, budget, sat_y, sat_s, sat_b], dtype=np.float32), self.low, self.high)
        return self.state.copy()

    def step(self, action: int):
        eco, budget, y, s, b = self.state

        if action == 0:   eff = self.K["tax_cut"]
        elif action == 1: eff = self.K["tax_raise"]
        elif action == 2: eff = self.K["edu_invest"]
        elif action == 3: eff = self.K["env_reg"]
        elif action == 4: eff = self.K["stimulus"]
        else:             raise ValueError("invalid action")

        # ノイズ（現実の不確実性）
        noise = lambda scale: self.rng.normal(0.0, scale)

        eco    = np.clip(eco    + eff["eco"]    + noise(0.02), -1, 1)
        budget = np.clip(budget + eff["budget"] + noise(0.02), -1, 1)
        y      = np.clip(y      + eff["y"]      + noise(0.02), -1, 1)
        s      = np.clip(s      + eff["s"]      + noise(0.02), -1, 1)
        b      = np.clip(b      + eff["b"]      + noise(0.02), -1, 1)

        self.state = np.array([eco, budget, y, s, b], dtype=np.float32)

        weighted_sat = self.wY*y + self.wS*s + self.wB*b
        penalty = self.lmb_budget * max(0.0, -budget) + self.lmb_eco * max(0.0, -eco)
        reward = float(weighted_sat - penalty)

        self.t += 1
        done = (self.t >= self.episode_len)

        return self.state.copy(), reward, done, {}  # Gymnasium互換に近い戻り値
