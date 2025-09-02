# recommender.py
# Core online recommender with warm-up, decay, diversity, and quality boost.

from __future__ import annotations
from collections import deque
from typing import Dict, Set, Tuple
import numpy as np

from utils import l2_normalize

class ImageRecommender:
    def __init__(
        self,
        E_unit: np.ndarray,
        quality_mask: np.ndarray | None = None,
        quality_scores: np.ndarray | None = None,   # Q_NORM (0..1), optional
        ahash_groups: Dict[int, Set[int]] | None = None,  # for exact dupes
        alpha=0.85, eta=1.6, warmup_n=5,
        recent_k=50, recent_weight=0.8, focus_gamma=1.3,
        diversity_last_k=20, diversity_min_cos=0.92,
        hide_exact_dupes=True, rng=None
    ):
        self.E = l2_normalize(np.asarray(E_unit), axis=1)
        self.N, self.D = self.E.shape

        self.alpha = float(alpha); self.eta = float(eta)
        self.eta0 = float(eta); self.eta_min = 0.10
        self.eta_decay_span = 50; self.use_decay = True

        self.warmup_n = int(warmup_n); self.focus_gamma = float(focus_gamma)
        self.recent_k = int(recent_k); self.recent_weight = float(recent_weight)
        self.diversity_last_k = int(diversity_last_k); self.diversity_min_cos = float(diversity_min_cos)
        self.hide_exact_dupes = bool(hide_exact_dupes)

        self.rng = rng or np.random.default_rng(42)
        base = self.E.mean(axis=0) + self.rng.normal(0.0, 0.05, size=self.D)
        self.preference = l2_normalize(base).astype(np.float32)

        self.quality_mask = quality_mask if quality_mask is not None else np.ones(self.N, bool)
        self.q_norm = quality_scores  # optional (0..1)
        self.ahash_groups = ahash_groups or {}

        self.seen = set()
        self._last_shown = deque(maxlen=self.diversity_last_k)
        self._recent = deque(maxlen=self.recent_k)

        self._warm_count = 0; self._warm_sum = np.zeros(self.D, np.float32)
        self._warm_weight = 0.0; self._warmed_up = (self.warmup_n <= 0)
        self._warm_order = self.rng.permutation(self.N); self._warm_ptr = 0
        self._updates_total = 0; self._eta_updates = 0

    #human louis added
    # def get_user_profile(self) -> np.ndarray:
    #     return self.preference

    # --- setters ---
    def set_quality_mask(self, mask: np.ndarray):
        self.quality_mask = mask

    def set_diversity(self, last_k: int, min_cos: float):
        self.diversity_last_k = int(last_k); self.diversity_min_cos = float(min_cos)
        self._last_shown = deque(list(self._last_shown), maxlen=self.diversity_last_k)

    # --- helpers ---
    def _passes_diversity(self, idx: int) -> bool:
        if not self._last_shown or self.diversity_min_cos >= 0.999:
            return True
        e = self.E[idx]
        return all(float(np.dot(e, self.E[li])) < self.diversity_min_cos for li in self._last_shown)

    def _mask_candidates(self) -> np.ndarray:
        mask = self.quality_mask.copy()
        if self.seen:
            mask[list(self.seen)] = False
        if self.hide_exact_dupes and self.seen:
            banned = set()
            for i in self.seen:
                banned |= self.ahash_groups.get(i, set())
            if banned:
                mask[list(banned)] = False
        return mask

    def _pick_warmup(self):
        mask = self._mask_candidates()
        while self._warm_ptr < self.N:
            idx = int(self._warm_order[self._warm_ptr]); self._warm_ptr += 1
            if not mask[idx] or not self._passes_diversity(idx): continue
            return idx
        return None

    def _update(self, idx: int, feedback: float):
        e = self.E[idx]; self.seen.add(idx); self._last_shown.append(idx)
        self._updates_total += 1

        # warm-up
        if not self._warmed_up:
            self._warm_count += 1
            if feedback > 0:
                self._warm_sum += feedback * e
                self._warm_weight += feedback
            if (self._warm_count >= self.warmup_n) or (self._warm_weight >= 1.0):
                if self._warm_weight > 0:
                    delta = self._warm_sum / max(self._warm_weight, 1e-9)
                    self.preference = l2_normalize(self.alpha * self.preference + self.eta * delta)
                self._warmed_up = True; self._warm_sum[:] = 0; self._warm_weight = 0.0
            return

        # dynamic eta decay
        if self.use_decay:
            self._eta_updates += 1
            t = min(self._eta_updates / max(1.0, float(self.eta_decay_span)), 1.0)
            self.eta = self.eta_min + (self.eta0 - self.eta_min) * (1.0 - t)
        else:
            self.eta = self.eta0

        # online EMA with recent centroid
        if feedback > 0:
            self._recent.append(feedback * e)
        recent_centroid = (np.mean(np.stack(self._recent, axis=0), axis=0)
                           if (self.recent_weight > 0 and len(self._recent) > 0) else 0.0)
        update_vec = self.alpha * self.preference + self.eta * feedback * e + self.recent_weight * recent_centroid
        self.preference = l2_normalize(update_vec)

    def recommend_next_smart(
        self, pool_k=300, lambda_div=0.50, quality_boost=0.25, near_dupe_thr=0.95
    ) -> Tuple[int | None, float | None]:
        if len(self.seen) >= self.N:
            return None, None
        if not self._warmed_up:
            idx = self._pick_warmup()
            return (idx, None) if idx is not None else (None, None)

        sims = self.E @ self.preference
        mask = self._mask_candidates() #quality mask used here
        sims = np.where(mask, sims, -np.inf)
        k = min(pool_k, np.isfinite(sims).sum())
        if k <= 0:
            return None, None
        pool = np.argpartition(-sims, range(k))[:k]

        # near-duplicate barrier
        if len(self._last_shown):
            sim_to_sel = self.E[pool] @ self.E[list(self._last_shown)].T
            too_close = (sim_to_sel.max(axis=1) >= float(near_dupe_thr))
            pool = pool[~too_close]
            if pool.size == 0:
                pool = np.argpartition(-sims, range(k))[:k]

        if not len(self._last_shown):
            j = int(pool[np.argmax(sims[pool])])
            return j, float(sims[j])

        sim_to_sel = self.E[pool] @ self.E[list(self._last_shown)].T
        novelty = 1.0 - sim_to_sel.max(axis=1)  # higher = more different
        q = (self.q_norm[pool].astype(np.float32) if self.q_norm is not None else 0.0)
        pref_part = sims[pool]

        score = (1.0 - lambda_div) * pref_part + lambda_div * novelty + float(quality_boost) * q
        j_local = int(np.argmax(score));
        j = int(pool[j_local])
        return j, float(sims[j])
