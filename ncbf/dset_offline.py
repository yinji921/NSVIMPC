from typing import NamedTuple

import ipdb
import numpy as np
from og.dyn_types import THFloat, TObs, TState
from og.normalization import MeanStd

from ncbf.offline.train_offline_alg import Traj


class S:
    VX, VY, WZ, WF, WR, PSI, X, Y, EPSI, EY, S = range(11)


class DsetOffline(NamedTuple):
    bT_obs: list[TObs]
    bTh_h: list[THFloat]
    bT_x: list[TState] = []
    is_obs_padded: bool = False

    @property
    def T_min(self):
        return min(len(T_obs) for T_obs in self.bT_obs)

    def remove_shorter_than(self, T: int):
        bT_obs, bTh_h = [], []
        for T_obs, Th_h in zip(self.bT_obs, self.bTh_h):
            if len(T_obs) >= T:
                bT_obs.append(T_obs)
                bTh_h.append(Th_h)
        return DsetOffline(bT_obs, bTh_h)

    def normalize(self):
        # Compute mean and std.
        # self.bT_obs shape is (n_trajs, T, n_obs)
        b_obs = np.concatenate(self.bT_obs, axis=0) # shape: (n_trajs * T, n_obs)
        obs_mean = np.mean(b_obs, axis=0)
        obs_std = np.std(b_obs, axis=0)

        bT_obs_norm = [(T_obs - obs_mean) / obs_std for T_obs in self.bT_obs]
        return DsetOffline(bT_obs_norm, self.bTh_h), obs_mean, obs_std

    def normalize_with(self, norm: MeanStd):
        bT_obs_norm = [norm.normalize(T_obs) for T_obs in self.bT_obs]
        return DsetOffline(bT_obs_norm, self.bTh_h)

    def pad_obs_final(self) -> "DsetOffline":
        # Turn bT_obs to bTp1_obs, where the final observation is only used as a part of the state_to and is a dummy.
        assert not self.is_obs_padded
        bTp1_obs = [np.concatenate([T_obs, T_obs[-1][None, :]], axis=0) for T_obs in self.bT_obs]
        return DsetOffline(bTp1_obs, self.bTh_h, self.bT_x, is_obs_padded=True)

    def sample_trajs(
        self, n_trajs: int, T_sample: int, rng: np.random.Generator, replace: bool, p_final: float = None
    ) -> Traj:
        assert self.is_obs_padded

        n_dset_trajs = len(self.bT_obs)
        traj_idxs = rng.choice(n_dset_trajs, n_trajs, replace=replace)
        bTp1_obs, bTh_h, bT_isterm = [], [], []

        for traj_idx in traj_idxs:
            Sp1_obs, Sh_h = self.bT_obs[traj_idx], self.bTh_h[traj_idx]
            assert len(Sp1_obs) == len(Sh_h) + 1

            # We need one more obs than h_h.
            n_idxs_sample = len(Sp1_obs) - (T_sample + 1) + 1
            p = None
            if p_final is not None:
                p_final_ = max(p_final, 1 / n_idxs_sample)
                p = np.full(n_idxs_sample, (1 - p_final_) / n_idxs_sample)
                p[-1] = p_final
                p = p / p.sum()

            idx0 = rng.choice(n_idxs_sample, p=p)
            # idx0 = rng.integers(0, len(T_obs) - (T_sample + 1))
            idx1 = idx0 + T_sample
            T_isterm = np.zeros(T_sample, dtype=bool)
            Tp1_obs = Sp1_obs[idx0 : idx1 + 1]
            Th_h = Sh_h[idx0:idx1]
            # ipdb.set_trace()
            # The last state terminates only if it is a crash, i.e., it is unsafe.
            # We terminate if it is the last state AND it is a crash
            T_lastunsafe = np.max(Th_h[-1]) > 0.0
            T_isterm[-1] = (idx1 == len(Sp1_obs) - 1) & T_lastunsafe #TODO: I this this should be "or", not "and"

            assert len(Tp1_obs) == T_sample + 1
            assert len(Th_h) == T_sample

            bTp1_obs.append(Tp1_obs)
            bTh_h.append(Th_h)
            bT_isterm.append(T_isterm)

        bTp1_obs, bTh_h, bT_isterm = np.stack(bTp1_obs), np.stack(bTh_h), np.stack(bT_isterm)
        # ipdb.set_trace()
        return Traj(bTp1_obs, bTh_h, bT_isterm)
