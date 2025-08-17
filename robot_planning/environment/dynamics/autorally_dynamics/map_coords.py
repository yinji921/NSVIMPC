import functools as ft

# import equinox
# from equinox.debug import breakpoint_if
import jax
import jax.debug as jd
import jax.lax as lax
import jax.numpy as np
import jax.numpy as jnp
import numpy as onp
from og.tree_utils import tree_where
import ast


def dist_to_segment(p, a, b):
    assert p.shape == a.shape == b.shape == (2,)
    pa = p - a
    ba = b - a
    h = jnp.clip(jnp.dot(pa, ba) / jnp.dot(ba, ba), 0.0, 1.0)
    return jnp.linalg.norm(pa - ba * h), h

class MapCA:
    def __init__(self, track_path, obstacles_config_data=None):
        """Load a map of a track which consists of line segments
        stored as dense (x, y) coordinates and curvatures at each waypoint."""
        # file_name = 'maps/CCRF/CCRF_2021-01-10.npz'
        # file_name = 'maps/CCRF/ccrf_track_optimal.npz'
        file_name = track_path
        track_dict = np.load(file_name)
        try:
            p_x = track_dict["X_cen_smooth"]
            p_y = track_dict["Y_cen_smooth"]
        except KeyError:
            p_x = track_dict["pts"][:, 0]
            p_y = track_dict["pts"][:, 1]
            self.rho = track_dict["curvature"]
        # centerline coord
        self.p = np.array([p_x, p_y])
        # vec from one point to next
        dif_vecs = self.p[:, 1:] - self.p[:, :-1]
        self.dif_vecs = dif_vecs.copy()
        self.slopes = dif_vecs[1, :] / dif_vecs[0, :]
        # midpoint of each line segment
        self.midpoints = self.p[:, :-1] + dif_vecs / 2
        # total distance along the path to each coordinate
        self.s = np.cumsum(np.linalg.norm(dif_vecs, axis=0))

        self.heading = np.arctan2(self.dif_vecs[1, :], self.dif_vecs[0, :])
        self.heading_rate = np.diff(self.heading, append=self.heading[0].reshape(-1))
        self.heading_rate = np.where(self.heading_rate > np.pi, self.heading_rate - 2 * np.pi, self.heading_rate)
        self.heading_rate = np.where(self.heading_rate < -np.pi, self.heading_rate + 2 * np.pi, self.heading_rate)

        self.s_total = onp.array(self.s[-1] + np.linalg.norm(self.p[:, -1] - self.p[:, 0]))
        self.obstacles_config_data = obstacles_config_data

    def get_obstacles(self):
        if self.obstacles_config_data is not None:
            obstacles = np.asarray(ast.literal_eval(self.obstacles_config_data.get("my_collision_checker_for_collision", "obstacles")))
            obstacles_radius = np.asarray(ast.literal_eval(self.obstacles_config_data.get("my_collision_checker_for_collision", "obstacles_radius")))
            obstacles_info = np.concatenate((obstacles, obstacles_radius[:, None]), axis=1)
            return obstacles_info

    def localize_one(self, M, psi, return_index=False):
        dists = np.linalg.norm(np.subtract(M.reshape((-1, 1)), self.midpoints), axis=0)
        mini = np.argmin(dists)
        p0 = self.p[:, mini]
        p1 = self.p[:, mini + 1]
        # plt.plot(M[0], M[1], 'x')
        # plt.plot(p0[0], p0[1], 'o')
        # plt.plot(p1[0], p1[1], 'o')
        ortho = -1 / self.slopes[mini]
        a = M[1] - ortho * M[0]
        a_0 = p0[1] - ortho * p0[0]
        a_1 = p1[1] - ortho * p1[0]
        printi = 0
        if a_0 < a < a_1 or a_1 < a < a_0:
            norm_dist = (
                np.sign(np.cross(p1 - p0, M - p0)) * np.linalg.norm(np.cross(p1 - p0, M - p0)) / np.linalg.norm(p1 - p0)
            )
            s_dist = np.linalg.norm(np.dot(M - p0, p1 - p0))
        else:
            printi = 1
            norm_dist = np.sign(np.cross(p1 - p0, M - p0)) * np.linalg.norm(M - p0)
            s_dist = 0
        s_dist += self.s[mini]
        head_dist = psi - np.arctan2(self.dif_vecs[1, mini], self.dif_vecs[0, mini])
        if head_dist > np.pi:
            # print(psi, np.arctan2(self.dif_vecs[1, mini], self.dif_vecs[0, mini]))
            head_dist -= 2 * np.pi
            # print(norm_dist, s_dist, head_dist * 180 / np.pi)
        elif head_dist < -np.pi:
            head_dist += 2 * np.pi
            # print(norm_dist, s_dist, head_dist * 180 / np.pi)
        # if printi:
        #     print(norm_dist, s_dist, head_dist*180/np.pi)
        #     printi=0
        # plt.show()
        if return_index:
            return head_dist, norm_dist, s_dist, mini
        else:
            return head_dist, norm_dist, s_dist

    def localize(self, M, psi, relative_to_s=None, return_ref_yaw_rate=False):
        """Convert N query points in cartesian coordinates - (X, Y) and Psi -
        to map-based curvilinear coordinates."""
        num_pts = M.shape[1]
        # If relative_to_s is provided, pick the waypoint nearest to that distance
        if relative_to_s is not None:
            dists = np.abs(relative_to_s - self.s).reshape(-1, 1)
        else:  # otherwise, pick the midpoint nearest to each of the query points
            dists = np.linalg.norm(
                np.subtract(
                    np.expand_dims(M, axis=1),
                    np.tile(np.expand_dims(self.midpoints, axis=2), (1, 1, num_pts)),
                ),
                axis=0,
            )
        # index of nearest midpoint on path (ie line segment) to each query point
        minis = np.argmin(dists, axis=0)

        # waypoint defining start of line segment nearest to each query point
        p0s = self.p[:, minis]
        # waypoint defining end of line segment nearest to each query point
        p1s = self.p[:, minis + 1]
        # plt.plot(M[0], M[1], 'x')
        # plt.plot(p0[0], p0[1], 'o')
        # plt.plot(p1[0], p1[1], 'o')

        orthos = -1 / self.slopes[minis]
        a = M[1, :] - orthos * M[0, :]
        a_0s = p0s[1, :] - orthos * p0s[0, :]
        a_1s = p1s[1, :] - orthos * p1s[0, :]
        # mask = np.where(((a_0s < a) & (a < a_1s)) | ((a_1s < a) & (a < a_0s)))[0]
        # norm_dist = np.zeros((1, num_pts))
        # s_dist = np.zeros_like(norm_dist)
        # norm_dist[0, :] = np.sign(np.cross(p1s - p0s, M - p0s, axis=0)) * np.linalg.norm(np.cross(p1s - p0s, M - p0s, axis=0).reshape((-1, 1)), axis=1) / np.linalg.norm(p1s - p0s, axis=0)
        # s_dist[0, mask] = np.linalg.norm(np.matmul(np.expand_dims(M - p0s, axis=1), np.expand_dims(p1s - p0s, axis=2)), axis=0)
        # if we are along a line segment, get the normal distance to the segment,
        # else we are at a corner between segments and simply egt the distance to the corner point
        norm_dists = np.where(
            ((a_0s < a) & (a < a_1s)) | ((a_1s < a) & (a < a_0s)),
            np.sign(np.cross(p1s - p0s, M - p0s, axis=0))
            * np.linalg.norm(np.cross(p1s - p0s, M - p0s, axis=0).reshape((-1, 1)), axis=1)
            / np.linalg.norm(p1s - p0s, axis=0),
            np.sign(np.cross(p1s - p0s, M - p0s, axis=0)) * np.linalg.norm(M - p0s, axis=0),
        )
        # if we are along a line segment, get the parallel distance along the nearest segment
        # else we are at a corner between segments and the along-path error is 0
        s_dists = np.where(
            ((a_0s < a) & (a < a_1s)) | ((a_1s < a) & (a < a_0s)),
            np.linalg.norm(
                np.matmul(np.expand_dims(M - p0s, axis=1), np.expand_dims(p1s - p0s, axis=2)),
                axis=0,
            ),
            0,
        )

        # If we're localizing globally along the path, add the cumulative distance to
        # this point
        if relative_to_s is None:
            s_dists += self.s[minis]

        head_dists = psi - np.arctan2(self.dif_vecs[1, minis], self.dif_vecs[0, minis])
        head_dists = np.where(head_dists > np.pi, head_dists - 2 * np.pi, head_dists)
        head_dists = np.where(head_dists < -np.pi, head_dists + 2 * np.pi, head_dists)
        # if printi:
        #     print(norm_dist, s_dist, head_dist*180/np.pi)
        #     printi=0
        # plt.show()

        # If we need to return the yaw rate of the reference path, do that
        # This will be the rate of change of the path's heading with respect to
        # distance along the path. Multiply by velocity along the path to get the
        # time rate of change
        if return_ref_yaw_rate:
            heading_rate = self.heading_rate[minis]
            return head_dists, norm_dists, s_dists, heading_rate
        else:
            return head_dists, norm_dists, s_dists

    def get_cur_reg_from_s(self, s):
        nearest = np.argmin(np.abs(s.reshape((-1, 1)) - self.s.reshape((1, -1))), axis=1)
        x0 = self.p[0, nearest]
        y0 = self.p[1, nearest]
        theta0 = np.arctan2(self.dif_vecs[1, nearest], self.dif_vecs[0, nearest])
        s = self.s[nearest]
        curvature = np.array(self.rho)[nearest]
        return x0, y0, theta0, s, curvature

    def get_lerp_from_s(self, s):
        # Linear interpolated version of the above.
        n_points = len(self.p[0])
        with jax.ensure_compile_time_eval():
            assert len(self.rho) == n_points
            assert len(self.s) == len(self.heading) == n_points - 1
            assert not np.allclose(self.p[0], self.p[-1])

        # Include both 0 and the additional distance from last point to the first point.
        s_aug = np.concatenate([np.zeros(1), self.s, self.s_total[None]], axis=0)
        assert s_aug.ndim == 1

        idx0 = np.searchsorted(s_aug, s, side="right") - 1
        idx1_aug = idx0 + 1
        idx1_wrap = idx1_aug % n_points

        s0, s1 = s_aug[idx0], s_aug[idx1_aug]

        # jd.print("s[idx0]: {}, s: {}, s[idx1]: {}, idx0: {}, idx1: {}", s0, s, s1, idx0, idx1_wrap)

        alpha = (s - s0) / (s1 - s0)

        x0, x1 = self.p[0, idx0], self.p[0, idx1_wrap]
        y0, y1 = self.p[1, idx0], self.p[1, idx1_wrap]
        rhos = np.array(self.rho)
        kappa0, kappa1 = rhos[idx0].squeeze(), rhos[idx1_wrap].squeeze()

        x = (1 - alpha) * x0 + alpha * x1
        y = (1 - alpha) * y0 + alpha * y1
        kappa = (1 - alpha) * kappa0 + alpha * kappa1

        # Heading defined at the line segment, not at the vertices.
        theta = self.heading[idx0]

        return x, y, theta, s, kappa

    def localize_lerp(self, pos, psi):
        assert pos.shape == (2,)
        assert psi.shape == tuple()

        n_points = len(self.p[0])
        with jax.ensure_compile_time_eval():
            assert len(self.rho) == n_points
            assert len(self.s) == len(self.heading) == n_points - 1
            assert not np.allclose(self.p[0], self.p[-1])

            # Include both 0 and the additional distance from last point to the first point.
            s_aug = np.concatenate([np.zeros(1), self.s, self.s_total[None]], axis=0)
            assert s_aug.shape == (n_points + 1,)

        p_points = self.p.T

        # Project the position onto the line segment.
        # - To do this, we assume that the centerline is a set of line segments.
        # - We find the "closest line segment" by looking at the minimum distance to each line segment,
        #   then projecting the point onto the line segment.
        # - Note that the resulting s will only be continuous if we are following the centerline exactly.
        #   Otherwise, if e_y is nonzero, then s will jump when the closest line segment changes.

        # There are n_points vertices and n_points midpoints, taking the wraparound into account.
        pp1_points = np.concatenate([p_points, p_points[[0], :]], axis=0)
        assert pp1_points.shape == (n_points + 1, 2)

        p_pt0 = p_points
        p_pt1 = jnp.roll(p_points, -1, axis=0)
        p_dist, p_h = jax.vmap(ft.partial(dist_to_segment, pos))(p_pt0, p_pt1)
        idx = jnp.argmin(p_dist)

        # We've now found which line segment we're on. Compute s and e_y.
        #     p_dist gives us the correct magnitude, but not the correct sign.
        pt0, pt1 = p_pt0[idx], p_pt1[idx]

        seg_vec = pt1 - pt0
        seg_vec = seg_vec / np.linalg.norm(seg_vec)
        seg_perp = np.array([-seg_vec[1], seg_vec[0]])
        e_y = np.dot(seg_perp, pos - pt0)

        h = p_h[idx]
        s = (1-h) * s_aug[idx] + h * s_aug[idx + 1]

        # Finally, compute psi.
        theta = self.heading[idx]
        e_psi = psi - theta
        #   Wrap e_psi to [-pi, pi].
        e_psi = (e_psi + np.pi) % (2 * np.pi) - np.pi

        return e_psi, e_y, s

    def get_cartesian_from_local(self, e_psi, e_y, s):
        x, y, theta, _, _ = self.get_lerp_from_s(s)
        pos0 = np.array([x, y])
        dpos = np.array([np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)])
        pos = pos0 + e_y * dpos
        psi = theta + e_psi
        return pos, psi
