import functools as ft
import pathlib
import pickle

import ipdb
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.collections import LineCollection
from og.color_utils import modify_hsv
from og.jax_utils import jax2np, jax_jit_np, jax_vmap
from og.path_utils import mkdir
from og.plot_utils import line_labels

from ncbf.ar_task import get_h_vector
from ncbf.avoid_utils import get_max_mc
from ncbf.dset_offline import S
from robot_planning.controllers.MPPI.MPPI import MPPI
from robot_planning.helper.convenience import get_ccrf_track, plot_track
from robot_planning.environment.ncbf import NCBF, AutorallyMPPINCBFCostEvaluator, MPPINCBFStochasticTrajectoriesSampler
from robot_planning.environment.robots.simulated_robot import SimulatedRobot
from robot_planning.factory.factories import robot_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.scripts.run_ar_cbf_online import get_and_set_config


def main(ckpt: pathlib.Path, crash_data_pkl: pathlib.Path):
    config_data, test_agent = get_and_set_config(clockwise=False)
    np.set_printoptions(linewidth=200)

    # Load the ncbf.
    ncbf = NCBF(ckpt)

    # Load the crash data.
    with open(crash_data_pkl, "rb") as f:
        crash_data = pickle.load(f)
    T_x, T_obs, Th_h = crash_data["T_x"], crash_data["T_obs"], crash_data["Th_h"]
    T_mppi_state = crash_data["T_mppi_state"]

    # Only plot last 50 timesteps.
    assert len(T_x) == len(T_obs) == len(Th_h) == len(T_mppi_state)

    last_kk = 40
    T_x, T_obs, Th_h = T_x[-last_kk:], T_obs[-last_kk:], Th_h[-last_kk:]
    T_mppi_state = T_mppi_state[-last_kk:]
    T, nx = T_x.shape

    # runs/online/0020-blueb-brown/ckpts/00008000/default/
    run_dir = ckpt.parent.parent.parent
    plot_dir = mkdir(run_dir / "dbgplots")

    #############################################################################
    # So.... what exactly is this mppi controller state that is making such a big difference?
    u_labels = ["Steering", "Throttle"]

    fig, axes = plt.subplots(2, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.axhline(0.0, color="C1", lw=0.8)
        ax.plot(T_mppi_state[0, ii, :], color="C0")
        ax.set_title(u_labels[ii])

    fig_path = plot_dir / "mppi_initialstate.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    #############################################################################
    # Next, see if we can reproduce a similar trajectory starting from the same state.
    agent: SimulatedRobot = factory_from_config(robot_factory_base, config_data, test_agent + "_agent")

    controller: MPPI = agent.controller
    controller.store_opt_traj = True
    sampler: MPPINCBFStochasticTrajectoriesSampler = controller.stochastic_trajectories_sampler
    sampler.ncbf_weights = ncbf.value_net.params

    cost_evaluator: AutorallyMPPINCBFCostEvaluator = controller.cost_evaluator

    # ---------------------------------------------------------------------------
    def rollout(x0, T_u):
        def body(x, u):
            x_new = controller.dynamics.propagate(x, u)
            return x_new, x_new

        x_final, T_xto = lax.scan(body, x0, T_u)
        Tp1_x = jnp.concatenate([x0[None, :], T_xto], axis=0)
        return Tp1_x

    # ---------------------------------------------------------------------------
    agent.reset_state(jnp.array(T_x[0, :-3]))
    controller.initial_control_sequence = jnp.array(T_mppi_state[0])

    T_x_new, TS_u_opt_new, TS_x_opt_new, T_mppistate_new, TS_psafe = [T_x[0, :-3]], [], [], [], []
    TbS_rollout, TbS_rollout2 = [], []
    for kk in range(T - 1):
        # Run the controller, but record everything.
        T_mppistate_new.append(np.array(controller.initial_control_sequence))

        #    First, call sample_detailed to get the constraint violation history during the rollout.
        state_cur_ = agent.get_state()
        v_ = controller.initial_control_sequence
        detailed_result = sampler.sample_detailed(
            state_cur_,
            v_,
            controller.control_horizon,
            controller.control_dim,
            controller.dynamics,
            cost_evaluator,
            control_bounds=controller.control_bounds,
        )
        S_psafe = detailed_result.info["T_psafe"]
        TS_psafe.append(np.array(S_psafe))

        state_next, cost, eval_time, action = agent.take_action_with_controller(return_time=True)
        T_x_new.append(np.array(agent.get_state()))

        xS_opttraj = np.array(controller.T_x_opt)
        S_opttraj = xS_opttraj.T
        TS_x_opt_new.append(S_opttraj)

        uS_opttraj = np.array(controller.T_u_opt)
        S_u_opttraj = uS_opttraj.T
        TS_u_opt_new.append(S_u_opttraj)

        TbS_rollout.append(controller.bT_rollout)

        # ----------------------------------------------------------------------------
        # Sanity check - the rollout controls should reproduce the rollout states.....
        bT_us = controller.bT_us
        bT_rollout2 = jax_jit_np(jax_vmap(ft.partial(rollout, state_cur_)))(bT_us)
        TbS_rollout2.append(bT_rollout2)

        # The first state of S_opttraj should be the state at the beginning.
        assert np.allclose(S_opttraj[0], T_x_new[-2])
        # The first action from S_u_opttraj should be the action that was taken.
        assert np.allclose(S_u_opttraj[0], action)

    T_x_new = np.stack(T_x_new, axis=0)
    assert T_x_new.shape == (T, 8)

    TS_x_opt_new = np.stack(TS_x_opt_new, axis=0)
    assert TS_x_opt_new.shape == (T - 1, controller.get_control_horizon(), 8)

    TS_u_opt_new = np.stack(TS_u_opt_new, axis=0)
    assert TS_u_opt_new.shape == (T - 1, controller.get_control_horizon() - 1, 2)

    T_mppistate_new = np.stack(T_mppistate_new, axis=0)
    TS_psafe = np.stack(TS_psafe, axis=0)
    TbS_rollout = np.stack(TbS_rollout, axis=0)
    TbS_rollout2 = np.stack(TbS_rollout2, axis=0)

    # ---------------------------------------------------------------------------
    # Reproducibility fun - see if we can reproduce T_x_new exactly.
    agent.reset_state(jnp.array(T_x[0, :-3]))
    controller.initial_control_sequence = jnp.array(T_mppistate_new[0])

    T_x_new2, TS_opttraj_new2 = [T_x[0, :-3]], []
    for kk in range(T - 1):
        # Run the controller, but record everything.
        state_next, cost, eval_time, action = agent.take_action_with_controller(return_time=True)
        T_x_new2.append(np.array(agent.get_state()))
        xS_opttraj = np.array(controller.T_x_opt)
        S_opttraj = xS_opttraj.T
        TS_opttraj_new2.append(S_opttraj)

    T_x_new2 = np.stack(T_x_new2, axis=0)
    assert T_x_new2.shape == (T, 8)

    TS_opttraj_new2 = np.stack(TS_opttraj_new2, axis=0)
    SS = controller.get_control_horizon()
    assert TS_opttraj_new2.shape == (T - 1, SS, 8)

    # ---------------------------------------------------------------------------
    # One more time, but with a clean mppi state.
    agent.reset_state(jnp.array(T_x[0, :-3]))
    controller.reset()

    T_x_cl, TS_opttraj_cl = [T_x[0, :-3]], []
    for kk in range(T - 1):
        # Run the controller, but record everything.
        state_next, cost, eval_time, action = agent.take_action_with_controller(return_time=True)
        T_x_cl.append(np.array(agent.get_state()))

        xS_opttraj = np.array(controller.T_x_opt)
        S_opttraj = xS_opttraj.T
        TS_opttraj_cl.append(S_opttraj)

    T_x_cl = np.stack(T_x_cl, axis=0)
    assert T_x_cl.shape == (T, 8)

    TS_opttraj_cl = np.stack(TS_opttraj_cl, axis=0)

    #############################################################################
    # Check Vh to determine where to start the debugging from.

    to_local = ft.partial(cost_evaluator.global_to_curvi, dynamics=agent.dynamics)
    T_xmap_new = jax_jit_np(jax_vmap(to_local))(T_x_new)

    get_h_vector_fn = ft.partial(get_h_vector, ncbf.h_cfg)
    Th_h = jax_jit_np(jax_vmap(get_h_vector_fn))(T_x_new[:, :8], T_xmap_new)

    Th_h_disc = np.array(get_max_mc(ncbf.alg_cfg.disc_gamma, Th_h, Th_h))

    get_Vh = ft.partial(ncbf.get_h, dynamics=agent.dynamics, ncbf_weights=ncbf.value_net.params)
    Th_Vh = jax_jit_np(jax_vmap(get_Vh))(T_x_new[:, :8], T_xmap_new)

    # Get last index where Vh is negative.
    idx_Vhneg = np.argmax(Th_Vh > 0) - 1
    logger.info("idx_Vhneg: {}".format(idx_Vhneg))

    idx_use = idx_Vhneg - 2

    TS_opttraj_new_use = TS_x_opt_new[idx_use:]
    TS_psafe_use = TS_psafe[idx_use:]
    TbS_rollout_use = TbS_rollout[idx_use:]
    TbS_rollout2_use = TbS_rollout2[idx_use:]

    TS_opttrajmap_new_use = jax_jit_np(jax_vmap(to_local, rep=2))(TS_opttraj_new_use)
    TS_Vh_opttraj_new_use = jax_jit_np(jax_vmap(get_Vh, rep=2))(TS_opttraj_new_use, TS_opttrajmap_new_use)

    #############################################################################
    # Plot the Vh and h vectors.
    # Also, for each of the predicted trajectories, plot the Vh.

    fig, axes = plt.subplots(3, sharex=True)
    [ax.axvline(idx_Vhneg, lw=0.4, color="C4") for ax in axes]
    ax = axes[0]

    ax.plot(Th_h[:, 0], lw=0.4, marker="o", color="0.2", ms=1, label="h")
    ax.plot(Th_h_disc[:, 0], lw=0.4, marker="o", color="C0", ms=1, alpha=0.8, label="h disc")
    ax.plot(Th_Vh[:, 0], lw=0.4, marker="o", color="C1", ms=1, alpha=0.8, label="Vh")
    line_labels(ax)
    ax.set_ylabel(r"$V^h$", ha="right", rotation=0)

    ax = axes[1]
    for kk, S_Vh_opttraj_new_use in enumerate(TS_Vh_opttraj_new_use[:5]):
        S_kk = idx_use + kk + np.arange(SS)
        ax.plot(S_kk, S_Vh_opttraj_new_use[:, 0], lw=0.4, marker="o", color=f"C{kk}", ms=1, alpha=0.8)
    ax.set_ylabel(r"$V^h$ pck", ha="right", rotation=0)

    ax = axes[2]
    for kk, S_psafe in enumerate(TS_psafe_use[:5]):
        S_kk = idx_use + kk + np.arange(SS - 1)
        ax.plot(S_kk, S_psafe, lw=0.4, marker="o", color=f"C{kk}", ms=1, alpha=0.8)
    ax.set_ylabel("p(safe)", ha="right", rotation=0)

    fig_path = plot_dir / "pred_new.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------
    # Why is the next state  different from the predicted next state....
    #    OH the sim has delta_t = 0.1, but the mppi controller has delta_t = 0.05.
    x_next_sim = jax2np(agent.dynamics.propagate(T_x_new[idx_use], TS_u_opt_new[idx_use, 0]))
    x_next_mppi = jax2np(controller.dynamics.propagate(T_x_new[idx_use], TS_u_opt_new[idx_use, 0]))
    x_next2_mppi = jax2np(controller.dynamics.propagate(x_next_mppi, TS_u_opt_new[idx_use, 0]))

    next_mppi_err = np.linalg.norm((x_next_mppi - x_next_sim) / np.abs(x_next_sim))
    next2_mppi_err = np.linalg.norm((x_next2_mppi - x_next_sim) / np.abs(x_next_sim))

    #############################################################################
    # Lets see how different the trajectory can be if we start from zero mppi state at idx_Vhneg.
    agent.reset_state(jnp.array(T_x_new[idx_use, :8]))
    controller.reset()

    T_new3 = 16

    T_x_new3, TS_opttraj_new3 = [T_x_new[idx_use, :8]], []
    for kk in range(T_new3 - 1):
        state_next, cost, eval_time, action = agent.take_action_with_controller(return_time=True)
        T_x_new3.append(np.array(agent.get_state()))

        xS_opttraj = np.array(controller.T_x_opt)
        S_opttraj = xS_opttraj.T
        TS_opttraj_new3.append(S_opttraj)

    T_x_new3 = np.stack(T_x_new3, axis=0)
    assert T_x_new3.shape == (T_new3, 8)

    TS_opttraj_new3 = np.stack(TS_opttraj_new3, axis=0)

    #############################################################################
    # Plot the scenario.
    track = get_ccrf_track()

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="datalim")

    traj_style = dict(lw=0.8, marker="o", ms=2, mec="none", alpha=0.8, zorder=5)

    ax.plot(T_x[:, S.X], T_x[:, S.Y], color="C1", mfc="C1", **traj_style)
    ax.plot(T_x[-1, S.X], T_x[-1, S.Y], marker="o", ms=3, color="C1", zorder=6)

    ax.plot(T_x_new[:, S.X], T_x_new[:, S.Y], color="C4", mfc="C4", **traj_style)
    ax.plot(T_x_new[idx_Vhneg, S.X], T_x_new[idx_Vhneg, S.Y], marker="^", ms=3, color="C4", alpha=0.7, zorder=8)
    ax.plot(T_x_new[idx_use, S.X], T_x_new[idx_use, S.Y], marker="P", ms=3, color="0.3", alpha=0.7, zorder=8)
    ax.plot(T_x_new[-1, S.X], T_x_new[-1, S.Y], marker="o", ms=3, color="C4", zorder=6)

    ax.plot(T_x_new2[:, S.X], T_x_new2[:, S.Y], color="C6", lw=2.0, alpha=0.5, zorder=4.5)

    ax.plot(T_x_new3[:, S.X], T_x_new3[:, S.Y], color="C6", mfc="C6", **traj_style)

    ax.plot(T_x_cl[:, S.X], T_x_cl[:, S.Y], color="C5", mfc="C5", **traj_style)
    ax.plot(T_x_cl[-1, S.X], T_x_cl[-1, S.Y], marker="o", ms=3, color="C5", zorder=6)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    plot_track(ax, track, 0.0, color="0.3", lw=0.8, alpha=0.2, zorder=3)
    plot_track(ax, track, 1.5, color="C0", lw=0.8, alpha=1.0, zorder=3)
    plot_track(ax, track, 2.0, color="C0", lw=0.8, alpha=1.0, zorder=3)

    ax.set(xlim=xlim, ylim=ylim)

    fig_path = plot_dir / "scenario.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    #############################################################################
    # Plot peacock on their own.
    peacock_style = dict(lw=1.2, alpha=0.8, zorder=6)

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="datalim")

    bT_segs = TS_x_opt_new[idx_use : idx_use + 5, :, [S.X, S.Y]]
    colors = [f"C{ii}" for ii in range(len(bT_segs))]
    line_col = LineCollection(bT_segs, colors=colors, **peacock_style)
    ax.add_collection(line_col)
    ax.autoscale_view()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # Also plot the rollouts.
    for kk, bS_rollout in enumerate(TbS_rollout_use[:1]):
        bS_segs = bS_rollout[:, :, [S.X, S.Y]]
        color = modify_hsv(f"C{kk}", s=-0.1, v=0.1)
        line_col = LineCollection(bS_segs, color=color, lw=0.5, alpha=0.3, zorder=5)
        ax.add_collection(line_col)

    for kk, bS_rollout in enumerate(TbS_rollout2_use[:1]):
        bS_segs = bS_rollout[:, :, [S.X, S.Y]]
        color = modify_hsv(f"C{kk + 1}", s=-0.1, v=0.1)
        line_col = LineCollection(bS_segs, color=color, lw=0.5, alpha=0.3, zorder=4)
        ax.add_collection(line_col)

    plot_track(ax, track, 0.0, color="0.5", lw=0.8, alpha=0.2, zorder=3)
    plot_track(ax, track, 1.5, color="0.4", lw=0.8, alpha=1.0, zorder=3)
    plot_track(ax, track, 2.0, color="0.2", lw=0.8, alpha=1.0, zorder=3)

    ax.set(xlim=xlim, ylim=ylim)

    fig_path = plot_dir / "peacock_new.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # Peacock plot.
    # bT_segs = TS_opttraj_new[:, :, [S.X, S.Y]]
    # line_col = LineCollection(bT_segs, color="C1", **peacock_style)
    # ax.add_collection(line_col)

    # bT_segs = TS_opttraj_cl[:, :, [S.X, S.Y]]
    # line_col = LineCollection(bT_segs, color="C5", **peacock_style)
    # ax.add_collection(line_col)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
