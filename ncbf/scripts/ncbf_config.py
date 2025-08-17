from ncbf.ar_task import ConstrCfg, ObsCfg


def get_cfgs():
    n_kappa = 10
    obs_cfg = ObsCfg(kappa_dt=0.2, n_kappa=n_kappa, dt_exp=1.25)

    # h_cfg = ConstrCfg(track_width=1.5, track_width_term=1.999, margin_lo=0.2, margin_hi=1.0, h_term=2.2)
    h_cfg = ConstrCfg(track_width=1.5, track_width_term=2.0, margin_lo=0.3, margin_hi=0.2, h_term=2.6)
    # h_cfg = ConstrCfg(track_width=1.3, track_width_term=1.9, margin_lo=0.2, margin_hi=1.0, h_term=2.2)
    # h_cfg = ConstrCfg(track_width=1.2, track_width_term=1.9, margin_lo=0.2, margin_hi=0.7, h_term=2.3)
    # h_cfg = ConstrCfg(track_width=1.1, track_width_term=1.9, margin_lo=0.3, margin_hi=0.2, h_term=2.6)

    return obs_cfg, h_cfg


def get_h_cfg_for(width: float):
    if width == 1.1:
        return ConstrCfg(track_width=1.1, track_width_term=1.9, margin_lo=0.3, margin_hi=0.2, h_term=2.6)
    if width == 1.2:
        return ConstrCfg(track_width=1.2, track_width_term=1.9, margin_lo=0.2, margin_hi=0.7, h_term=2.3)
    if width == 1.3:
        return ConstrCfg(track_width=1.3, track_width_term=1.9, margin_lo=0.2, margin_hi=1.0, h_term=2.2)
    if width == 1.5:
        return ConstrCfg(track_width=1.5, track_width_term=1.999, margin_lo=0.2, margin_hi=1.0, h_term=2.2)

    raise ValueError("Invalid width {}".format(width))
