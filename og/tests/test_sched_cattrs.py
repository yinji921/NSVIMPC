import ipdb
from attrs import define

from og.cfg_utils import Cfg
from og.schedules import Constant, LinDecay, Schedule


@define
class TestCfg(Cfg):
    sched: float | Schedule


def main():
    cfg1 = TestCfg(Constant(1.234))
    d = cfg1.asdict()
    print(d)
    cfg1_ = TestCfg.fromdict(d)
    print(cfg1_)
    print()

    cfg2 = TestCfg(LinDecay(3.0, 4.0, 5, 6))
    d = cfg2.asdict()
    print(d)
    cfg2_ = TestCfg.fromdict(d)
    print(cfg2_)
    print()

    cfg3 = TestCfg(5.678)
    d = cfg3.asdict()
    print(d)
    cfg3_ = TestCfg.fromdict(d)
    print(cfg3_)
    print()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
