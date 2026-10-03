import argparse

from .config import load_config
from .pipeline import run

p = argparse.ArgumentParser(description="Run the CATF prototype pipeline")
p.add_argument("--config", default=None)
p.add_argument("--input", default=None, help="CSV with timestamp,user,target,text (default: synthetic data)")
p.add_argument("--out", default=None)
p.add_argument("--seed", type=int, default=None)
a = p.parse_args()
over = {"data": {"synthetic": {"seed": a.seed}}, "forecast": {"seed": a.seed}} if a.seed is not None else None
run(load_config(a.config, over), a.input, a.out)
