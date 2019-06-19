import yaml
from addict import Dict
with open('data.yml') as f:
    cfg = yaml.load(f)
    cfg = Dict(cfg)