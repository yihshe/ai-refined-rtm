#!/bin/sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 -m pdb temp.py
# python3 -m pdb rtm_torch/Resources/PROSAIL/prospect.py
# python3 -m pdb rtm_unittest.py