#!/usr/bin/env bash

set -euo pipefail

# python -m src.main \
#     --modes simple weather comparaison \
#     --year 2026 \
#     --grand-prix "Hungarian" \
#     --session "Qualifying" \
#     --driver1 NOR \
#     --driver2 HAM \
#     --save-gap \
#     --list-drivers PIA VER RUS HAM ALO \
#     --session-bars \
#     --aero-analysis

python -m src.main \
    --modes simple \
    --year 2026 \
    --grand-prix 17 \
    --session "Qualifying" \
    --driver1 ANT \
    --driver2 VER \
    --plot-gap \
    # --save-gap \
    # --use_Q1_data
    # --aero-analysis
