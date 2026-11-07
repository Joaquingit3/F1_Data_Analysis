#!/usr/bin/env bash

set -u

YEAR=2026
SESSION="Qualifying"

MODES=(
    "simple"
)

# GRAND_PRIX_LIST=(
#     "Australian"
#     "China"
#     "Japan"
#     "Miami"
#     "Canada"
#     "Monaco"
#     "Catalunya"
#     "Austrian"
#     "British"
#     "Belgian"
#     "Hungarian"
#     "Italian"
#     "Madring"
# )

GRAND_PRIX_LIST=(
    "Kuala Lumpur"
    "Singapur"
)

FAILED_RUNS=()

for GRAND_PRIX in "${GRAND_PRIX_LIST[@]}"; do
    echo
    echo "########################################################"
    echo "Procesando ${YEAR} ${GRAND_PRIX} - ${SESSION}"
    echo "########################################################"

    if python -m src.main \
        --modes "${MODES[@]}" \
        --year "$YEAR" \
        --grand-prix "$GRAND_PRIX" \
        --session "$SESSION" \
        --driver1 VER \
        --driver2 HAM \
        --save-gap \
        --use_Q1_data
    then
        echo "Ejecución completada: ${GRAND_PRIX}"
    else
        echo "Error procesando: ${GRAND_PRIX}" >&2
        FAILED_RUNS+=("$GRAND_PRIX")
    fi
done

# for GRAND_PRIX in "${GRAND_PRIX_LIST[@]}"; do
#     echo
#     echo "########################################################"
#     echo "Procesando ${YEAR} ${GRAND_PRIX} - ${SESSION}"
#     echo "########################################################"

#     if python -m src.main \
#         --modes "${MODES[@]}" \
#         --year "$YEAR" \
#         --grand-prix "$GRAND_PRIX" \
#         --session "$SESSION" \
#         --driver1 NOR \
#         --driver2 HAM \
#         --save-gap \
#         --list-drivers PIA VER RUS HAM ALO \
#         --session-bars \
#         --aero-analysis
#     then
#         echo "Ejecución completada: ${GRAND_PRIX}"
#     else
#         echo "Error procesando: ${GRAND_PRIX}" >&2
#         FAILED_RUNS+=("$GRAND_PRIX")
#     fi
# done

if (( ${#FAILED_RUNS[@]} > 0 )); then
    echo
    echo "Grandes Premios con errores:"
    printf ' - %s\n' "${FAILED_RUNS[@]}"
    exit 1
fi