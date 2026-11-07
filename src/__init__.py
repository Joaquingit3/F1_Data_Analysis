DICT_TEAM_COLOR = {
    "Ferrari": "#dc0000",
    "Renault": "cyan",
    "McLaren": "silver",
    "Mercedes": "#00d2be",
    "Red Bull": "#0600ef",
    "Red Bull Racing": "#0600ef",
    "Toro Rosso": "#469bff",
    "Benetton": "cyan"
}

# Modos Globales
MODES = (
    "simple",
    "weather",
    "comparaison",
    "comparaison-corners",
    "comparaison-sessions",
    "compare-circuits",
    "plot-circuits",
)

# Argumentos permitidos explícitamente por cada modo.
MODE_OPTIONS = {
    "simple": {
        "--driver1",
        "--driver2",
        "--update-stats",
        "--save-gap",
        "--plot-gap",
        "--minisectors",
        "--acc",
        "--rectas-drs",
        "--create-session",
        "--use_Q1_data",
    },
    "weather": set(),
    "comparaison": {
        "--list-drivers",
        "--distance",
        "--by-corners",
        "--list-distances",
        "--list-corners",
        "--session-bars",
        "--name",
        "--aero-analysis",
    },
    "comparaison-corners": {
        "--list-distances",
        "--session-bars",
    },
    "comparaison-sessions": set(),
    "compare-circuits": set(),
    "plot-circuits": {
        "--plot-circuits-pca",
        "--cluster",
    },
}


GLOBAL_OPTIONS = {
    "--modes",
    "--year",
    "--grand-prix",
    "--session",
    "--help",
}


# Modos que necesitan cargar una sesión concreta de FastF1.
SESSION_MODES = {
    "simple",
    "weather",
    "comparaison",
    "comparaison-corners",
}