import numpy as np
import pandas as pd

def show_gap_vuelta_by_driver(driver, laps_pitin, laps_pitout, df):
    # Mensaje inicial
    text = f"\nParadas {driver}"

    # Mostramos el tiempo antes y despues de la vuelta de entrada y de la vuelta de salida
    for lap_pitin, lap_pitout in zip(laps_pitin, laps_pitout):
        # Antes de la parada
        vuelta_anterior_parada = lap_pitin - 1.0

        delta_antes = df.loc[df['LapNumber'] == vuelta_anterior_parada, 'Delta'].iloc[0]
        text += "\n El Gap en la Vuelta {} era {:.3f}s".format(vuelta_anterior_parada, delta_antes)

        # Despues de la parad
        vuelta_despues_parada = lap_pitout

        delta_despues = df.loc[df['LapNumber'] == vuelta_despues_parada, 'Delta'].iloc[0]
        text += "\n El Gap en la Vuelta {} era {:.3f}s".format(vuelta_despues_parada, delta_despues)

    # Return Text
    text += "\n"
    return text


def check_dnf(laps_driver, total_laps):
    max_laps = max(laps_driver['LapNumber'])

    if max_laps < total_laps:
        # Ha abandonado
        return True
    else:
        return False


def check_if_driver_dnf(laps_driver, total_laps):
    # Evaluamos si el piloto abandona
    driver_dnf = check_dnf(laps_driver, total_laps)

    # Calculamos la ultima vuelta del piloto
    if driver_dnf:
        final_lap = max(laps_driver['LapNumber'])
    else:
        final_lap = total_laps

    return final_lap

