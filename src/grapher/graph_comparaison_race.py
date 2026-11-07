# Library
import json
import os
import numpy as np
import pandas as pd
import fastf1 as ff1
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.text import TextPath
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as patches

from tabulate import tabulate

from fastf1 import plotting
from fastf1 import utils
from datetime import timedelta

from sklearn import linear_model

from src.utils.functions import format_time, calcula_lateralacc, calcula_longacc, paint_maxspeed, find_maxspeed_one_driver
from src.utils.functions_race import show_gap_vuelta_by_driver, check_dnf, check_if_driver_dnf

# CONSTANTS
ZOOM = 0.07
X = 0.27
Y = 0.01

# Funciones de calculo
def obtain_last_lap(event):
    # Calcular la ultima vuelta que da el race control
    df_racecontrol = event.race_control_messages
    mask_checkeredflag = df_racecontrol["Flag"] == "CHEQUERED"
    final_laps_control = df_racecontrol.loc[mask_checkeredflag, "Lap"].to_list()[0]

    # Obtener las vueltas totales por evento
    total_laps_event = event.total_laps

    # Comparar
    if final_laps_control < total_laps_event:
        # Tomamos las vueltas de control
        return final_laps_control
    elif final_laps_control == total_laps_event:
        # Tomamos las vueltas del evento
        return total_laps_event

def generate_correct_laptimes(laps_driver):
    # Checkear que no haya vueltas nat
    list_corrected_laps = []
    last_lap = laps_driver['LapNumber'].iloc[-1]
    
    if sum(pd.isna(laps_driver['LapTime'])) > 0:
        # Hay vueltas Na
        for i in range(laps_driver.shape[0]):
            lap = laps_driver['LapTime'].iloc[i]
            lap_number = laps_driver['LapNumber'].iloc[i]

            # TODO Ver si como tratar la primera vuelta que con este metodo no cuadra con Laptimes
            if pd.isna(lap):
                if last_lap == lap_number:
                    list_corrected_laps.append(pd.NaT)
                else:
                    # Vuelta NaT, corregir como la diferencia entre los starttime consecutivos i+1-i
                    start_time_i = laps_driver['LapStartTime'].iloc[i]
                    start_time_next = laps_driver['LapStartTime'].iloc[i+1]
                    real_time_lap = start_time_next - start_time_i

                    list_corrected_laps.append(real_time_lap)
            else:
                list_corrected_laps.append(lap)
    else:
        list_corrected_laps = laps_driver['LapTime'].to_list()
    return list_corrected_laps


def initialize_laptype(laps_driver, final_laps_race):
    n = laps_driver.shape[0]
    if n < final_laps_race:
        # Ha abandonado
        init_vector = ["Primera_Vuelta"] + ["Normal"] * (n - 2) + ["Abandono"]
    else:
        # NO ha abandonado
        init_vector = ["Primera_Vuelta"] + ["Normal"] * (n-1)

    return init_vector


def order_driver_by_sesion(event, drivers):
    # Calcular df sesion
    df_session_results = event.results[['Position', 'Abbreviation']]

    # Hacer seleccion - salen ordenado de mmayor posicion a menor
    mask_drivers = df_session_results['Abbreviation'].isin(drivers)
    filter_df = df_session_results[mask_drivers]
    list_ordered_drivers = filter_df['Abbreviation'].to_list()

    return list_ordered_drivers


def order_the_laps(laps_driver_1, laps_driver_2, drivers):
    # Loop de drivers
    driver_first = drivers[0]

    # Check driver in laps to compare
    driver_laps1 = laps_driver_1["Driver"].iloc[0]

    if driver_first == driver_laps1:
        laps_driver_first = laps_driver_1.copy()
        laps_driver_second = laps_driver_2.copy()

    else:
        laps_driver_first = laps_driver_2.copy()
        laps_driver_second = laps_driver_1.copy()

    return laps_driver_first, laps_driver_second

def get_list_teams_ordered_by_sesion(event, drivers):
    # Calcular df sesion
    df_session_results = event.results[['Position', 'Abbreviation', 'TeamName']]

    # Hacer seleccion - salen ordenado de mmayor posicion a menor
    mask_drivers = df_session_results['Abbreviation'].isin(drivers)
    filter_df = df_session_results[mask_drivers]
    list_ordered_teams = filter_df['TeamName'].to_list()

    return list_ordered_teams


def show_gap_vuelta(driver_1, driver, laps_pitin, laps_pitout, df):
    # Mensaje inicial
    text = f"\nParadas {driver}"

    if driver != driver_1:
        # Mostramos el tiempo antes y despues de la vuelta de entrada y de la vuelta de salida
        for lap_pitin, lap_pitout in zip(laps_pitin, laps_pitout):
            # Antes de la parada
            vuelta_anterior_parada = lap_pitin - 1.0

            delta_antes = df.loc[df['LapNumber'] == vuelta_anterior_parada, f'Delta_{driver}_{driver_1}'].iloc[0]
            text += "\n El Gap en la Vuelta {} era {:.3f}s entre {} y {}".format(vuelta_anterior_parada,
                                                                                 delta_antes,
                                                                                 driver_1,
                                                                                 driver)

            # Despues de la parada
            vuelta_despues_parada = lap_pitout

            delta_despues = df.loc[df['LapNumber'] == vuelta_despues_parada, f'Delta_{driver}_{driver_1}'].iloc[0]
            text += "\n El Gap en la Vuelta {} era {:.3f}s entre {} y {}".format(vuelta_despues_parada,
                                                                                 delta_despues,
                                                                                 driver_1,
                                                                                 driver)

    else:
        # Mostramos el tiempo antes y despues de la vuelta de entrada y de la vuelta de salida
        for lap_pitin, lap_pitout in zip(laps_pitin, laps_pitout):
            # Antes de la parada
            vuelta_anterior_parada = lap_pitin - 1.0

            # Comparativas
            columns = df.columns.tolist()
            delta_columns = [column for column in columns if column.startswith('Delta_')]

            for column in delta_columns:
                driver_compare = column.split("_")[1]

                delta_antes = df.loc[df['LapNumber'] == vuelta_anterior_parada, column].iloc[0]
                text += "\n El Gap en la Vuelta {} era {:.3f}s entre {} y {}".format(vuelta_anterior_parada,
                                                                                     delta_antes,
                                                                                     driver_1,
                                                                                     driver_compare)

                # Despues de la parad
                vuelta_despues_parada = lap_pitout

                delta_despues = df.loc[df['LapNumber'] == vuelta_despues_parada, column].iloc[0]
                text += "\n El Gap en la Vuelta {} era {:.3f}s entre {} y {}".format(vuelta_despues_parada,
                                                                                     delta_despues,
                                                                                     driver_1,
                                                                                     driver_compare)

    # Return Text
    text += "\n"
    return text

def generate_color_linestyle_apha(laps_driver_1, laps_driver_2, event):
    # Cargamos el color, estilo y alpha de los pilotos
    team_driver_1 = laps_driver_1['Team'].iloc[0]
    team_driver_2 = laps_driver_2['Team'].iloc[0]
    color_driver1 = ff1.plotting.get_team_color(team_driver_1, event)
    color_driver2 = ff1.plotting.get_team_color(team_driver_2, event)

    if team_driver_1 == team_driver_2:
        # Las vueltas son con el mismo equipo
        linestyle_driver_1 = '-'
        alpha_driver_1 = 1.0
        linestyle_driver_2 = '--'
        alpha_driver_2 = 0.5
    else:
        # Las vueltas son con de distinto equipo
        alpha_driver_1 = 1.0
        linestyle_driver_1 = '-'
        alpha_driver_2 = 1.0
        linestyle_driver_2 = '-'

    return color_driver1, color_driver2, linestyle_driver_1, linestyle_driver_2, alpha_driver_1, alpha_driver_2


def generate_color_linestyle_apha_v2(teams, drivers, event):
    # Lista de colores
    colors = [ff1.plotting.get_team_color(team, event) for team in teams]

    # Crear Dataframe
    d = {
        "Driver": drivers,
        "Team": teams
    }
    df = pd.DataFrame(d)
    list_unique_teams = np.unique(df["Team"].to_list())

    # Loop team unique
    alphas =[]
    linestyles = []
    for team in list_unique_teams:
        mask_team = df["Team"] == team

        if sum(mask_team) > 1:
            # Hay mas de uno
            n_teams = sum(mask_team)
            for i in range(n_teams):
                if i == 0:
                    alphas.append(1.0)
                    linestyles.append('-')
                else:
                    alphas.append(0.5)
                    linestyles.append('--')
        else:
            # Solo hay uno
            alphas.append(1.0)
            linestyles.append('-')

    return colors, linestyles, alphas


def build_df_deltas_future(laps_driver_1, laps_driver_2, drivers, event, final_lap_race):
    # Crear vector de vueltas y añadirlo al dict auxiliar
    lapnumbers = [i+1 for i in range(final_lap_race)]
    d = {'LapNumber': lapnumbers}

    # Loop de drivers
    driver_first = drivers[0]
    laps_driver_first = event.laps.pick_drivers(driver_first)['LapTime'].to_list()
    laps_driver_first = [laptime.total_seconds() for laptime in laps_driver_first]

    for i in range(1, len(drivers)):
        # Calculate laps of driver t compare
        driver_compare = drivers[i]
        laps_driver_compare = event.laps.pick_drivers(driver_compare)['LapTime'].to_list()
        laps_driver_compare = [laptime.total_seconds() for laptime in laps_driver_compare]

        # Calculate diff laptimes
        diff_laptimes = [float(t1 - t2) for t1, t2 in zip(laps_driver_compare, laps_driver_first)]

        # Añadir a dict
        d[f"DiffLapTime_{driver_compare}_{driver_first}"] = diff_laptimes

        # Calcular delta
        d[f"Delta{driver_compare}_{driver_first}"] = d[f"DiffLapTime_{driver_compare}_{driver_first}"].cumsum()

    # Convertir d a df
    df = pd.DataFrame(d)

    return df


def get_min_lap_by_two(final_lap_1, final_lap_2):
    if final_lap_1 <= final_lap_2:
        return final_lap_1
    else:
        return final_lap_2
    

def get_final_lap(final_lap_d1, final_lap_d2, final_lap_race):
    # Get final lap of the drivers
    final_lap_drivers = get_min_lap_by_two(final_lap_d1, final_lap_d2)

    # Compare final_lap_drivers with final_lap_race
    final_lap_compairaison = get_min_lap_by_two(final_lap_drivers, final_lap_race)
    return int(final_lap_compairaison)


def build_df_deltas(laps_driver_1, laps_driver_2, drivers, event, final_lap_race):
    # Crear vector de vueltas y añadirlo al dict auxiliar
    # check si el algo de los dos pilotos está doblado
    final_lap_d1 = laps_driver_1['LapNumber'].iloc[-1]
    final_lap_d2 = laps_driver_2['LapNumber'].iloc[-1]

    final_lap_compairaison = get_final_lap(final_lap_d1, final_lap_d2, final_lap_race)
    lapnumbers = [i + 1 for i in range(final_lap_compairaison)]
    d = {'LapNumber': lapnumbers}

    # Loop de drivers
    driver_first = drivers[0]
    driver_second = drivers[1]

    # Calcular vueltas driver first and second
    laps_driver_first = laps_driver_1['LapTime'].to_list()
    laps_driver_first = [laptime.total_seconds() for laptime in laps_driver_first]

    laps_driver_second = laps_driver_2['LapTime'].to_list()
    laps_driver_second = [laptime.total_seconds() for laptime in laps_driver_second]

    # Calculate diff laptimes
    diff_laptimes = [float(t1 - t2) for t1, t2 in zip(laps_driver_second, laps_driver_first)]

    # Añadir a dict
    d[f"DiffLapTime_{driver_second}_{driver_first}"] = diff_laptimes

    # Convertir d a df
    df = pd.DataFrame(d)

    '''
    # Checkear y Resetear Delta si hay bandera roja
    df_race_control = event.race_control_messages
    mask_redflags = df_race_control["Flag"] == "RED"

    if sum(mask_redflags) > 0:
        # Hay banderas rojas
        laps_redflag = df_race_control.loc[mask_redflags, "Lap"].to_list()

        # Calcular deltas antes de la bandera
        index_ini = 0
        deltas = []
        for lap in laps_redflag:
            # Calcular delta antes de la bandera roja
            index_lap = lap - 1
            delta_before_redflag = sum(df[f"DiffLapTime_{driver_second}_{driver_first}"][index_ini:index_lap])
            deltas.append(delta_before_redflag)

            # Actualizar index_ini para la siguiente bandera
            index_ini = lap

        # Restar en la vuelta posterior a las banderas rojas la delta en DiffLapTime
        for i, lap in enumerate(laps_redflag):
            index = lap
            df.loc[index, f"DiffLapTime_{driver_second}_{driver_first}"] -= deltas[i]
    '''

    # Calcular delta
    df[f"Delta_{driver_second}_{driver_first}"] = df[f"DiffLapTime_{driver_second}_{driver_first}"].cumsum()

    return df


def genera_extremos_tiempo(laps_driver_1, laps_driver_2):
    # Mascara de tiemp Normal
    mask_normal_driver_1 = laps_driver_1['LapType'] == 'Normal'
    mask_normal_driver_2 = laps_driver_2['LapType'] == 'Normal'

    time_max_d1 = max(laps_driver_1.loc[mask_normal_driver_1, 'LapTime'].to_list())
    time_max_d2 = max(laps_driver_2.loc[mask_normal_driver_2, 'LapTime'].to_list())
    time_min_d1 = min(laps_driver_1.loc[mask_normal_driver_1, 'LapTime'].to_list())
    time_min_d2 = min(laps_driver_2.loc[mask_normal_driver_2, 'LapTime'].to_list())

    if time_max_d1 >= time_max_d2:
        time_max = time_max_d1
    else:
        time_max = time_max_d2

    if time_min_d1 <= time_min_d2:
        time_min = time_min_d1
    else:
        time_min = time_min_d2

    return time_min, time_max

def genera_extremos_delta(df):
    # Extraer columnas delta
    columns = df.columns.tolist()
    delta_columns = [column for column in columns if column.startswith('Delta_')]

    # Loop deltas
    for i, column in enumerate(delta_columns):
        if i == 0:
            min_delta = min(df[column].to_list())
            max_delta = max(df[column].to_list())
        else:
            min_aux = min(df[column].to_list())
            max_aux = max(df[column].to_list())

            if min_aux < min_delta:
                min_delta = min_aux

            if max_aux > max_delta:
                max_delta = max_aux

    # Check a take the higher absolute value of min and max delta
    min_delta_absoute = abs(min_delta)
    if min_delta_absoute > max_delta:
        delta = min_delta_absoute
    else:
        delta = max_delta

    extreme_min = -delta - 2.0
    extreme_max = delta + 2.0

    return extreme_min, extreme_max


def extract_real_pitin_pitout_laps(laps_driver_1, event):
    # Create delta tyre life column
    driver = laps_driver_1['Driver'].iloc[0]
    results = event.results
    final_status = results.loc[results['Abbreviation']==driver, 'Status'].iloc[0]
    laps_driver_1['delta_tyrelife'] = laps_driver_1['TyreLife'].diff()

    # PitOuttimes
    mask_pitoutimes_notNA = laps_driver_1['PitOutTime'].notna()
    mask_delta_tyrelife_neg = laps_driver_1['delta_tyrelife'] < 0.0
    laps_pitout_driver_1 = laps_driver_1.loc[mask_pitoutimes_notNA & mask_delta_tyrelife_neg, 'LapNumber'].to_list()

    # Check si entro
    if len(laps_pitout_driver_1) > 0:
        # PitIntimes
        laps_pitin_driver_1 = [lap - 1.0 for lap in laps_pitout_driver_1]

        # Check abandom case
        total_laps_pitin_driver_1 = laps_driver_1.loc[laps_driver_1['PitInTime'].notna(), 'LapNumber'].to_list() 
    
        if (total_laps_pitin_driver_1[-1] > laps_pitin_driver_1[-1]) & (final_status != "Finished"):
            # Entro a boxes una vez más por abandono
            laps_pitin_driver_1.append(total_laps_pitin_driver_1[-1])
    else:
        laps_pitin_driver_1 = []
    return laps_pitin_driver_1, laps_pitout_driver_1


def extract_pitin_pitout_times(laps_driver, pitin_laps, pitout_laps):
    # Time pitin
    times_pitin = []
    for pitin_lap in pitin_laps:
        times_pitin.append(laps_driver.loc[laps_driver['LapNumber'] == pitin_lap, 'PitInTime'].iloc[0])

    # Time pitout
    times_pitout = []
    for pitout_lap in pitout_laps:
        times_pitout.append(laps_driver.loc[laps_driver['LapNumber'] == pitout_lap, 'PitOutTime'].iloc[0])

    return times_pitin, times_pitout


def create_real_stint_column(laps_driver_1):
    # Create delta tyre life column
    laps_driver_1['delta_tyrelife'] = laps_driver_1['TyreLife'].diff()

    # stint frontair laps
    laps_frontair_stint = [1.0]
    laps_frontair_stint.extend(laps_driver_1.loc[laps_driver_1['delta_tyrelife'] < 0.0, 'LapNumber'].to_list())
    laps_frontair_stint.append(laps_driver_1['LapNumber'].iloc[-1])

    # Create stint pairs
    stint_pairs = []
    for i in range(len(laps_frontair_stint) - 1):
        stint_pairs.append((laps_frontair_stint[i], laps_frontair_stint[i+1]))

    # Create column 
    for i, (ini, fin) in enumerate(stint_pairs):
        if i == 0:
            aux = [i+1] * int(fin - ini)
            real_stint = aux
        elif i == (len(stint_pairs) - 1):
            aux = [i+1] * int(fin - ini + 1)
            real_stint.extend(aux)
        else:
            aux = [i+1] * int(fin - ini)
            real_stint.extend(aux)

    laps_driver_1['RealStint'] = real_stint


def pitstop_function(laps_driver_1, time_max, color_driver1, driver_1, event, ax1, hatch=None):
    # Buscar vueltas de entrada y salida de los pits (Only for pitstop or abandom)
    laps_pitin_driver_1, laps_pitout_driver_1 = extract_real_pitin_pitout_laps(laps_driver_1, event)

    # Define alpha
    use_alpha = 0.7 if hatch is None else 0.3

    # Revisar si hay mas tiempos de entrada que de salida
    if len(laps_pitout_driver_1) < len(laps_pitin_driver_1):
        # Ha habido un abandono posiblemente => Delimitamos con el vector más pequeño
        for i in range(len(laps_pitin_driver_1)):
            if i != (len(laps_pitin_driver_1) -1):
                # Pintar franja de parada
                mask_lap1 = laps_driver_1['LapNumber'] >= laps_pitin_driver_1[i]
                mask_lap2 = laps_driver_1['LapNumber'] <= laps_pitout_driver_1[i]
                subset = laps_driver_1.loc[mask_lap1 & mask_lap2,]
                if hatch is not None:
                    ax1.fill_between(
                        subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                        timedelta(days=0, hours=0, minutes=0, seconds=0, milliseconds=0), # Limite inferior del área sombreada en el eje x
                        time_max,  # Limite superior del área sombreada en el eje x
                        color=color_driver1, 
                        alpha=use_alpha, 
                        label=f'{driver_1} PitStop {i + 1}',
                        hatch=hatch
                    )
                else:
                   ax1.fill_between(
                        subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                        timedelta(days=0, hours=0, minutes=0, seconds=0, milliseconds=0), # Limite inferior del área sombreada en el eje x
                        time_max,  # Limite superior del área sombreada en el eje x
                        color=color_driver1, 
                        alpha=use_alpha, 
                        label=f'{driver_1} PitStop {i + 1}'
                    )

                # Añado las vueltas a tipo PitStop
                laps_driver_1.loc[laps_driver_1['LapNumber'].between(laps_pitin_driver_1[i],
                                                                     laps_pitout_driver_1[i]), "LapType"] = "PitStop"
            else:
                # Añado la ultima parada de abandono
                # Pintar franja de parada
                lap_final = max(laps_driver_1['LapNumber'])
                mask_lap1 = laps_driver_1['LapNumber'] >= laps_pitin_driver_1[i]
                mask_lap2 = laps_driver_1['LapNumber'] <= lap_final
                subset = laps_driver_1.loc[mask_lap1 & mask_lap2,]

                if hatch is not None:
                    ax1.fill_between(
                        subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                        timedelta(days=0, hours=0, minutes=0, seconds=0, milliseconds=0), # Limite inferior del área sombreada en el eje x
                        time_max,  # Limite superior del área sombreada en el eje x
                        color="yellow", 
                        alpha=use_alpha, 
                        label=f'{driver_1} Abandono',
                        hatch=hatch
                    )
                else:
                    ax1.fill_between(
                        subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                        timedelta(days=0, hours=0, minutes=0, seconds=0, milliseconds=0), # Limite inferior del área sombreada en el eje x
                        time_max,  # Limite superior del área sombreada en el eje x
                        color="yellow", 
                        alpha=use_alpha, 
                        label=f'{driver_1} Abandono'
                    )

                # Añado las vueltas a tipo PitStop
                laps_driver_1.loc[laps_driver_1['LapNumber'].between(laps_pitin_driver_1[i],
                                                                     lap_final), "LapType"] = "Abandono"

    elif len(laps_pitout_driver_1) == len(laps_pitin_driver_1):
        # Solo han sido paradas
        for i in range(len(laps_pitin_driver_1)):
            # Pintar franja de parada
            mask_lap1 = laps_driver_1['LapNumber'] >= laps_pitin_driver_1[i]
            mask_lap2 = laps_driver_1['LapNumber'] <= laps_pitout_driver_1[i]
            subset = laps_driver_1.loc[mask_lap1 & mask_lap2,]

            if hatch is not None:
                ax1.fill_between(
                    subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                    timedelta(days=0, hours=0, minutes=0, seconds=0, milliseconds=0), # Limite inferior del área sombreada en el eje x
                    time_max, # Limite superior del área sombreada en el eje x
                    color=color_driver1, 
                    alpha=use_alpha, 
                    label=f'{driver_1} PitStop {i + 1}',
                    hatch=hatch
                )
            else:
                ax1.fill_between(
                    subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                    timedelta(days=0, hours=0, minutes=0, seconds=0, milliseconds=0), # Limite inferior del área sombreada en el eje x
                    time_max, # Limite superior del área sombreada en el eje x
                    color=color_driver1, 
                    alpha=use_alpha, 
                    label=f'{driver_1} PitStop {i + 1}'
                )

            # Añado las vueltas a tipo PitStop
            laps_driver_1.loc[laps_driver_1['LapNumber'].between(laps_pitin_driver_1[i],
                                                                 laps_pitout_driver_1[i]), "LapType"] = "PitStop"
    else:
        print("Ha habido mas vueltas de salida que de entrada al box :(")


def pitstop_function_deltas(laps_driver, time_max, color_driver, driver_1, driver, text, df, event, ax1, hatch=None):
    # Buscar vueltas de entrada y salida de los pits
    laps_pitin_driver, laps_pitout_driver = extract_real_pitin_pitout_laps(laps_driver, event)
    use_alpha = 0.7 if hatch is None else 0.3

    # Revisar si hay mas tiempos de entrada que de salida
    if len(laps_pitout_driver) < len(laps_pitin_driver):
        # Ha habido un abandono posiblemente => Delimitamos con el vector más pequeño
        for i in range(len(laps_pitin_driver)):
            if i != (len(laps_pitin_driver) -1):
                # Pintar franja de parada
                mask_lap1 = laps_driver['LapNumber'] >= laps_pitin_driver[i]
                mask_lap2 = laps_driver['LapNumber'] <= laps_pitout_driver[i]
                subset = laps_driver.loc[mask_lap1 & mask_lap2,]

                # Set hatch
                if hatch is not None:
                    ax1.fill_between(
                        subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                        -time_max,
                        time_max,
                        color=color_driver, 
                        alpha=use_alpha, 
                        label=f'{driver} PitStop {i + 1}',
                        hatch=hatch
                    )
                else:
                    ax1.fill_between(
                        subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                        -time_max,
                        time_max,
                        color=color_driver, 
                        alpha=use_alpha, 
                        label=f'{driver} PitStop {i + 1}'
                    )

                # Añado las vueltas a tipo PitStop
                laps_driver.loc[laps_driver['LapNumber'].between(laps_pitin_driver[i],
                                                                 laps_pitout_driver[i]), "LapType"] = "PitStop"
            else:
                # Añado la ultima parada de abandono
                # Pintar franja de parada
                lap_final = max(laps_driver['LapNumber'])
                mask_lap1 = laps_driver['LapNumber'] >= laps_pitin_driver[i]
                mask_lap2 = laps_driver['LapNumber'] <= max(lap_final)
                subset = laps_driver.loc[mask_lap1 & mask_lap2,]

                # Set hatch
                if hatch is not None:
                    ax1.fill_between(
                        subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                        -time_max,
                        time_max,
                        color=color_driver, 
                        alpha=use_alpha, 
                        label=f'{driver} PitStop {i + 1}',
                        hatch=hatch
                    )
                else:
                    ax1.fill_between(
                        subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                        -time_max,
                        time_max,
                        color=color_driver, 
                        alpha=use_alpha, 
                        label=f'{driver} PitStop {i + 1}'
                    )

                # Añado las vueltas a tipo PitStop
                laps_driver.loc[laps_driver['LapNumber'].between(laps_pitin_driver[i],
                                                                 lap_final), "LapType"] = "Abandono"
            ## Poner el Gap del piloto second antes de la parada del piloto 1
            text += show_gap_vuelta(driver_1, driver, laps_pitin_driver, laps_pitout_driver, df)

    elif len(laps_pitout_driver) == len(laps_pitin_driver):
        # Solo han sido paradas
        for i in range(len(laps_pitin_driver)):
            # Pintar franja de parada
            mask_lap1 = laps_driver['LapNumber'] >= laps_pitin_driver[i]
            mask_lap2 = laps_driver['LapNumber'] <= laps_pitout_driver[i]
            subset = laps_driver.loc[mask_lap1 & mask_lap2,]

            # Set hatch
            if hatch is not None:
                ax1.fill_between(
                    subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                    -time_max,
                    time_max,
                    color=color_driver, 
                    alpha=use_alpha, 
                    label=f'{driver} PitStop {i + 1}',
                    hatch=hatch
                )
            else:
                ax1.fill_between(
                    subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                    -time_max,
                    time_max,
                    color=color_driver, 
                    alpha=use_alpha, 
                    label=f'{driver} PitStop {i + 1}'
                )

            # Añado las vueltas a tipo PitStop
            laps_driver.loc[laps_driver['LapNumber'].between(laps_pitin_driver[i],
                                                             laps_pitout_driver[i]), "LapType"] = "PitStop"
        ## Poner el Gap del piloto second antes de la parada del piloto 1
        text += show_gap_vuelta(driver_1, driver, laps_pitin_driver, laps_pitout_driver, df)
    else:
        print("Ha habido mas vueltas de salida que de entrada al box :(")
        text += "Ha habido mas vueltas de salida que de entrada al box :("

    return text


def calcula_pitstopstimes(laps_driver, driver, event):
    # Calcular vectores de hora de entrada y salida
    laps_pitin_driver_first, laps_pitout_driver_first = extract_real_pitin_pitout_laps(laps_driver, event)
    times_pitin_driver_first, times_pitout_driver_first = extract_pitin_pitout_times(laps_driver, laps_pitin_driver_first, laps_pitout_driver_first)

    #times_pitin_driver_first = laps_driver.loc[laps_driver['PitInTime'].notna(), 'PitInTime'].to_list()
    #times_pitout_driver_first = laps_driver.loc[laps_driver['PitOutTime'].notna(), 'PitOutTime'].to_list()

    # Determinar el tiempo de pitstop
    text = f"\nTiempos de las Paradas de {driver}\n"

    for i in range(len(times_pitin_driver_first)):
        aux = times_pitout_driver_first[i] - times_pitin_driver_first[i]
        text += f" Parada {i+1} fue de {aux.total_seconds()}s\n"

    text += "\n"
    return text


def manage_vsc(df_race_control, mask, race_stats):
    # Estructura {"tipo": "VirtualSafetyCar", "laps": [28, 28]}
    # Crear un subset con mascara + mas cara redflag y checredflag
    subset = df_race_control.loc[mask,].copy()

    # Obtener lista de despliegue de SC y de In this lap
    mask_vsafetycardeployed = subset['Message'] == 'VIRTUAL SAFETY CAR DEPLOYED'
    mask_vsafetycarending = subset['Message'] == 'VIRTUAL SAFETY CAR ENDING'

    laps_vscdeployed = subset.loc[mask_vsafetycardeployed, 'Lap'].to_list()
    laps_vscending = subset.loc[mask_vsafetycarending, 'Lap'].to_list()

    # Ver si hay el mismo numero de cada uno o no
    if len(laps_vscdeployed) == len(laps_vscending):
        ## Si lo hay
        ## -> Añadir a race_stats los pares deploy-inthislap
        for lap_deployed, lap_ending in zip(laps_vscdeployed, laps_vscending):
            race_stats.append({"tipo": "VirtualSafetyCar", "laps": [lap_deployed, lap_ending]})

    elif len(laps_vscdeployed) > len(laps_vscending):
        ## No lo hay
        ## -> Emparezar el caso de deployed sin in this lap cn red flag o con chequered flag
        ## Recorrer el df SC junto bandera roja o a cuadros si es necesario
        mask_redflags = df_race_control["Flag"] == "RED"
        mask_checkered = df_race_control["Flag"] == "CHEQUERED"
        mask_scdeployed = df_race_control["Message"] == "SAFETY CAR DEPLOYED"

        df = df_race_control.loc[mask_checkered | mask_redflags | mask | mask_scdeployed,]
        df["NewStatus"] = [status if status != "" else df["Message"].iloc[i] for i, status in enumerate(df["Status"])]
        mask_sc = df_race_control["Message"].str.startswith("SAFETY")
        df.loc[mask_sc, "NewStatus"] = "SC DEPLOYED"

        vsc_ranges = []

        # Variables para rastrear el estado actual
        current_vsc_deployed = None

        # Recorrer el DataFrame fila por fila
        for _, row in df.iterrows():
            status = row['Status']
            lap = row['Lap']

            # Si encontramos 'VSC DEPLOYED' y no hay un rango activo, iniciamos un nuevo rango
            if status == 'DEPLOYED' and current_vsc_deployed is None:
                current_vsc_deployed = lap

            # Si encontramos 'VSC ENDING' y hay un rango activo, lo cerramos
            elif status == 'ENDING' and current_vsc_deployed is not None:
                vsc_ranges.append((current_vsc_deployed, lap))
                current_vsc_deployed = None

            # Si encontramos un evento que cierra un VSC desparejado
            elif status in ['RED FLAG', 'CHECKERED FLAG', 'SC DEPLOYED'] and current_vsc_deployed is not None:
                vsc_ranges.append((current_vsc_deployed, lap))
                current_vsc_deployed = None

        # Rellenar con sc_ranges en race_control
        for i in range(len(vsc_ranges)):
            lap_deployed = vsc_ranges[0]
            lap_inthislap = vsc_ranges[1]
            race_stats.append({"tipo": "VirtualSafetyCar", "laps": [lap_deployed, lap_inthislap]})

    return race_stats


def manage_sc(df_race_control, mask, race_stats):
    # Estructura {"tipo": "SafetyCar", "laps": [28, 28]}
    # Crear un subset con mascara + mas cara redflag y checredflag
    subset = df_race_control.loc[mask,].copy()

    # Obtener lista de despliegue de SC y de In this lap
    mask_safetycardeployed = subset['Message'] == 'SAFETY CAR DEPLOYED'
    mask_safetycarinthislap = subset['Message'] == 'SAFETY CAR IN THIS LAP'

    # Check flux normal SC
    if sum(mask_safetycardeployed) > 0:
        # Normal SC flux 
        laps_scdeployed = subset.loc[mask_safetycardeployed, 'Lap'].to_list()
        laps_scinthislap = subset.loc[mask_safetycarinthislap, 'Lap'].to_list()

        # Ver si hay el mismo numero de cada uno o no
        if len(laps_scdeployed) == len(laps_scinthislap):
            ## Si lo hay
            ## -> Añadir a race_stats los pares deploy-inthislap
            for lap_deployed, lap_inthislap in zip(laps_scdeployed, laps_scinthislap):
                race_stats.append({"tipo": "SafetyCar", "laps": [lap_deployed, lap_inthislap]})

        elif len(laps_scdeployed) > len(laps_scinthislap):
            ## No lo hay
            ## -> Emparezar el caso de deployed sin in this lap cn red flag o con chequered flag
            ## Recorrer el df SC junto bandera roja o a cuadros si es necesario
            mask_redflags = df_race_control["Flag"] == "RED"
            mask_checkered = df_race_control["Flag"] == "CHEQUERED"
            df = df_race_control.loc[mask_checkered | mask_redflags | mask,]
            df["Status"] = [status if status is not None else df["Message"].iloc[i] for i, status in enumerate(df["Status"])]

            sc_ranges = []
            unmatched_deployed = []
            current_deployed = None # variable del estado actual

            # Recorrer el DataFrame fila por fila
            for _, row in df.iterrows():
                status = row['Status']
                lap = row['Lap']

                # Si encontramos un 'DEPLOYED' y no hay un 'current_deployed' activo, empezamos un nuevo rango
                if status == 'DEPLOYED' and current_deployed is None:
                    current_deployed = lap

                # Si encontramos un 'IN THIS LAP' y hay un 'current_deployed' activo, cerramos el rango
                elif status == 'IN THIS LAP' and current_deployed is not None:
                    sc_ranges.append([current_deployed, lap])
                    current_deployed = None

                # Si encontramos otro 'DEPLOYED' mientras ya hay un 'current_deployed', es un caso consecutivo sin cerrar
                elif status == 'DEPLOYED' and current_deployed is not None:
                    unmatched_deployed.append(current_deployed)
                    current_deployed = lap

                # Si encontramos una bandera roja o final de carrera, cerramos cualquier rango abierto
                elif status in ['RED FLAG', 'CHEQUERED FLAG'] and current_deployed is not None:
                    sc_ranges.append([current_deployed, lap])
                    current_deployed = None

            # Rellenar con sc_ranges en race_control
            for i in range(len(sc_ranges)):
                laps = sc_ranges[i]
                lap_deployed = laps[0]
                lap_inthislap = laps[1]
                race_stats.append({"tipo": "SafetyCar", "laps": [lap_deployed, lap_inthislap]})
    else:
        # SC at the begining
        mask_race_start_behind_sc = df_race_control["Message"] == "RACE WILL START BEHIND THE SAFETY CAR"
        mask_rolling_start = df_race_control["Message"] == "ROLLING START"

        # Laps
        print(subset)
        laps_scdeployed = df_race_control.loc[mask_race_start_behind_sc, 'Lap'].to_list()[0]
        laps_scinthislap = df_race_control.loc[mask_rolling_start, 'Lap'].to_list()[0]

        # Append race_stats
        race_stats.append({"tipo": "SafetyCar", "laps": [laps_scdeployed, laps_scinthislap]})

    return race_stats

def manage_redflag(df_race_control, mask, race_stats):
    # Estructura {"tipo": RedFlad", "laps": [28, 32]}
    # Check mask
    if sum(mask) > 0:
        # Hay bandera ROJA
        # Crear subset red
        lap_red = df_race_control.loc[mask, "Lap"].iloc[0]

        # Save
        race_stats.append({"tipo": "RedFlag", "laps": [lap_red, lap_red]})

    return race_stats


# Manejar SafetyCars y Virtual SafetyCar ademas de que puede que un SC o VSC no acabe y halla dos DEPLOYED por bandera roja
def safety_control_function(laps_driver_1, laps_driver_2, time_max, event, ax1):
    # COnstants and variables
    dict_control = {
        "SafetyCar": {
            "tipo_eventos": "SAFETY CAR",
            "color": "gray",
            "line_color": "white"
        },
        "VirtualSafetyCar": {
            "tipo_eventos": "VIRTUAL SAFETY CAR",
            "color": "#d3d3a1",
            "line_color": "goldenrod"
        },
        "RedFlag": {
            "tipo_eventos": "VIRTUAL SAFETY CAR",
            "color": "red",
            "line_color": "red"
        }
    }

    # Crear Dataframe de eventos que llevan a parar la carrera
    df_race_control = event.race_control_messages
    mask_safetycar = df_race_control["Category"] == "SafetyCar"
    mask_redflags = df_race_control["Flag"] == "RED"
    mask_checkered = df_race_control["Flag"] == "CHEQUERED"

    mask_race_start_behind_sc = df_race_control["Message"] == "RACE WILL START BEHIND THE SAFETY CAR"
    mask_rolling_start = df_race_control["Message"] == "ROLLING START"
    df_race_control = df_race_control.loc[mask_safetycar | mask_redflags | mask_checkered | mask_race_start_behind_sc | mask_rolling_start,]

    # Recorrer eventos por ver si estan en df_race_control
    type_events = ["VirtualSafetyCar", "SafetyCar", "RedFlag"]
    race_stats = []
    for type in type_events:
        if type == "VirtualSafetyCar":
            mask_vsc = df_race_control["Message"].str.startswith("VIRTUAL")
            if sum(mask_vsc) > 0:
                # Hay VSC
                race_stats = manage_vsc(df_race_control, mask_vsc, race_stats)
        elif type == "SafetyCar":
            mask_sc = df_race_control["Message"].str.contains("SAFETY")
            mask_vsc = df_race_control["Message"].str.startswith("VIRTUAL")
            mask_joined = mask_sc & ~mask_vsc
            if sum(mask_joined) > 0:
                # Hay SC
                race_stats = manage_sc(df_race_control, mask_sc, race_stats)
        elif type == "RedFlag":
            mask_redflag = df_race_control["Flag"] == "RED"
            if sum(mask_redflag) > 0:
                # Hay RedFlag
                race_stats = manage_redflag(df_race_control, mask_redflag, race_stats)

    # Resultado
    '''Race_stats = [
       {"tipo": "VirtualSafetyCar", "laps": [28, 28]},
       {"tipo": "SafetyCar", "laps": [30, 32]},
       {"tipo": "RedFlag", "laps": [32, 32]},
       {"tipo": "SafetyCar", "laps": [39, 42]}
    ]'''

    # Pasar Race stats a df
    df_race_stats = pd.DataFrame(race_stats)

    # Recorrer Race Stats
    cont_sc = 0
    cont_vsc = 0
    cont_red = 0
    for i in range(df_race_stats.shape[0]):
        tipo = df_race_stats["tipo"].iloc[i]
        mask_tipo = df_race_stats["tipo"] == tipo

        color = dict_control[tipo]["color"]
        line_color = dict_control[tipo]["line_color"]

        laps = df_race_stats["laps"].iloc[i]
        lap_ini = laps[0]
        lap_fin = laps[1]

        # Asignar cnt depende del tipo
        if tipo == "VirtualSafetyCar":
            cont = cont_vsc
        elif tipo == "SafetyCar":
            cont = cont_sc
        elif tipo == "RedFlag":
            cont = cont_red

        if sum(mask_tipo) > 1:
            if tipo != "RedFlag":
                # Hay mas de uno
                if lap_ini == lap_fin:
                    # Plot a vertical line in the extremes
                    ax1.axvline(x=lap_ini, linestyle='--', linewidth=1.0, color=line_color, label=tipo)

                    # Añado las vueltas como type
                    laps_driver_1.loc[laps_driver_1['LapNumber'] == lap_ini, "LapType"] = tipo
                    laps_driver_2.loc[laps_driver_2['LapNumber'] == lap_ini, "LapType"] = tipo

                    # Actualizar cont y añado texto
                    if tipo == "VirtualSafetyCar":
                        cont_vsc += 1
                    elif tipo == "SafetyCar":
                        cont_sc += 1
                else:
                    mask_lap1 = laps_driver_1['LapNumber'] >= lap_ini
                    mask_lap2 = laps_driver_1['LapNumber'] <= lap_fin

                    subset = laps_driver_1.loc[mask_lap1 & mask_lap2,]
                    ax1.fill_between(
                        subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                        timedelta(days=0, hours=0, minutes=0, seconds=0, milliseconds=0),  # Limite inferior
                        time_max,  # Limite superior del área sombreada en el eje x
                        color=color,
                        alpha=0.3,
                        label=f'{tipo} {cont + 1}'
                    )

                    # Plot a vertical line in the extremes
                    ax1.axvline(x=lap_ini, linestyle='--', linewidth=1.0, color=line_color)
                    ax1.axvline(x=lap_fin, linestyle='--', linewidth=1.0, color=line_color)

                    # Añado las vueltas como type
                    laps_driver_1.loc[laps_driver_1['LapNumber'].between(lap_ini, lap_fin), "LapType"] = tipo
                    laps_driver_2.loc[laps_driver_2['LapNumber'].between(lap_ini, lap_fin), "LapType"] = tipo

                    # Actualizar cont
                    if tipo == "VirtualSafetyCar":
                        cont_vsc += 1
                    elif tipo == "SafetyCar":
                        cont_sc += 1

            else:
                # Plot a vertical line in the extremes
                ax1.axvline(x=lap_ini, linestyle='--', linewidth=1.0, color=line_color, label=f'{tipo} {cont + 1}')

                # Añado las vueltas como type
                laps_driver_1.loc[laps_driver_1['LapNumber'] == lap_ini, "LapType"] = tipo
                laps_driver_2.loc[laps_driver_2['LapNumber'] == lap_ini, "LapType"] = tipo

                # Actualizar cont
                cont_red += 1

        elif sum(mask_tipo) == 1:
            if tipo != "RedFlag":
                if lap_ini == lap_fin:
                    # Plot a vertical line in the extremes
                    ax1.axvline(x=lap_ini, linestyle='--', linewidth=1.0, color=line_color, label=tipo)

                    # Añado las vueltas como type
                    laps_driver_1.loc[laps_driver_1['LapNumber'] == lap_ini, "LapType"] = tipo
                    laps_driver_2.loc[laps_driver_2['LapNumber'] == lap_ini, "LapType"] = tipo

                    # Actualizar cont
                    if tipo == "VirtualSafetyCar":
                        cont_vsc += 1
                    elif tipo == "SafetyCar":
                        cont_sc += 1

                else:
                    # Hay mas de uno
                    mask_lap1 = laps_driver_1['LapNumber'] >= lap_ini
                    mask_lap2 = laps_driver_1['LapNumber'] <= lap_fin

                    subset = laps_driver_1.loc[mask_lap1 & mask_lap2,]
                    ax1.fill_between(
                        subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                        timedelta(days=0, hours=0, minutes=0, seconds=0, milliseconds=0),  # Limite inferior
                        time_max,  # Limite superior del área sombreada en el eje x
                        color=color,
                        alpha=0.3,
                        label=tipo
                    )

                    # Plot a vertical line in the extremes
                    ax1.axvline(x=lap_ini, linestyle='--', linewidth=1.0, color=line_color)
                    ax1.axvline(x=lap_fin, linestyle='--', linewidth=1.0, color=line_color)

                    # Añado las vueltas como type
                    laps_driver_1.loc[laps_driver_1['LapNumber'].between(lap_ini, lap_fin), "LapType"] = tipo
                    laps_driver_2.loc[laps_driver_2['LapNumber'].between(lap_ini, lap_fin), "LapType"] = tipo

                    # Actualizar cont
                    if tipo == "VirtualSafetyCar":
                        cont_vsc += 1
                    elif tipo == "SafetyCar":
                        cont_sc += 1

            else:
                # Plot a vertical line in the extremes
                ax1.axvline(x=lap_ini, linestyle='-', linewidth=1.0, color=line_color, label=tipo)

                # Añado las vueltas como type
                laps_driver_1.loc[laps_driver_1['LapNumber'] == lap_ini, "LapType"] = tipo
                laps_driver_2.loc[laps_driver_2['LapNumber'] == lap_ini, "LapType"] = tipo

                # Actualizar cont
                cont_red += 1


def safety_control_function_deltas(laps_driver_1, laps_driver_2, delta, event, ax1):
    # COnstants and variables
    dict_control = {
        "SafetyCar": {
            "tipo_eventos": "SAFETY CAR",
            "color": "gray",
            "line_color": "white"
        },
        "VirtualSafetyCar": {
            "tipo_eventos": "VIRTUAL SAFETY CAR",
            "color": "#d3d3a1",
            "line_color": "goldenrod"
        },
        "RedFlag": {
            "tipo_eventos": "VIRTUAL SAFETY CAR",
            "color": "red",
            "line_color": "red"
        }
    }

    # Crear Dataframe de eventos que llevan a parar la carrera
    df_race_control = event.race_control_messages
    mask_safetycar = df_race_control["Category"] == "SafetyCar"
    mask_redflags = df_race_control["Flag"] == "RED"
    mask_checkered = df_race_control["Flag"] == "CHEQUERED"

    mask_race_start_behind_sc = df_race_control["Message"] == "RACE WILL START BEHIND THE SAFETY CAR"
    mask_rolling_start = df_race_control["Message"] == "ROLLING START"
    df_race_control = df_race_control.loc[mask_safetycar | mask_redflags | mask_checkered | mask_race_start_behind_sc | mask_rolling_start,]

    # Recorrer eventos por ver si estan en df_race_control
    type_events = ["VirtualSafetyCar", "SafetyCar", "RedFlag"]
    race_stats = []
    for type in type_events:
        if type == "VirtualSafetyCar":
            mask_vsc = df_race_control["Message"].str.startswith("VIRTUAL")
            if sum(mask_vsc) > 0:
                # Hay VSC
                race_stats = manage_vsc(df_race_control, mask_vsc, race_stats)
        elif type == "SafetyCar":
            mask_sc = df_race_control["Message"].str.contains("SAFETY")
            mask_vsc = df_race_control["Message"].str.startswith("VIRTUAL")
            mask_joined = mask_sc & ~mask_vsc
            if sum(mask_joined) > 0:
                # Hay SC
                race_stats = manage_sc(df_race_control, mask_sc, race_stats)
        elif type == "RedFlag":
            mask_redflag = df_race_control["Flag"] == "RED"
            if sum(mask_redflag) > 0:
                # Hay RedFlag
                race_stats = manage_redflag(df_race_control, mask_redflag, race_stats)

    # Resultado
    '''Race_stats = [
       {"tipo": "VirtualSafetyCar", "laps": [28, 28]},
       {"tipo": "SafetyCar", "laps": [30, 32]},
       {"tipo": "RedFlag", "laps": [32, 32]},
       {"tipo": "SafetyCar", "laps": [39, 42]}
    ]'''

    # Pasar Race stats a df
    df_race_stats = pd.DataFrame(race_stats)

    # Recorrer Race Stats
    cont_sc = 0
    cont_vsc = 0
    cont_red = 0
    for i in range(df_race_stats.shape[0]):
        tipo = df_race_stats["tipo"].iloc[i]
        mask_tipo = df_race_stats["tipo"] == tipo

        color = dict_control[tipo]["color"]
        line_color = dict_control[tipo]["line_color"]

        laps = df_race_stats["laps"].iloc[i]
        lap_ini = laps[0]
        lap_fin = laps[1]

        # Asignar cnt depende del tipo
        if tipo == "VirtualSafetyCar":
            cont = cont_vsc
        elif tipo == "SafetyCar":
            cont = cont_sc
        elif tipo == "RedFlag":
            cont = cont_red

        if sum(mask_tipo) > 1:
            if tipo != "RedFlag":
                if lap_ini == lap_fin:
                    # Plot a vertical line in the extremes
                    ax1.axvline(x=lap_ini, linestyle='--', linewidth=1.0, color=line_color, label=tipo)

                    # Añado las vueltas como type
                    laps_driver_1.loc[laps_driver_1['LapNumber'] == lap_ini, "LapType"] = tipo
                    laps_driver_2.loc[laps_driver_2['LapNumber'] == lap_ini, "LapType"] = tipo

                    # Actualizar cont y Añado texto de tipo
                    if tipo == "VirtualSafetyCar":
                        ax1.text(lap_ini, delta*0.9, f"VSC {cont + 1}", fontdict={"fontsize": 14}, ha="center", va="center", color=line_color)
                        cont_vsc += 1
                    elif tipo == "SafetyCar":
                        ax1.text(lap_ini, delta*0.9, f"SC {cont + 1}", fontdict={"fontsize": 14}, ha="center", va="center", color=line_color)
                        cont_sc += 1
                else:
                    mask_lap1 = laps_driver_1['LapNumber'] >= lap_ini
                    mask_lap2 = laps_driver_1['LapNumber'] <= lap_fin

                    subset = laps_driver_1.loc[mask_lap1 & mask_lap2,]
                    ax1.fill_between(
                        subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                        -delta, # Limite inferior
                        delta,  # Limite superior
                        color=color,
                        alpha=0.3,
                        label=f'{tipo} {cont + 1}'
                    )

                    # Plot a vertical line in the extremes
                    ax1.axvline(x=lap_ini, linestyle='--', linewidth=1.0, color=line_color)
                    ax1.axvline(x=lap_fin, linestyle='--', linewidth=1.0, color=line_color)

                    # Añado las vueltas como type
                    laps_driver_1.loc[laps_driver_1['LapNumber'].between(lap_ini, lap_fin), "LapType"] = tipo
                    laps_driver_2.loc[laps_driver_2['LapNumber'].between(lap_ini, lap_fin), "LapType"] = tipo

                    # Actualizar cont y Añado texto de tipo
                    lap_med = (lap_ini + lap_fin) / 2
                    if tipo == "VirtualSafetyCar":
                        ax1.text(lap_med, delta*0.9, f"VSC {cont + 1}", fontdict={"fontsize": 14}, ha="center", va="center", color=line_color)
                        cont_vsc += 1
                    elif tipo == "SafetyCar":
                        ax1.text(lap_med, delta*0.9, f"SC {cont + 1}", fontdict={"fontsize": 14}, ha="center", va="center", color=line_color)
                        cont_sc += 1

            else:
                # Plot a vertical line in the extremes
                ax1.axvline(x=lap_ini, linestyle='--', linewidth=1.0, color=line_color, label=f'{tipo} {cont + 1}')

                # Añado las vueltas como type
                laps_driver_1.loc[laps_driver_1['LapNumber'] == lap_ini, "LapType"] = tipo
                laps_driver_2.loc[laps_driver_2['LapNumber'] == lap_ini, "LapType"] = tipo

                # Actualizar cont y Añado texto de tipo
                ax1.text(lap_ini, delta*0.9, "RedFlag", fontdict={"fontsize": 14}, ha="center", va="center", color=line_color)
                cont_red += 1

        elif sum(mask_tipo) == 1:
            if tipo != "RedFlag":
                if lap_ini == lap_fin:
                    # Plot a vertical line in the extremes
                    ax1.axvline(x=lap_ini, linestyle='--', linewidth=1.0, color=line_color, label=tipo)

                    # Añado las vueltas como type
                    laps_driver_1.loc[laps_driver_1['LapNumber'] == lap_ini, "LapType"] = tipo
                    laps_driver_2.loc[laps_driver_2['LapNumber'] == lap_ini, "LapType"] = tipo

                    # Actualizar cont
                    if tipo == "VirtualSafetyCar":
                        ax1.text(lap_ini, delta*0.9, f"VSC", fontdict={"fontsize": 14}, ha="center", va="center", color=line_color)
                        cont_vsc += 1
                    elif tipo == "SafetyCar":
                        ax1.text(lap_ini, delta*0.9, f"SC", fontdict={"fontsize": 14}, ha="center", va="center", color=line_color)
                        cont_sc += 1

                else:
                    # Hay mas de uno
                    mask_lap1 = laps_driver_1['LapNumber'] >= lap_ini
                    mask_lap2 = laps_driver_1['LapNumber'] <= lap_fin

                    subset = laps_driver_1.loc[mask_lap1 & mask_lap2,]
                    ax1.fill_between(
                        subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                        -delta,  # Limite inferior
                        delta,  # Limite superior
                        color=color,
                        alpha=0.3,
                        label=tipo
                    )

                    # Plot a vertical line in the extremes
                    ax1.axvline(x=lap_ini, linestyle='--', linewidth=1.0, color=line_color)
                    ax1.axvline(x=lap_fin, linestyle='--', linewidth=1.0, color=line_color)

                    # Añado las vueltas como type
                    laps_driver_1.loc[laps_driver_1['LapNumber'].between(lap_ini, lap_fin), "LapType"] = tipo
                    laps_driver_2.loc[laps_driver_2['LapNumber'].between(lap_ini, lap_fin), "LapType"] = tipo

                    # Actualizar cont
                    lap_med = (lap_ini + lap_fin) / 2
                    if tipo == "VirtualSafetyCar":
                        ax1.text(lap_med, delta*0.9, f"VSC", fontdict={"fontsize": 14}, ha="center", va="center", color=line_color)
                        cont_vsc += 1
                    elif tipo == "SafetyCar":
                        ax1.text(lap_med, delta*0.9, f"SC", fontdict={"fontsize": 14}, ha="center", va="center", color=line_color)
                        cont_sc += 1

            else:
                # Plot a vertical line in the extremes
                ax1.axvline(x=lap_ini, linestyle='-', linewidth=1.0, color=line_color, label=tipo)

                # Añado las vueltas como type
                laps_driver_1.loc[laps_driver_1['LapNumber'] == lap_ini, "LapType"] = tipo
                laps_driver_2.loc[laps_driver_2['LapNumber'] == lap_ini, "LapType"] = tipo

                # Actualizar cont
                ax1.text(lap_ini, delta*0.9, "RedFlag", fontdict={"fontsize": 14}, ha="center", va="center", color=line_color)
                cont_red += 1


def scatter_compound_driver(laps_driver, compound_colors, alpha, ax):
    # Recorrer los compuestos usados por piloto
    for compound in laps_driver['Compound'].unique():
        # Creo subconjunto de vueltas con ese compuesto y solo vueltas "Normal"
        mask_compound = laps_driver['Compound'] == compound
        mask_normal_laps = laps_driver["LapType"] == "Normal"
        subset = laps_driver.loc[mask_compound & mask_normal_laps,].copy()

        # Scatter
        ax.scatter(
            subset['LapNumber'],
            subset['LapTime'],
            color=compound_colors[compound],
            alpha=alpha,
            zorder=2
        )


def graph_driver_laptimes(laps_driver, driver, color_driver, alpha_driver, linestyle_driver, ax):
    # Obtener los Stints del driver
    stints_driver = laps_driver['Stint'].unique().tolist()

    for i, stint in enumerate(stints_driver):
        # Crear el subset
        mask_stint = laps_driver['Stint'] == stint
        mask_laps = laps_driver["LapType"] == "Normal"
        subset_stint = laps_driver.loc[mask_stint & mask_laps,]

        if i == (len(stints_driver) - 1):
            # Añado el label SOLO en el último caso
            ax.plot(
                subset_stint['LapNumber'],
                subset_stint['LapTime'],
                label=driver,
                color=color_driver,
                alpha=alpha_driver,
                linewidth=2,
                linestyle=linestyle_driver,
                zorder=1
            )
        else:
            ax.plot(
                subset_stint['LapNumber'],
                subset_stint['LapTime'],
                color=color_driver,
                alpha=alpha_driver,
                linewidth=2,
                linestyle=linestyle_driver,
                zorder=1
            )


def calcula_time_limits_and_driver(laps_driver_1, laps_driver_2):
    # Subset
    laps_normal_driver_1 = laps_driver_1.loc[laps_driver_1['LapType'] == "Normal", 'LapTime'].to_list()
    laps_normal_driver_2 = laps_driver_2.loc[laps_driver_2['LapType'] == "Normal", 'LapTime'].to_list()

    # Limits
    time_max_d1 = max(laps_normal_driver_1)
    time_max_d2 = max(laps_normal_driver_2)
    time_min_d1 = min(laps_normal_driver_1)
    time_min_d2 = min(laps_normal_driver_2)

    if time_max_d1 >= time_max_d2:
        time_max = time_max_d1
    else:
        time_max = time_max_d2

    if time_min_d1 <= time_min_d2:
        time_min = time_min_d1
        driver = "1"
    else:
        time_min = time_min_d2
        driver = "2"

    return time_min, time_max, driver


def determina_media_std_del_piloto_tmin(laps):
    laps_normal = laps.loc[laps['LapType'] == "Normal", 'LapTime'].to_list()
    laps_in_seconds = [lap.total_seconds() for lap in laps_normal]

    mean = pd.to_timedelta(np.mean(laps_in_seconds), unit='s')
    std = pd.to_timedelta(np.std(laps_in_seconds), unit='s')

    return mean, std


def calcula_x_limits(laps_driver_1):
    # Obtain max_laps
    max_laps = max(laps_driver_1['LapNumber'])
    return 0.5, max_laps + 0.5

def calcula_y_limits(laps_driver_1, laps_driver_2):
    # Calcula los time limits
    time_min, time_max, driver = calcula_time_limits_and_driver(laps_driver_1, laps_driver_2)

    # Determinar mean and std
    if driver == "1":
        mean, std = determina_media_std_del_piloto_tmin(laps_driver_1)
    elif driver == "2":
        mean, std = determina_media_std_del_piloto_tmin(laps_driver_2)

    # EXTREMOS
    delta = timedelta(days=0, hours=0, minutes=0, seconds=2, milliseconds=0)
    # y_min = mean - 2*std
    y_min = time_min - delta
    y_max = time_max + delta

    return y_min, y_max


def add_team_logo(ax, x, y, zoom):
    # Load logo
    path = os.path.join(os.getcwd(), "src", "outputs", "logo_graficos.png")
    img = mpimg.imread(path)

    # Crear objeto imagen
    imagebox = OffsetImage(img, zoom=zoom, interpolation='bicubic', resample=True)

    # Crear la anotacion
    ab = AnnotationBbox(imagebox, (x, y),  # Coordenadas (x, y) para la esquina inferior izquierda
                        xycoords='axes fraction',  # Coordenadas relativas al área de dibujo
                        box_alignment=(1, 0),
                        # Alinear la esquina inferior izquierda de la imagen con las coordenadas
                        frameon=False)  # Sin marco alrededor de la imagen

    # Añadir al eje
    ax.add_artist(ab)


def plot_axis1(
    laps_driver_1, 
    laps_driver_2, 
    compound_colors, 
    alphas, 
    drivers, 
    color_drivers, 
    linestyles, 
    plot_title, 
    ax
):
    # Extraction
    alpha_driver_1 = alphas[0]
    alpha_driver_2 = alphas[1]

    driver1 = drivers[0]
    driver2 = drivers[1]

    color_driver_1 = color_drivers[0]
    color_driver_2 = color_drivers[1]

    linestyle_driver_1 = linestyles[0]
    linestyle_driver_2 = linestyles[1]

    # 1) Puntos del neumatico usado en la vuelta
    scatter_compound_driver(laps_driver_1, compound_colors, alpha_driver_1, ax)
    scatter_compound_driver(laps_driver_2, compound_colors, alpha_driver_2, ax)

    # 2) LapTime vs LapNumber (Linea)
    graph_driver_laptimes(laps_driver_1, driver1, color_driver_1, alpha_driver_1, linestyle_driver_1, ax)
    graph_driver_laptimes(laps_driver_2, driver2, color_driver_2, alpha_driver_2, linestyle_driver_2, ax)

    # 3) Set Axis Limits
    ## 3.1) Calcular los limites
    y_min, y_max = calcula_y_limits(laps_driver_1, laps_driver_2)
    x_min, x_max = calcula_x_limits(laps_driver_1)

    ## 3.2) Añadir los limites al grafico
    ax.set_ylim([y_min, y_max])
    ax.set_xlim([x_min, x_max])
    ax.set_title(plot_title, fontsize=22, fontweight='bold')

    # 4) Titulo y nombre de los ejes
    handles, labels = ax.get_legend_handles_labels()
    num_elementos_leyenda = len(labels)

    if num_elementos_leyenda > 6:
        ax.legend(loc='upper right', ncols=4, fontsize=12)
    else:
        ax.legend(fontsize=12)

    # ax.set_xlabel('Vuelta', fontsize=16)
    ax.set_ylabel('Tiempo por vuelta', fontsize=16)
    ax.tick_params(axis='y', which='major', labelsize=15, colors='white')
    ax.tick_params(axis='x', which='major', labelsize=15, colors='white')

    # 5) Añadir la imagen logo
    y = 0.75
    add_team_logo(ax, X, Y, ZOOM)


def plot_axis1_deltas(
    laps_driver_1, 
    laps_driver_2, 
    colors, 
    alphas, 
    linestyles, 
    drivers,
    final_lap_race, 
    plot_title, 
    event, 
    ax1
):
    # 1) Construir el df final
    df_deltas = build_df_deltas(laps_driver_1, laps_driver_2, drivers, event, final_lap_race)

    # 2) Safety Control - Out of Loop
    time_min, time_max = genera_extremos_delta(df_deltas)
    safety_control_function_deltas(laps_driver_1, laps_driver_2, time_max, event, ax1)

    # .. Future Loop ...
    # 3) Pitstop funcion, calcular tiempo perdido en la parada, show gap vuelta by driver
    text = calcula_pitstopstimes(laps_driver_1, drivers[0], event)
    text += calcula_pitstopstimes(laps_driver_2, drivers[1], event)

    if colors[0] == colors[1]:
        text = pitstop_function_deltas(
            laps_driver=laps_driver_1,
            time_max=time_max,
            color_driver=colors[0],
            driver_1=drivers[0],
            driver=drivers[0],
            text=text,
            df=df_deltas,
            event=event,
            ax1=ax1,
            hatch=None
        )
        text = pitstop_function_deltas(
            laps_driver=laps_driver_2,
            time_max=time_max,
            color_driver=colors[1],
            driver_1=drivers[0],
            driver=drivers[1],
            text=text,
            df=df_deltas,
            event=event,
            ax1=ax1,
            hatch='/'
        )
    else:
        text = pitstop_function_deltas(
            laps_driver=laps_driver_1,
            time_max=time_max,
            color_driver=colors[0],
            driver_1=drivers[0],
            driver=drivers[0],
            text=text,
            df=df_deltas,
            event=event,
            ax1=ax1,
            hatch=None
        )
        text = pitstop_function_deltas(
            laps_driver=laps_driver_2,
            time_max=time_max,
            color_driver=colors[1],
            driver_1=drivers[0],
            driver=drivers[1],
            text=text,
            df=df_deltas,
            event=event,
            ax1=ax1,
            hatch=None
        )

    # 4) Plotear Delta
    ax1.plot(
        df_deltas['LapNumber'],
        df_deltas[f'Delta_{drivers[1]}_{drivers[0]}'],
        label=drivers[1],
        color=colors[1],
        alpha=alphas[1],
        linewidth=3,
        linestyle=linestyles[1]
    )
    # print("Deltas finales")
    # print(f" Delta Inicio lap 42: {df_deltas[f'Delta_{drivers[1]}_{drivers[0]}'].iloc[40]}")
    # print(f" Delta Inicio lap 43: {df_deltas[f'Delta_{drivers[1]}_{drivers[0]}'].iloc[41]}")
    # print(f" Delta Inico lap 44: {df_deltas[f'Delta_{drivers[1]}_{drivers[0]}'].iloc[42]}")
    # print(f" Delta Final lap 44: {df_deltas[f'Delta_{drivers[1]}_{drivers[0]}'].iloc[43]}")

    # 5) Mostrar Gap Final
    text += "\nEl Gap final fue de {:.3f}s en la vuelta {}".format(
        df_deltas.loc[df_deltas.index[-1], f'Delta_{drivers[1]}_{drivers[0]}'],
        df_deltas.loc[df_deltas.index[-1], 'LapNumber']
    )
    # ...

    # 6) Plotear Linea de referencia Driver 1
    ax1.axhline(
        0.0,
        label=drivers[0],
        color=colors[0],
        alpha=alphas[0],
        linewidth=3,
        linestyle=linestyles[0]
    )

    # 7) Añadir logo
    ## Seleccionar la parte de la curva entre logo
    mask_lap15 = df_deltas['LapNumber'] <= 15
    y_curve = df_deltas.loc[mask_lap15, f'Delta_{drivers[1]}_{drivers[0]}']
    mask_delta_high = y_curve >= time_max *0.6

    if sum(mask_delta_high) > 0:
        y = 0.8
    else:
        y = Y

    add_team_logo(ax1, X, y, ZOOM)

    # 9) Poner titulo, ejes ..
    ax1.set_title(plot_title, fontsize=22, fontweight='bold')
    ax1.set_ylabel(f"Gap (s)", fontsize=17)
    ax1.set_xlabel("Vuelta", fontsize=17)
    ax1.set_ylim(time_min, time_max)
    ax1.set_xlim(0.5, final_lap_race + 0.5)
    ax1.tick_params(axis='y', which='major', labelsize=15, colors='white')
    ax1.tick_params(axis='x', which='major', labelsize=15, colors='white')
    ax1.invert_yaxis()  # invertir eje y
    ax1.legend(loc="upper right", fontsize=14, ncols=3)
    ax1.grid(True)

    return text


def graph_tyre_strategy(laps_driver, compound_colors, driver, max_laps, event, ax1, ax):
    # Necesary lists and values
    laps_pitin_driver, laps_pitout_driver = extract_real_pitin_pitout_laps(laps_driver, event)
    create_real_stint_column(laps_driver)
    stints_driver = np.unique(laps_driver['RealStint'].to_list())

    rect_y = 0
    rect_height = 0.1

    # Obtener la lista de Neumaticos usados por el Driver
    tyres_used_driver = []
    for stint in stints_driver:
        tyre = laps_driver.loc[laps_driver['RealStint'] == stint, 'Compound'].to_list()[0]
        tyres_used_driver.append(tyre)

    segment_colors = [compound_colors[tyre] for tyre in tyres_used_driver]

    previous_vuelta = 0
    vueltas = laps_pitin_driver

    for i, vuelta in enumerate(vueltas + [max_laps + 1]):  # Añadimos x.max() para el último segmento
        color = segment_colors[i % len(segment_colors)]
        ax.axvspan(previous_vuelta, vuelta, color=color, alpha=1.0)
        # Dibujar una línea divisoria negra
        if previous_vuelta != 0:  # No dibujar línea antes del primer segmento
            ax.axvline(x=previous_vuelta, color='black', linewidth=1.0)

        previous_vuelta = vuelta

    ax.set_xticks(vueltas)
    ax.set_ylim(rect_y, rect_y + rect_height)
    ax.set_yticks([])
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.tick_params(axis='x', which='major', labelsize=10, colors='white')
    ax.set_ylabel(driver, rotation=0, labelpad=20, va='center', fontsize=15)

    # Alinear los límites de los ejes x
    ax.set_xlim(ax1.get_xlim())
    ax.grid(False)

# Funcion plot
def plot_race_comparaison_two_drivers(laps_driver_1, laps_driver_2, driver_1, driver_2, event, save=True):
    # Limitacion a mano
    # laps_driver_1 = laps_driver_1[laps_driver_1['LapNumber']<66.0]
    # laps_driver_2 = laps_driver_2[laps_driver_2['LapNumber']<66.0]

    # ---------------- INICIO -----------------------------
    # 1) Encontrar la ultima vuelta comun de d1 y d2
    final_laps_race = obtain_last_lap(event=event)

    # 2) Corregir las vueltas de los pilotos por si hay alguna vuelta nula
    laps_driver_1['LapTime'] = generate_correct_laptimes(laps_driver_1)
    laps_driver_2['LapTime'] = generate_correct_laptimes(laps_driver_2)

    # 3) Asignar una nueva columna de tipo y poner Primera vuelta
    laps_driver_1['LapType'] = initialize_laptype(laps_driver_1, final_laps_race)
    laps_driver_2['LapType'] = initialize_laptype(laps_driver_2, final_laps_race)

    # 4) Iniciar grafico
    # 4.1) Titulo del grafico y nombre del archivo
    plot_title = f"{event.event.year} {event.event.EventName} {event.name} {driver_1} vs {driver_2}"
    plot_filename = plot_title.replace(" ", "_") + ".pdf"

    # 4.2) Setup del grafico - Creamos subplots de diferentes tamaños
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1,
                                        gridspec_kw={'height_ratios': [6, 0.35, 0.35]},
                                        figsize=(12, 7),
                                        dpi=700)

    # 4.3) Genera color, linestyle y alpha de los graficos
    color_driver1, color_driver2, linestyle_driver_1, linestyle_driver_2, alpha_driver_1, alpha_driver_2 = generate_color_linestyle_apha(laps_driver_1, laps_driver_2, event)

    # 4.4)  Definimos el diccinario de los colores de los neumáticos
    compound_colors = {'SOFT': 'red', 'MEDIUM': 'yellow', 'HARD': 'white', 'INTERMEDIATE': 'green','EXTREMEWET': 'blue'}

    # 4.5) Crea limites de tiemp del grafico
    time_min, time_max = genera_extremos_tiempo(laps_driver_1, laps_driver_2)

    # ------------------ PLOTEO -------------------------------------------------------
    ## AXIS1 - Tiempo por vuelta, punto de neumático
    # ---------------- PITSTOP -----------------------------
    # 5.1) Busca pitstop, colorea las vueltas, añada typo de vuelta
    if color_driver1 == color_driver2:
        pitstop_function(laps_driver_1, time_max, color_driver1, driver_1, event, ax1, None)
        pitstop_function(laps_driver_2, time_max, color_driver2, driver_2, event, ax1, '///')
    else:
        pitstop_function(laps_driver_1, time_max, color_driver1, driver_1, event, ax1, None)
        pitstop_function(laps_driver_2, time_max, color_driver2, driver_2, event, ax1, None)

    # ---------------- CONTROL EVENTS: SAFETYCARS, VSC & RED FLAGS -----------------------------
    # 6) Busca SAFETY CONTROL, colorea las vueltas, añada typo de vuelta - Driver 1 & Driver 2
    safety_control_function(laps_driver_1, laps_driver_2, time_max, event, ax1)

    # TODO ---------------- YELLOW FLAGS Y DOUBLE YELLOW FLAG -----------------------------
    # 7.1) Busca YELLOW FLAG, colorea las vueltas, añada typo de vuelta - Driver 1 & Driver 2
    # 7.2) Busca DOUBLE YELLOW, colorea las vueltas, añada typo de vuelta - Driver 1 & Driver 2

    # Nota: No veo mucho cambio en la comparativa con otras vueltas sin bandera amarilla,
    #       además que engorrosaría más el gráfico.

    alphas = [alpha_driver_1, alpha_driver_2]
    drivers = [driver_1, driver_2]
    color_drivers = [color_driver1, color_driver2]
    linestyles = [linestyle_driver_1, linestyle_driver_2]
    plot_axis1(
        laps_driver_1=laps_driver_1,
        laps_driver_2=laps_driver_2,
        compound_colors=compound_colors,
        alphas=alphas,
        drivers=drivers,
        color_drivers=color_drivers,
        linestyles=linestyles,
        plot_title=plot_title,
        ax=ax1
    )

    ## AXIS 2 - Estrategía de neumático del driver1
    graph_tyre_strategy(laps_driver_1, compound_colors, driver_1, final_laps_race, event, ax1, ax2)

    ## AXIS 3 - Estrategía de neumático del driver2
    graph_tyre_strategy(laps_driver_2, compound_colors, driver_2, final_laps_race, event, ax1, ax3)

    ## Alinear los límites de los ejes x
    ax3.set_xlim(ax1.get_xlim())

    # ----------------------------- SAVE ----------------------------------------------
    ## Ajustar márgenes
    plt.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.1, hspace=0.3)

    ## Guardado del gráfico
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        eventname = event.event.EventName.split(" ")[0]
        path = f"{event.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        # Crear Subdirectorio sino existe
        if not os.path.exists(path):
            os.makedirs(path)

        # Crear el subdirectorio carrera
        path = os.path.join(path, event.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar figura
        plot_filename = path + "/" + plot_filename
        plt.savefig(plot_filename, format='pdf')
    else:
        plt.show()


def graph_interval_two_drivers(laps_driver_1, laps_driver_2, drivers, event, save=True):
    # ----------------------------- INICIO ----------------------------------------------
    # 1.1) Encontrar la última vuelta de la carrera
    final_lap_race = obtain_last_lap(event=event)

    # 1.2) Inicializamos LapType pero teniendo en cuenta abandonos
    laps_driver_1['LapType'] = initialize_laptype(laps_driver_1, final_lap_race)
    laps_driver_2['LapType'] = initialize_laptype(laps_driver_2, final_lap_race)

    # 1.3) Corregir vueltas NA
    laps_driver_1['LapTime'] = generate_correct_laptimes(laps_driver_1)
    laps_driver_2['LapTime'] = generate_correct_laptimes(laps_driver_2)

    # 1.4) Poner orden de los pilotos por clasificación final del evento y asignar bien laps_driver_1 y laps_driver_2
    drivers_order = order_driver_by_sesion(event, drivers)
    laps_driver_1, laps_driver_2 = order_the_laps(laps_driver_1, laps_driver_2, drivers_order)

    # 1.5) Asignar line_style, alphas, colores
    teams = get_list_teams_ordered_by_sesion(event, drivers_order)
    colors, linestyles, alphas = generate_color_linestyle_apha_v2(teams, drivers_order, event)

    # ----------------------------- PLOT ----------------------------------------------
    # 2.0) Defino grafico
    driver_first = drivers_order[0]
    driver_second = drivers_order[1]
    plot_title = (f"{event.event.year} {event.event.EventName} {event.name} Distancia en carrera "
                  f"{driver_first} y {driver_second}")
    plot_filename = plot_title.replace(" ", "_") + ".pdf"

    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 7), dpi=700)

    # 2.1) Genero grafico
    text = plot_axis1_deltas(
        laps_driver_1,
        laps_driver_2,
        colors,
        alphas,
        linestyles,
        drivers_order,
        final_lap_race,
        plot_title,
        event,
        ax1
    )

    # ----------------------------- SAVE ----------------------------------------------
    ## Ajustar márgenes
    plt.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.1, hspace=0.3)

    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        eventname = event.event.EventName.split(" ")[0]
        path = f"{event.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)
        path = os.path.join(path, event.name)

        # Export text
        pathtext = f"Race_data_{driver_first}_{driver_second}.txt"
        f = open(os.path.join(path, pathtext), "w")
        f.write(text)
        f.close()

        # Guardar figura
        plot_filename = path + "/" + plot_filename
        plt.savefig(plot_filename, format='pdf')
    else:
        plt.show()


def graph_interval_two_drivers_zoomed(laps_driver_1, laps_driver_2, drivers, laps, event, save=True):
    # ----------------------------- INICIO ----------------------------------------------
    # 1.1) Encontrar la última vuelta de la carrera
    final_lap_race = obtain_last_lap(event=event)
    # 1.2) Inicializamos LapType pero teniendo en cuenta abandonos
    laps_driver_1['LapType'] = initialize_laptype(laps_driver_1, final_lap_race)
    laps_driver_2['LapType'] = initialize_laptype(laps_driver_2, final_lap_race)
    # 1.3) Corregir vueltas NA
    laps_driver_1['LapTime'] = generate_correct_laptimes(laps_driver_1)
    laps_driver_2['LapTime'] = generate_correct_laptimes(laps_driver_2)
    # 1.4) Poner orden de los pilotos por clasificación final del evento y asignar bien laps_driver_1 y laps_driver_2
    drivers_order = order_driver_by_sesion(event, drivers)
    laps_driver_1, laps_driver_2 = order_the_laps(laps_driver_1, laps_driver_2, drivers_order)
    # 1.5) Asignr line_style, alphas, colores
    teams = get_list_teams_ordered_by_sesion(event, drivers_order)
    colors, linestyles, alphas = generate_color_linestyle_apha_v2(teams, drivers_order, event)
    # 1.6) Create df_deltas
    df_deltas = build_df_deltas(laps_driver_1, laps_driver_2, drivers, event, final_lap_race)
    # 1.7) Apply zoom
    mask_lap_ini = df_deltas['LapNumber'] >= laps[0]
    mask_lap_fin = df_deltas['LapNumber'] <= laps[1]
    subset = df_deltas.loc[mask_lap_ini&mask_lap_fin,]
    # ----------------------------- PLOT ----------------------------------------------
    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 7), dpi=700)
    ax1.plot(
        subset['LapNumber'],
        subset[f'Delta_{drivers[1]}_{drivers[0]}'],
        label=drivers[1],
        color=colors[1],
        alpha=alphas[1],
        linewidth=2,
        linestyle=linestyles[1]
    )
    ax1.axhline(
        0.0,
        label=drivers[0],
        color=colors[0],
        alpha=alphas[0],
        linewidth=2,
        linestyle=linestyles[0]
    )
    ax1.set_title("Gap entre Antonelli y Hamilton")
    ax1.set_ylabel(f"Gap a {driver[0]}(s)")
    ax1.set_xlabel("Vuelta")
    ax1.set_xlim(laps[0] - 0.5, laps[1] + 0.5)
    ax1.set_ylim(0.0, max(subset[f"Delta_{drivers[1]}_{drivers[0]}"]))
    # ax1.invert_yaxis()  # invertir eje y
    ax1.legend()
    ax1.grid(True)
    path = os.path.join(os.getcwd(), "src", "outputs", "gap_ANT_HAM.png")
    plt.savefig(path, format='png')
