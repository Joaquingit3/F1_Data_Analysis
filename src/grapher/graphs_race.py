# Library
import os
import requests
import pycountry
import numpy as np
import pandas as pd
import fastf1 as ff1
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

from io import BytesIO
from PIL import Image
from collections import defaultdict
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.text import TextPath
from matplotlib.patches import FancyArrowPatch, Ellipse
import matplotlib.patches as patches

from tabulate import tabulate

from fastf1 import plotting
from fastf1 import utils
from datetime import timedelta

from sklearn import linear_model

from src.utils.functions import format_time, calcula_lateralacc, calcula_longacc, paint_maxspeed, find_maxspeed_one_driver, get_list_sorted, load_image
from src.utils.functions_race import show_gap_vuelta_by_driver, check_dnf, check_if_driver_dnf
from src.grapher.graph_comparaison_race import obtain_last_lap
from src.logger.logging_utils import find_drivers_noLap_warning
from src.creators.creator import convert_and_save_in_png
from src.s3_manager.s3_functions import upload_file_s3
from src.utils.functions import DICT_STATUS
from src.common.const import DICT_RACE_COUNTRYCODE

FASTF1_DEBUG=1


def plot_race_comparaison_two_drivers(laps_driver_1, laps_driver_2, driver_1, driver_2, corners, event, save=True):
    # ---------------- PILOTOS ABANDONA -----------------------------
    # 1.1) Buscar la última vuelta de los pilotos
    total_laps = event.total_laps

    final_lap1 = check_if_driver_dnf(laps_driver_1, total_laps)
    final_lap2 = check_if_driver_dnf(laps_driver_2, total_laps)

    # 1.2) Definimos la ultima vuelta del gráfico como la menor vuelta
    #      final de los dos pilotos
    if final_lap1 < final_lap2:
        final_lap = final_lap1
    else:
        final_lap = final_lap2

    # 1.3) Quitar las vueltas superiores a final_lap en ambos
    if final_lap < total_laps:
        laps_driver_1 = laps_driver_1[
            ~laps_driver_1['LapNumber'].between(final_lap, total_laps)]
        laps_driver_2 = laps_driver_2[
            ~laps_driver_2['LapNumber'].between(final_lap, total_laps)]

    # ---------------- VUELTAS NULAS -----------------------------
    # 2) Ver si alguna vuelta es nula y borrarlas
    mask_isna_1 = laps_driver_1['LapTime'].isna()
    mask_isna_2 = laps_driver_2['LapTime'].isna()

    laps_isna_driver1 = laps_driver_1.loc[mask_isna_1, 'LapNumber'].to_list()
    laps_isna_driver2 = laps_driver_2.loc[mask_isna_2, 'LapNumber'].to_list()

    laps_driver_1 = laps_driver_1.loc[~mask_isna_1,]
    laps_driver_2 = laps_driver_2.loc[~mask_isna_2,]

    # ---------------- GRAPH SETUP -----------------------------
    # 3.1) Titulo del grafico y nombre del archivo
    plot_title = f"{event.event.year} {event.event.EventName} {event.name} {driver_1} vs {driver_2}"
    plot_filename = plot_title.replace(" ", "_") + ".pdf"

    # 3.2) Setup del grafico
    # Creamos subplots de diferentes tamaños
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1,
                                        gridspec_kw={'height_ratios': [6, 0.35, 0.35]},
                                        figsize=(12, 7),
                                        dpi=700)

    # 3.3) Cargamos el color, estilo y alpha de los pilotos
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

    # 3.4) Definimos el diccinario de los colores de los neumáticos
    compound_colors = {'SOFT': 'red', 'MEDIUM': 'yellow', 'HARD': 'white', 'INTERMEDIATE': 'green',
                       'EXTREMEWET': 'blue'}

    # 3.5) Limites de tiempo (min y MAX) para los graficos
    time_max_d1 = max(laps_driver_1['LapTime'])
    time_max_d2 = max(laps_driver_2['LapTime'])
    time_min_d1 = min(laps_driver_1['LapTime'])
    time_min_d2 = min(laps_driver_2['LapTime'])

    if time_max_d1 >= time_max_d2:
        time_max = time_max_d1
    else:
        time_max = time_max_d2

    if time_min_d1 <= time_min_d2:
        time_min = time_min_d1
    else:
        time_min = time_min_d2

    # ---------------- PITSTOP -----------------------------
    # 4) Buscar las vueltas de PARADA
    ## Driver 1
    laps_pitin_driver_1 = laps_driver_1.loc[laps_driver_1['PitInTime'].notna(), 'LapNumber'].to_list()
    laps_pitout_driver_1 = laps_driver_1.loc[laps_driver_1['PitOutTime'].notna(), 'LapNumber'].to_list()

    if len(laps_pitout_driver_1) < len(laps_pitin_driver_1):
        # Ha habido un abandono posiblemente => Delimitamos con el vector más pequeño
        for i in range(len(laps_pitout_driver_1)):
            # Pintar franja de parada
            mask_lap1 = laps_driver_1['LapNumber'] >= laps_pitin_driver_1[i]
            mask_lap2 = laps_driver_1['LapNumber'] <= laps_pitout_driver_1[i]
            subset = laps_driver_1.loc[mask_lap1 & mask_lap2,]
            ax1.fill_between(
                subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                timedelta(days=0, hours=0, minutes=0, seconds=0, milliseconds=0),
                # Limite inferior del área sombreada en el eje x
                time_max,  # Limite superior del área sombreada en el eje x
                color=color_driver1, alpha=0.3, label=f'{driver_1} PitStop {i + 1}'
            )
            # Quitar vueltas
            laps_driver_1 = laps_driver_1[
                ~laps_driver_1['LapNumber'].between(laps_pitin_driver_1[i], laps_pitout_driver_1[i])]

    else:
        for i in range(len(laps_pitin_driver_1)):
            # Pintar franja de parada
            mask_lap1 = laps_driver_1['LapNumber'] >= laps_pitin_driver_1[i]
            mask_lap2 = laps_driver_1['LapNumber'] <= laps_pitout_driver_1[i]
            subset = laps_driver_1.loc[mask_lap1 & mask_lap2,]
            ax1.fill_between(
                subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                timedelta(days=0, hours=0, minutes=0, seconds=0, milliseconds=0),
                # Limite inferior del área sombreada en el eje x
                time_max,  # Limite superior del área sombreada en el eje x
                color=color_driver1, alpha=0.3, label=f'{driver_1} PitStop {i + 1}'
            )
            # Quitar vueltas
            laps_driver_1 = laps_driver_1[
                ~laps_driver_1['LapNumber'].between(laps_pitin_driver_1[i], laps_pitout_driver_1[i])]
    ## Driver 2
    laps_pitin_driver_2 = laps_driver_2.loc[laps_driver_2['PitInTime'].notna(), 'LapNumber'].to_list()
    laps_pitout_driver_2 = laps_driver_2.loc[laps_driver_2['PitOutTime'].notna(), 'LapNumber'].to_list()

    if len(laps_pitout_driver_2) < len(laps_pitin_driver_2):
        # Ha habido un abandono posiblemente => Delimitamos con el vector más pequeño
        for i in range(len(laps_pitout_driver_2)):
            # Pintar franja de parada
            mask_lap1 = laps_driver_2['LapNumber'] >= laps_pitin_driver_2[i]
            mask_lap2 = laps_driver_2['LapNumber'] <= laps_pitout_driver_2[i]
            subset = laps_driver_2.loc[mask_lap1 & mask_lap2,]
            ax1.fill_between(
                subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                timedelta(days=0, hours=0, minutes=0, seconds=0, milliseconds=0),
                # Limite inferior del área sombreada en el eje x
                time_max,  # Limite superior del área sombreada en el eje x
                color=color_driver2, alpha=0.3, label=f'{driver_2} PitStop {i + 1}'
            )
            # Quitar vueltas
            laps_driver_2 = laps_driver_2[
                ~laps_driver_2['LapNumber'].between(laps_pitin_driver_2[i], laps_pitout_driver_2[i])]

    else:
        for i in range(len(laps_pitin_driver_2)):
            # Pintar franja de parada
            mask_lap1 = laps_driver_2['LapNumber'] >= laps_pitin_driver_2[i]
            mask_lap2 = laps_driver_2['LapNumber'] <= laps_pitout_driver_2[i]
            subset = laps_driver_2.loc[mask_lap1 & mask_lap2,]
            ax1.fill_between(
                subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                timedelta(days=0, hours=0, minutes=0, seconds=0, milliseconds=0),
                # Limite inferior del área sombreada en el eje x
                time_max,  # Limite superior del área sombreada en el eje x
                color=color_driver2, alpha=0.3, label=f'{driver_2} PitStop {i + 1}'
            )
            # Quitar vueltas
            laps_driver_2 = laps_driver_2[
                ~laps_driver_2['LapNumber'].between(laps_pitin_driver_2[i], laps_pitout_driver_2[i])]

    # 4) Buscar SAFETY CARS Y Virtuals y borrar vueltas
    # Buscar Periodos de SafetyCar para eliminar de la grafica
    mask_safetycar = event.race_control_messages['Category'] == 'SafetyCar'

    if sum(mask_safetycar) > 0:
        # SafetyCars puros
        mask_safetycardeployed = event.race_control_messages['Message'] == 'SAFETY CAR DEPLOYED'
        mask_safetycarinthislap = event.race_control_messages['Message'] == 'SAFETY CAR IN THIS LAP'

        laps_scdeployed = event.race_control_messages.loc[mask_safetycardeployed, 'Lap'].to_list()
        laps_scinthislap = event.race_control_messages.loc[mask_safetycarinthislap, 'Lap'].to_list()

        # Tomar periodos de safetycar y eliminar esas vueltas del conteneo de vueltas
        if len(laps_scdeployed) == 1:
            # Plot the rectangle of SC
            i = 0
            mask_lap1 = laps_driver_1['LapNumber'] >= laps_scdeployed[i]
            mask_lap2 = laps_driver_1['LapNumber'] <= laps_scinthislap[i]
            subset = laps_driver_1.loc[mask_lap1 & mask_lap2,]
            ax1.fill_between(
                subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                timedelta(days=0, hours=0, minutes=0, seconds=0, milliseconds=0),  # Limite inferior
                time_max,  # Limite superior del área sombreada en el eje x
                color='gray', alpha=0.3, label=f'Safety Car'
            )

            # Plot a vertical line in the extremes
            ax1.axvline(x=laps_scdeployed[i], linestyle='--', linewidth=1.0, color='darkgray')
            ax1.axvline(x=laps_scinthislap[i], linestyle='--', linewidth=1.0, color='darkgray')

            # Delete SafetyCar laps
            laps_driver_1 = laps_driver_1[
                ~laps_driver_1['LapNumber'].between(laps_scdeployed[i], laps_scinthislap[i])]
            laps_driver_2 = laps_driver_2[
                ~laps_driver_2['LapNumber'].between(laps_scdeployed[i], laps_scinthislap[i])]

        elif len(laps_scdeployed) > 1:
            for i in range(0, len(laps_scdeployed)):
                # Plot the rectangle of SC
                mask_lap1 = laps_driver_1['LapNumber'] >= laps_scdeployed[i]
                mask_lap2 = laps_driver_1['LapNumber'] <= laps_scinthislap[i]
                subset = laps_driver_1.loc[mask_lap1 & mask_lap2,]
                ax1.fill_between(
                    subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                    timedelta(days=0, hours=0, minutes=0, seconds=0, milliseconds=0), # Limite inferior
                    time_max,  # Limite superior del área sombreada en el eje x
                    color='gray', alpha=0.3, label=f'Safety Car {i + 1}'
                )

                # Plot a vertical line in the extremes
                ax1.axvline(x=laps_scdeployed[i], linestyle='--', linewidth=1.0, color='darkgray')
                ax1.axvline(x=laps_scinthislap[i], linestyle='--', linewidth=1.0, color='darkgray')

                # Delete SafetyCar laps
                laps_driver_1 = laps_driver_1[
                    ~laps_driver_1['LapNumber'].between(laps_scdeployed[i], laps_scinthislap[i])]
                laps_driver_2 = laps_driver_2[
                    ~laps_driver_2['LapNumber'].between(laps_scdeployed[i], laps_scinthislap[i])]

        # Buscar Virtuals SafetyCar
        mask_vsafetycardeployed = event.race_control_messages['Message'] == 'VIRTUAL SAFETY CAR DEPLOYED'
        mask_vsafetycarinthislap = event.race_control_messages['Message'] == 'VIRTUAL SAFETY CAR ENDING'

        laps_vscdeployed = event.race_control_messages.loc[mask_vsafetycardeployed, 'Lap'].to_list()
        laps_vscinthislap = event.race_control_messages.loc[mask_vsafetycarinthislap, 'Lap'].to_list()

        # Tomar periodos de safetycar y eliminar esas vueltas del conteneo de vueltas
        for i in range(0, len(laps_vscdeployed)):
            if (laps_vscdeployed[i] not in laps_isna_driver1) or (laps_vscdeployed[i] not in laps_isna_driver2):
                # Plot the rectangle of SC
                mask_lap1 = laps_driver_1['LapNumber'] >= laps_vscdeployed[i]
                mask_lap2 = laps_driver_1['LapNumber'] <= laps_vscinthislap[i]
                subset = laps_driver_1.loc[mask_lap1 & mask_lap2,]
                ax1.fill_between(
                    subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
                    timedelta(days=0, hours=0, minutes=0, seconds=0, milliseconds=0),  # Limite inferior
                    time_max,  # Limite superior del área sombreada en el eje x
                    color='gray', alpha=0.3, label=f'Virtual Safety Car {i + 1}'
                )

                # Plot a vertical line in the extremes
                ax1.axvline(x=laps_vscdeployed[i], linestyle='--', linewidth=1.0, color='darkgray')
                ax1.axvline(x=laps_vscinthislap[i], linestyle='--', linewidth=1.0, color='darkgray')

                # Delete SafetyCar laps
                laps_driver_1 = laps_driver_1[
                    ~laps_driver_1['LapNumber'].between(laps_vscdeployed[i], laps_vscinthislap[i])]
                laps_driver_2 = laps_driver_2[
                    ~laps_driver_2['LapNumber'].between(laps_vscdeployed[i], laps_vscinthislap[i])]

    # 4.1) Buscar Banderas amarillas y doble banderas amarillas
    # Buscar Periodos de Bnadera amarilla y borrar la vuelta
    mask_1 = laps_driver_1['LapNumber'] == 50.0
    mask_2 = laps_driver_2['LapNumber'] == 50.0

    laps_driver_1 = laps_driver_1.loc[~mask_1,]
    laps_driver_2 = laps_driver_2.loc[~mask_2,]

    '''mask_dobleyellow = event.race_control_messages['Flag'] == 'DOUBLE YELLOW'
    mask_yellow = event.race_control_messages['Flag'] == 'YELLOW'

    laps_doubleyellow = event.race_control_messages.loc[mask_dobleyellow, 'Lap']
    laps_yellow = event.race_control_messages.loc[mask_yellow,'Lap']
    laps_total_yellow = laps_yellow + laps_doubleyellow
    laps_unique_yellow = np.unique(laps_total_yellow)

    if len(laps_unique_yellow) > 0:
        # Hay vueltas de bandera amarilla => Borrarlas
        for lap in laps_unique_yellow:
            mask1_lap = laps_driver_1['LapNumber'] == lap
            mask2_lap = laps_driver_2['LapNumber'] == lap

            laps_driver_1 = laps_driver_1.loc[~mask1_lap,]
            laps_driver_2 = laps_driver_2.loc[~mask2_lap,]'''

    # 5) PLOT DEL GRAFICO
    ## 5.1) Puntos del neumatico usado en la vuelta
    for compound in laps_driver_1['Compound'].unique():
        subset = laps_driver_1[laps_driver_1['Compound'] == compound]
        ax1.scatter(subset['LapNumber'], subset['LapTime'],
                    color=compound_colors[compound], alpha=alpha_driver_1, zorder=2)
    # se puede poner edgecolor pero en black no me convence

    for compound in laps_driver_2['Compound'].unique():
        subset = laps_driver_2[laps_driver_2['Compound'] == compound]
        ax1.scatter(subset['LapNumber'], subset['LapTime'],
                    color=compound_colors[compound], alpha=alpha_driver_2, zorder=2)

    ## 5.2) LapTime vs LapNumber (Linea)
    ### Stints driver 1
    stints_driver_1 = laps_driver_1['Stint'].unique().tolist()

    for i in range(len(stints_driver_1)):
        subset_stint = laps_driver_1.loc[laps_driver_1['Stint'] == stints_driver_1[i]]
        if i == (len(stints_driver_1) - 1):
            ax1.plot(subset_stint['LapNumber'], subset_stint['LapTime'], label=driver_1,
                     color=color_driver1, alpha=alpha_driver_1, linewidth=2, linestyle=linestyle_driver_1,
                     zorder=1)
        else:
            ax1.plot(subset_stint['LapNumber'], subset_stint['LapTime'],
                     color=color_driver1, alpha=alpha_driver_1, linewidth=2, linestyle=linestyle_driver_1,
                     zorder=1)

    ### Stints driver 2
    stints_driver_2 = laps_driver_2['Stint'].unique().tolist()
    for i in range(len(stints_driver_2)):
        subset_stint = laps_driver_2.loc[laps_driver_2['Stint'] == stints_driver_2[i]]
        if i == (len(stints_driver_2) - 1):
            ax1.plot(subset_stint['LapNumber'], subset_stint['LapTime'], label=driver_2,
                     color=color_driver2, alpha=alpha_driver_2, linewidth=2, linestyle=linestyle_driver_2,
                     zorder=1)
        else:
            ax1.plot(subset_stint['LapNumber'], subset_stint['LapTime'],
                     color=color_driver2, alpha=alpha_driver_2, linewidth=2, linestyle=linestyle_driver_2,
                     zorder=1)

    ax1.legend()
    ax1.set_xlabel('Vuelta')
    ax1.set_ylabel('Tiempo por vuelta')

    ## Set ylims
    time_max_d1 = max(laps_driver_1['LapTime'])
    time_max_d2 = max(laps_driver_2['LapTime'])
    time_min_d1 = min(laps_driver_1['LapTime'])
    time_min_d2 = min(laps_driver_2['LapTime'])
    max_laps = max(laps_driver_1['LapNumber'])

    if time_max_d1 >= time_max_d2:
        time_max = time_max_d1
    else:
        time_max = time_max_d2

    if time_min_d1 <= time_min_d2:
        time_min = time_min_d1
    else:
        time_min = time_min_d2

    delta = timedelta(days=0, hours=0, minutes=0, seconds=2, milliseconds=0)
    ax1.set_ylim([time_min - delta, time_max + delta])
    ax1.set_xlim([0.5, max_laps + 0.5])
    ax1.set_title(plot_title)

    ## Añadir la imagen logo
    path = os.path.join(os.getcwd(), "src", "outputs", "logo_graficos.png")
    img = mpimg.imread(path)
    imagebox = OffsetImage(img, zoom=0.08, interpolation='bicubic', resample=True)
    ab = AnnotationBbox(imagebox, (0.005, 0.01),  # Coordenadas (x, y) para la esquina inferior izquierda
                        xycoords='axes fraction',  # Coordenadas relativas al área de dibujo
                        box_alignment=(0, 0),  # Alinear la esquina inferior izquierda de la imagen con las coordenadas
                        frameon=False)  # Sin marco alrededor de la imagen
    ax1.add_artist(ab)

    ## 5.3) Estrategia de neumaticos
    rect_y = 0
    rect_height = 0.1

    ### Driver 1
    tyres_used_driver_1 = []
    for stint in stints_driver_1:
        tyre = laps_driver_1.loc[laps_driver_1['Stint'] == stint, 'Compound'].to_list()[0]
        tyres_used_driver_1.append(tyre)

    segment_colors = [compound_colors[tyre] for tyre in tyres_used_driver_1]

    previous_vuelta = 0
    vueltas = laps_pitin_driver_1

    for i, vuelta in enumerate(vueltas + [max_laps + 1]):  # Añadimos x.max() para el último segmento
        color = segment_colors[i % len(segment_colors)]
        ax2.axvspan(previous_vuelta, vuelta, color=color, alpha=1.0)
        # Dibujar una línea divisoria negra
        if previous_vuelta != 0:  # No dibujar línea antes del primer segmento
            ax2.axvline(x=previous_vuelta, color='black', linewidth=1.0)

        previous_vuelta = vuelta

    ax2.set_xticks(vueltas)
    ax2.set_ylim(rect_y, rect_y + rect_height)
    ax2.set_yticks([])
    ax2.tick_params(axis='x', which='both', bottom=False, top=False)
    ax2.set_ylabel(driver_1, rotation=0, labelpad=20, va='center')

    # Alinear los límites de los ejes x
    ax2.set_xlim(ax1.get_xlim())
    ax2.grid(False)

    ### Driver 2
    tyres_used_driver_2 = []
    for stint in stints_driver_2:
        tyre = laps_driver_2.loc[laps_driver_2['Stint'] == stint, 'Compound'].to_list()[0]
        tyres_used_driver_2.append(tyre)

    segment_colors = [compound_colors[tyre] for tyre in tyres_used_driver_2]

    #previous_vuelta = min(laps_driver_2['LapNumber']) - 0.5
    previous_vuelta = 0
    vueltas = laps_pitin_driver_2

    for i, vuelta in enumerate(vueltas + [max_laps + 1]):  # Añadimos x.max() para el último segmento
        color = segment_colors[i % len(segment_colors)]
        ax3.axvspan(previous_vuelta, vuelta, color=color, alpha=1.0)
        # Dibujar una línea divisoria negra
        if previous_vuelta != 0:  # No dibujar línea antes del primer segmento
            ax3.axvline(x=previous_vuelta, color='black', linewidth=1.0)

        previous_vuelta = vuelta

    ax3.set_xticks(vueltas)
    ax3.tick_params(axis='x', which='both', bottom=False, top=False)
    ax3.set_ylim(rect_y, rect_y + rect_height)
    ax3.set_yticks([])
    ax3.set_ylabel(driver_2, rotation=0, labelpad=20, va='center')
    ax3.grid(False)

    # Alinear los límites de los ejes x
    ax3.set_xlim(ax1.get_xlim())

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


def calcula_pitstopstimes(laps_driver, driver):
    # Calcular vectores de hora de entrada y salida
    times_pitin_driver_first = laps_driver.loc[laps_driver['PitInTime'].notna(), 'PitInTime'].to_list()
    times_pitout_driver_first = laps_driver.loc[laps_driver['PitOutTime'].notna(), 'PitOutTime'].to_list()

    # Determinar el tiempo de pitstop
    text = f"\nTiempos de las Paradas de {driver}"

    for i in range(len(times_pitin_driver_first)):
        aux = times_pitout_driver_first[i] - times_pitin_driver_first[i]
        text += f" Parada {i+1} fue de {aux.total_seconds()}s"

    text += "\n"
    return text


def graph_interval_two_drivers(laps_driver_1, laps_driver_2, driver_1, driver_2, event, save=True):
    # 1) Calcular la diferencia entre pilotos
    ## 1.0) Ver si algun piloto abandono
    total_laps = event.total_laps
    driver1_dnf = check_dnf(laps_driver_1, total_laps)

    if driver1_dnf:
        final_lap1 = max(laps_driver_1['LapNumber']) - 1.0
    else:
        final_lap1 = total_laps

    driver2_dnf = check_dnf(laps_driver_2, total_laps)

    if driver2_dnf:
        final_lap2 = max(laps_driver_2['LapNumber']) - 1.0
    else:
        final_lap2 = total_laps

    if final_lap1 < final_lap2:
        final_lap = final_lap1
    else:
        final_lap = final_lap2

    # Quitar vueltas si uno de los pilotos dió menos de las vueltas totales para comparar
    if final_lap < total_laps:
        laps_driver_1 = laps_driver_1[
            ~laps_driver_1['LapNumber'].between(final_lap, total_laps)]
        laps_driver_2 = laps_driver_2[
            ~laps_driver_2['LapNumber'].between(final_lap, total_laps)]

    # 1.0.1) Ver si alguna vuelta es nula y borrarlas
    mask_isna_1 = laps_driver_1['LapTime'].isna()
    mask_isna_2 = laps_driver_2['LapTime'].isna()

    laps_driver_1 = laps_driver_1.loc[~mask_isna_1,]
    laps_driver_2 = laps_driver_2.loc[~mask_isna_2,]

    ## 1.1) Piloto que llego primero
    laptimes_driver1 = laps_driver_1['LapTime'].to_list()
    laptimes_driver1 = [laptime.total_seconds() for laptime in laptimes_driver1]
    laptimes_driver2 = laps_driver_2['LapTime'].to_list()
    laptimes_driver2 = [laptime.total_seconds() for laptime in laptimes_driver2]

    totaltime_driver1 = sum(laptimes_driver1)
    totaltime_driver2 = sum(laptimes_driver2)

    if totaltime_driver1 < totaltime_driver2:
        # Gano el Piloto 1
        diff_laptimes = [float(t1 - t2) for t1, t2 in zip(laptimes_driver2, laptimes_driver1)]

        driver_first = driver_1
        driver_second = driver_2

        laps_driver_first = laps_driver_1.copy()
        laps_driver_second = laps_driver_2.copy()
        team_driver_first = laps_driver_first['Team'].iloc[0]
        team_driver_second = laps_driver_second['Team'].iloc[0]
        color_driver_first = ff1.plotting.get_team_color(team_driver_first, event)
        color_driver_second = ff1.plotting.get_team_color(team_driver_second, event)

    else:
        # Gano el Piloto2
        diff_laptimes = [float(t1 - t2) for t1, t2 in zip(laptimes_driver1, laptimes_driver2)]

        driver_first = driver_2
        driver_second = driver_1

        laps_driver_first = laps_driver_2.copy()
        laps_driver_second = laps_driver_1.copy()
        team_driver_first = laps_driver_first['Team'].iloc[0]
        team_driver_second = laps_driver_second['Team'].iloc[0]
        color_driver_first = ff1.plotting.get_team_color(team_driver_first, event)
        color_driver_second = ff1.plotting.get_team_color(team_driver_second, event)

    if team_driver_first == team_driver_second:
        # Las vueltas son con el mismo equipo
        linestyle_driver_first = '-'
        alpha_driver_first = 1.0
        linestyle_driver_second = '--'
        alpha_driver_second = 0.5
    else:
        # Las vueltas son con de distinto equipo
        alpha_driver_first = 1.0
        linestyle_driver_first = '-'
        alpha_driver_second = 1.0
        linestyle_driver_second = '-'

    ## 1.2) Crear Dataframe de lapnumber, diff_laptimes y delta
    lapnumbers = laps_driver_1['LapNumber'].to_list()

    d = {'LapNumber': lapnumbers, 'DiffLapTime': diff_laptimes}
    df = pd.DataFrame(d)

    df['Delta'] = df['DiffLapTime'].cumsum()

    ## Calcula pitstop times
    text = calcula_pitstopstimes(laps_driver_first, driver_first)
    text += calcula_pitstopstimes(laps_driver_second, driver_second)

    # 2) PLOT
    ## 2.1) Plot configuration
    plot_title = (f"{event.event.year} {event.event.EventName} {event.name} Distancia en carrera "
                  f"{driver_first} y {driver_second}")
    plot_filename = plot_title.replace(" ", "_") + ".pdf"

    # 2) Setup del grafico
    # Creamos subplots de diferentes tamaños
    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 7), dpi=700)

    ## 2.2) Plot Delta
    ax1.plot(df['LapNumber'], df['Delta'], label=driver_second,
             color=color_driver_second, alpha=alpha_driver_second,
             linewidth=2, linestyle=linestyle_driver_second)

    ## 2.3) Plot linea de referencia en 0
    ax1.axhline(0.0, label=driver_first, color=color_driver_first, alpha=alpha_driver_first,
               linewidth=2, linestyle=linestyle_driver_first)

    ## 2.4) Pitstops
    ### Parametros
    time_max = max(df['Delta'])
    time_min = min(df['Delta'])

    if time_max >= abs(time_min):
        time_max = time_max + 5
        time_min = -time_max
    else:
        time_max = abs(time_min) + 5
        time_min = -time_max

    ### Driver 1
    laps_pitin_driver_first = laps_driver_first.loc[laps_driver_first['PitInTime'].notna(), 'LapNumber'].to_list()
    laps_pitout_driver_first = laps_driver_first.loc[laps_driver_first['PitOutTime'].notna(), 'LapNumber'].to_list()

    for i in range(len(laps_pitin_driver_first)):
        # Pintar franja de parada
        mask_lap1 = laps_driver_first['LapNumber'] >= laps_pitin_driver_first[i]
        mask_lap2 = laps_driver_first['LapNumber'] <= laps_pitout_driver_first[i]
        subset = laps_driver_first.loc[mask_lap1 & mask_lap2,]
        ax1.fill_between(
            subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
            time_min, # Limite inferior del área sombreada en el eje x
            time_max,  # Limite superior del área sombreada en el eje x
            color=color_driver_first, alpha=0.3, label=f'{driver_first} PitStop {i + 1}'
        )

    ## Poner el Gap del piloto second antes de la parada del piloto 1
    text += show_gap_vuelta_by_driver(driver_first, laps_pitin_driver_first, laps_pitout_driver_first, df)

    ### Driver second
    laps_pitin_driver_second = laps_driver_second.loc[laps_driver_second['PitInTime'].notna(), 'LapNumber'].to_list()
    laps_pitout_driver_second = laps_driver_second.loc[laps_driver_second['PitOutTime'].notna(), 'LapNumber'].to_list()

    for i in range(len(laps_pitin_driver_second)):
        # Pintar franja de parada
        mask_lap1 = laps_driver_second['LapNumber'] >= laps_pitin_driver_second[i]
        mask_lap2 = laps_driver_second['LapNumber'] <= laps_pitout_driver_second[i]
        subset = laps_driver_second.loc[mask_lap1 & mask_lap2,]
        ax1.fill_between(
            subset['LapNumber'],  # Usamos los tiempos del DataFrame filtrado
            time_min, # Limite inferior del área sombreada en el eje x
            time_max,  # Limite superior del área sombreada en el eje x
            color=color_driver_second, alpha=0.3, label=f'{driver_second} PitStop {i + 1}'
        )

    ## Poner el Gap del piloto second antes de la parada del piloto 1
    text += show_gap_vuelta_by_driver(driver_second, laps_pitin_driver_second, laps_pitout_driver_second, df)

    ## Poner el Gap final
    text += "\nEl Gap final fue de {:.3f}s en la vuelta {}".format(df.loc[df.index[-1], 'Delta'],
                                                                   df.loc[df.index[-1], 'LapNumber'])
    #text += "\nEl Gap final fue de {:.3f}s".format(df.loc[df.index[-1], 'Delta'])

    # Print text
    print(text)

    ## 2.4) Axis and plot configuration
    max_laps = max(laps_driver_1['LapNumber'])

    ax1.set_title(plot_title)
    #ax1.set_ylabel(f"Delante {driver_first} | Delante {driver_second}")
    ax1.set_ylabel(f"Gap (s)")
    ax1.set_xlabel("Vuelta")
    ax1.set_ylim(time_min, time_max)
    ax1.set_xlim(0.5, max_laps + 0.5)
    ax1.invert_yaxis() # invertir eje y
    ax1.legend()
    ax1.grid(True)

    ## Añadir la imagen logo
    path = os.path.join(os.getcwd(), "src", "outputs", "logo_graficos.png")
    img = mpimg.imread(path)
    imagebox = OffsetImage(img, zoom=0.09, interpolation='bicubic', resample=True)
    ab = AnnotationBbox(imagebox, (0.005, 0.01),
    #ab = AnnotationBbox(imagebox, (0.005, 0.81),  # Coordenadas (x, y) para la esquina inferior izquierda
                        xycoords='axes fraction',  # Coordenadas relativas al área de dibujo
                        box_alignment=(0, 0),  # Alinear la esquina inferior izquierda de la imagen con las coordenadas
                        frameon=False)  # Sin marco alrededor de la imagen
    ax1.add_artist(ab)

    ## 2.4) Plot save
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


def plot_positionchange_driver_withlaps_destacar(drv, driver, session, final_lap_race, df_results, ax):
    drv_laps = session.laps.pick_drivers(drv)
    abb = drv_laps['Driver'].iloc[0]
    color = ff1.plotting.get_driver_color(abb, session)
    dict_status = DICT_STATUS

    initial_position = session.results.loc[session.results['Abbreviation'] == abb, 'GridPosition'].iloc[0]
    if initial_position == 0.0:
        initial_position = 20.0

    laps = [0.0] + drv_laps['LapNumber'].to_list()
    last_lap = drv_laps['LapNumber'].iloc[-1]
    position = [initial_position] + drv_laps['Position'].to_list()

    if last_lap < final_lap_race:
        # No hizo toda la carrera
        finish_position = df_results.loc[df_results["Abbreviation"] == abb, "Position"].iloc[0]
        status = df_results.loc[df_results["Abbreviation"] == abb, "Status"].iloc[0]

        # Añadimos al vector position y laps
        position += [finish_position] * int(final_lap_race - last_lap)
        laps += [i + 1 for i in range(int(last_lap), int(final_lap_race))]

        # Checkear y corregir valores nan de position
        position = [finish_position if pd.isna(pos) else pos for pos in position]

        # Añadir Status final como texto en este caso
        x = final_lap_race + 0.5
        y = finish_position + 0.2
        text = dict_status[status]
        ax.text(x, y, text, fontsize=12, c="yellow")

    # Destacar al piloto and plot
    if abb == driver:
        ax.plot(laps, position, label=abb, color=color, alpha=1.0, linewidth=3)
    else:
        ax.plot(laps, position, label=abb, color=color, alpha=0.5, linewidth=1)


def plot_positionchange_driver_withlaps_nodestacar(drv, df_results, session, final_lap_race, ax):
    dict_status = DICT_STATUS
    drv_laps = session.laps.pick_drivers(drv)
    if drv_laps.empty:
        plot_positionchange_driver_withnolaps(drv, df_results, session, dict_status, final_lap_race, ax)
    else:
        abb = drv_laps['Driver'].iloc[0]
        color = ff1.plotting.get_driver_color(abb, session)

        initial_position = session.results.loc[session.results['Abbreviation'] == abb, 'GridPosition'].iloc[0]
        if initial_position == 0.0:
            initial_position = 20.0

        '''initial_position = position_initial[abb]'''

        laps = [0.0] + drv_laps['LapNumber'].to_list()
        last_lap = drv_laps['LapNumber'].iloc[-1]
        position = [initial_position] + drv_laps['Position'].to_list()

        if last_lap < final_lap_race:
            # No hizo toda la carrera
            finish_position = df_results.loc[df_results["Abbreviation"] == abb, "Position"].iloc[0]
            status = df_results.loc[df_results["Abbreviation"] == abb, "Status"].iloc[0]

            # Añadimos al vector position y laps
            position += [finish_position] * int(final_lap_race - last_lap)
            laps += [i + 1 for i in range(int(last_lap), int(final_lap_race))]

            # Checkear y corregir valores nan de position
            position = [finish_position if pd.isna(pos) else pos for pos in position]

            # Añadir Status final como texto en este caso
            if status != "+1 Lap":
                x = final_lap_race + 0.5
                y = finish_position + 0.2
                text = dict_status[status]
                ax.text(x, y, text, fontsize=12, c="yellow")

            # Plotear al piloto
            ax.plot(laps, position, label=abb, color=color, alpha=1.0, linewidth=1)

        else:
            # Check final status
            final_status = df_results.loc[df_results["Abbreviation"] == abb, "Status"].iloc[0]

            # Add as last position finish position
            finish_position = df_results.loc[df_results["Abbreviation"] == abb, "Position"].iloc[0]
            position[-1] = finish_position
            
            if final_status == "Finished":
                ax.plot(laps, position, label=abb, color=color, alpha=1.0, linewidth=1)
            else:
                if final_status != "+1 Lap":
                    x = final_lap_race + 0.5
                    y = finish_position + 0.2
                    print(f"abb={abb}, final_status = {final_status}, final_lap_race={final_lap_race}, finish_position={finish_position}")
                    text = dict_status[final_status]
                    ax.text(x, y, text, fontsize=12, c="yellow")
                ax.plot(laps, position, label=abb, color=color, alpha=1.0, linewidth=1)
            

def plot_positionchange_driver_withnolaps(drv, df_results, session, final_lap_race, ax):
    dict_status = DICT_STATUS
    initial_position = df_results.loc[df_results['DriverNumber'] == drv, 'GridPosition'].iloc[0]
    finish_position = df_results.loc[df_results['DriverNumber'] == drv, "Position"].iloc[0]
    status = df_results.loc[df_results['DriverNumber'] == drv, "Status"].iloc[0]
    abb = df_results.loc[df_results['DriverNumber'] == drv, "Abbreviation"].iloc[0]
    color = ff1.plotting.get_driver_color(abb, session)

    laps = [i for i in range(0, int(final_lap_race) + 1)]
    position = [initial_position] + [finish_position] * int(final_lap_race)

    # Añadir Status final como texto en este caso
    x = final_lap_race + 0.5
    y = finish_position + 0.2
    text = dict_status[status]
    ax.text(x, y, text, fontsize=12, c="yellow")

    # Plot
    ax.plot(laps, position, label=abb, color=color, alpha=0.5, linewidth=1)


def plot_positionchanges_race(session, df_logs, destacar, driver=None, save=True):
    # Obtener la ultima vuelta y df de resultados
    final_lap_race = obtain_last_lap(event=session)
    df_results = session.results[['Position', 'Abbreviation', 'FullName', 'Status', 'TeamName', 'CountryCode', 'Points', 'DriverNumber', 'GridPosition']]
    dict_status = {
        "Accident": "Accidente",
        "Withdrew": "No comenzó",
        "Disqualified": "Descalificado",
        "+1 Lap": "+1 Vuelta",
        "Radiator": "Radiador",
        "Engine": "Motor",
        "Abandon": "Abandono",
        "Spun off": "Trompeo",
        "Collision damage": "Daños por Colisión",
        "Collision": "Colisión",
        "Retired": "Retirado",
        "Did_not_start": "No Comenzo",
        "Lapped": "Doblado"
    }

    '''position = [4.0, 1.0, 10.0, 3.0, 2.0, 12.0, 6.0, 7.0, 17.0, 9.0, 5.0, 8.0, 11.0, 13.0, 18.0, 15.0, 14.0, 16.0, 20.0, 19.0]
    points = [12.0, 25.0, 2.0, 15.0, 18.0, 0.0, 8.0, 6.0, 0.0, 2.0, 10.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    status = [None, None, None, None, None, None, None, None, "Abandon", None, None, None, None, None, "Abandon", "+1 Lap", None, "Abandon", "Accident", "Accident"]
    df_results['Position'] = position
    df_results['Points'] = points
    df_results['Status'] = status'''
    try:
        # PLOT
        fig, ax = plt.subplots(figsize=(12, 7), dpi=700)

        # Plot drivers

        # Buscar pilotos que no han dado vueltas
        list_nolapdrivers = find_drivers_noLap_warning(df_logs)

        # Check len of list_nolapdrivers
        if len(list_nolapdrivers) > 0:
            if destacar:
                for drv in session.drivers:
                    if drv not in list_nolapdrivers:
                        plot_positionchange_driver_withlaps_destacar(drv, driver, session, final_lap_race, df_results, ax)
                    else:
                        plot_positionchange_driver_withnolaps(drv, df_results, session, final_lap_race, ax)
            else:
                for drv in session.drivers:
                    if drv not in list_nolapdrivers:
                        plot_positionchange_driver_withlaps_nodestacar(drv, df_results, session, final_lap_race, ax)
        else:
            if destacar:
                for drv in session.drivers:
                    plot_positionchange_driver_withlaps_destacar(drv, driver, session, final_lap_race, df_results, ax)
            else:
                for drv in session.drivers:
                    plot_positionchange_driver_withlaps_nodestacar(drv, df_results, session, final_lap_race, ax)

        # Plot sttings
        plot_title = (f"{session.event.year} {session.event.EventName} {session.name} Cambios de Posición")
        positions = [i+1 for i in range(20)]
        ax.set_title(plot_title)
        ax.set_ylim([20.5, 0.5])
        ax.set_xlim([0, final_lap_race])

        ax.set_yticks(positions)
        ax.tick_params(axis='y', which='minor', left=False)

        ax.grid(axis='y', which='minor', visible=False)
        ax.set_xlabel('Vuelta')
        ax.set_ylabel('Posicion')

        ax.legend(bbox_to_anchor=(1.0, 1.03))
        plt.tight_layout()

        ## TODO Añadir la imagen logo - NO QUEDA FUERA
        '''path = os.path.join("outputs", "logo_graficos.png")
        img = mpimg.imread(path)
        imagebox = OffsetImage(img, zoom=0.062, interpolation='bicubic', resample=True)
        ab = AnnotationBbox(imagebox, (1.3, 0.01),  # Coordenadas (x, y) para la esquina inferior izquierda
                            xycoords='axes fraction',  # Coordenadas relativas al área de dibujo
                            box_alignment=(1, 0),  # Alinear la esquina inferior izquierda de la imagen con las coordenadas
                            frameon=False)  # Sin marco alrededor de la imagen
        ax.add_artist(ab)'''

        # Save
        if save:
            # Crear subdirectorio con el nombre del pais_año para guardar los outputs
            eventname = session.event.EventName.split(" ")[0]
            path = f"{session.event.year}_{eventname}"
            path = os.path.join(os.getcwd(), "src", "outputs", path)

            # Crear el subdirectorio de GP
            if not os.path.exists(path):
                os.makedirs(path)

            # Crear el subdirectorio del nombre del evento
            path = os.path.join(path, session.name)
            if not os.path.exists(path):
                os.makedirs(path)

            # Guardar figura
            if destacar:
                plot_filename = path + f"/Position_changes_destacar_{driver}.pdf"
                plt.savefig(plot_filename, format='pdf')
            else:
                plot_filename = path + "/Position_changes.pdf"
                plt.savefig(plot_filename, format='pdf')
        else:
            plt.show()

    except Exception as e:
        raise e
    except sum(pd.isna(df_results['Status'])) == df_results.shape[0]:
        raise Exception("Error: Los resultados No han sido actualizados en FastAPI, añadirlos manualmente")
IOC_TO_ISO3 = {
    "MON": "MCO",
    "NED": "NLD",
    "SUI": "CHE",
}

def get_flag_image(country_code3):
    """Descarga la bandera a partir del código ISO Alpha-3."""

    country_code = IOC_TO_ISO3.get(country_code3.upper(), country_code3.upper())
    country = pycountry.countries.get(alpha_3=country_code)

    if country is None:
        raise ValueError(f"Código de país inválido: {country_code3}")

    country_code2 = country.alpha_2.lower()

    url = f"https://flagcdn.com/w40/{country_code2}.png"

    response = requests.get(url)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content)).convert("RGBA")
    img = img.resize((40, 26), Image.LANCZOS)

    return img


def decide_linestyles(team_to_drivers, df_historic):
    linestyles = {}
    for team, drivers in team_to_drivers.items():
        if len(drivers) == 1:
            linestyles[drivers[0]] = 'solid'
        elif len(drivers) == 2:
            d1, d2 = drivers
            p1 = df_historic[d1].iloc[-1]
            p2 = df_historic[d2].iloc[-1]
            if p1 >= p2:
                linestyles[d1] = 'solid'
                linestyles[d2] = '--'
            else:
                linestyles[d1] = '--'
                linestyles[d2] = 'solid'
        else:
            for d in drivers:
                linestyles[d] = 'solid'  # fallback
    return linestyles


def make_plot_mundial(df_historic, session, mundial_type, save=True):
    try:
        # Extract important data
        x = df_historic['Carrera'].to_list()
        nraces = df_historic.shape[0] -1

        # Encontrar el orden de puntos en base a la ultima fila 
        driver_team = session.results[['Abbreviation', 'TeamName']].set_index('Abbreviation').to_dict()['TeamName']
        values = get_list_sorted(df_historic)

        # Encontrar mapa de pilotos equipo
        team_to_drivers = defaultdict(list)
        for driver in values:
            team = driver_team.get(driver, "Unknown")
            team_to_drivers[team].append(driver)

        # Decidir los linestyles
        linestyles = decide_linestyles(team_to_drivers, df_historic)

        # PLOT
        fig, ax = plt.subplots(figsize=(12, 7), dpi=700)

        for value in values:
            if mundial_type == "Pilotos":
                try:
                    color = ff1.plotting.get_driver_color(value, session)
                except KeyError:
                    color = "black"
            elif mundial_type == "Equipos":
                color = ff1.plotting.get_team_color(value, session)
            else:
                raise Exception(f"Incorrect mundial_type: {mundial_type}")
            
            label = f"{value} - {df_historic[value].iloc[-1]}"
            style = linestyles.get(value, 'solid')

            ax.plot(x, df_historic[value], label=label, color=color, alpha=1.0, linewidth=3, linestyle=style)
            ax.plot(x, df_historic[value], "o", color=color, alpha=1.0, linewidth=3)

        # Plot Settings
        points_max = df_historic.loc[df_historic.index[-1],].max()
        plot_title = f"Mundial de {mundial_type}"
        ax.set_title(plot_title, fontsize=24, fontweight='bold')
        ax.set_xlim([0, nraces+0.05])
        ax.set_ylim(bottom=-10)
        ax.set_ylim([-0.025*points_max, points_max+5.0])

        keys = [i+1 for i in range(nraces)]

        # Colocar banderas en los ticks del eje X
        for i, nombre in enumerate(keys):
            if nombre not in DICT_RACE_COUNTRYCODE:
                continue
            codigo = DICT_RACE_COUNTRYCODE[nombre]['countrycode'].lower()
            bandera = get_flag_image(codigo)
            imagebox = OffsetImage(bandera, zoom=0.5)
            ab = AnnotationBbox(imagebox, (i + 1, -3), frameon=False, box_alignment=(0.5, 1))
            ax.add_artist(ab)

        # Añadir logo
        if mundial_type == "Pilotos":
            x_logo = 0.38
            y_logo = 0.77
            zoom = 0.09
        elif mundial_type == "Equipos":
            x_logo = 0.4
            y_logo = 0.77
            zoom = 0.09
        else:
            raise Exception(f"Incorrect mundial_type: {mundial_type}")

        add_team_logo(ax=ax, x=x_logo, y=y_logo, zoom=zoom)

        # Limpiar el eje X (sin texto debajo)
        ax.set_xticks(range(len(keys)+1))
        ax.set_xticklabels([""] * (len(keys)+1))
        ax.tick_params(axis='y', which='minor', left=False)
        ax.tick_params(axis='x', which='both', length=0)
        ax.grid(axis='y', which='minor', visible=False)
        # ax.set_xlabel('Carrera', fontsize=18)
        ax.set_ylabel('Puntos', fontsize=18)
        ax.tick_params(axis='y', which='major', labelsize=15) 
        ax.legend(bbox_to_anchor=(1.0, 1.02))

        plt.tight_layout()

        # Save
        if save:
            # Crear subdirectorio con el nombre del pais_año para guardar los outputs
            eventname = session.event.EventName.split(" ")[0]
            path = f"{session.event.year}_{eventname}"
            path = os.path.join(os.getcwd(), "src", "outputs", path)

            # Crear el subdirectorio de GP
            if not os.path.exists(path):
                os.makedirs(path)

            # Crear el subdirectorio del nombre del evento
            path = os.path.join(path, session.name)
            if not os.path.exists(path):
                os.makedirs(path)

            # Guardar figura en pdf 
            nombre = f"mundial_{mundial_type}_historic"
            nombre_pdf = nombre + ".pdf"
            nombre_png = nombre + ".png"

            plot_filename = os.path.join(path, nombre_pdf)
            plt.savefig(plot_filename, format='pdf')
            
            ## Convert to png
            input_path = path
            path_imagenes = input_path
            convert_and_save_in_png(input_path, path_imagenes, nombre)

            ## Upload to S3
            filepath = os.path.join(path, nombre_png)
            s3path = "imagenes/historico"
            upload_file_s3(filepath, s3path)

        else:
            plt.show()

    except Exception as e:
        raise e


def put_flags_in_xaxis(keys, ax, delta):
    for i, nombre in enumerate(keys):
        if nombre not in DICT_RACE_COUNTRYCODE:
            continue
        codigo = DICT_RACE_COUNTRYCODE[nombre]['countrycode'].lower()
        bandera = get_flag_image(codigo)
        imagebox = OffsetImage(bandera, zoom=0.5)
        ab = AnnotationBbox(imagebox, (i + 1, delta), frameon=False, box_alignment=(0.5, 1))
        ax.add_artist(ab)


def make_plot_posiciones(df_historic_posiciones_pilotos, session, save=True):
    # Extract drivers list
    drivers_list = df_historic_posiciones_pilotos.columns.to_list()
    drivers_list = [driver for driver in drivers_list if driver != "Carrera"]
    carreras_list = df_historic_posiciones_pilotos['Carrera'].to_list()

    # Races
    nraces = df_historic_posiciones_pilotos.shape[0]
    keys = [i+1 for i in range(nraces)]

    # Part of cofiguration of save graphics
    # Crear subdirectorio con el nombre del pais_año para guardar los outputs
    eventname = session.event.EventName.split(" ")[0]
    path = f"{session.event.year}_{eventname}"
    path = os.path.join(os.getcwd(), "src", "outputs", path)

    # Crear el subdirectorio de GP
    if not os.path.exists(path):
        os.makedirs(path)

    # Crear el subdirectorio del nombre del evento
    path = os.path.join(path, session.name)
    if not os.path.exists(path):
        os.makedirs(path)

    for driver in drivers_list:
        # PLOT
        fig, ax = plt.subplots(figsize=(10, 10), dpi=700)
        plt.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.05, hspace=0.3)
        if driver == "DOO":
            color = ff1.plotting.get_driver_color('GAS', session)
        else:
            color = ff1.plotting.get_driver_color(driver, session)
        plot_title = f"Histórico Posiciones {driver}"

        # Principal line
        ax.plot(carreras_list, df_historic_posiciones_pilotos[driver], color=color, linewidth=3, linestyle='solid')

        # Plot points
        for x, y in zip(carreras_list, df_historic_posiciones_pilotos[driver]):
            # Circulo como marker
            ax.scatter(x, y, s=1500, color='#1E1C1C', zorder=3)

            # Definir color posicion
            if y == 1:
                color_posicion = 'gold'
                weight_posicion = 'bold'
            elif y == 2:
                color_posicion = 'darkgray'
                weight_posicion = 'bold'
            elif y == 3:
                color_posicion = '#B87333'
                weight_posicion = 'bold'
            else:
                color_posicion = 'white'
                weight_posicion = 'normal'

            # Texto centrado encima del círculo
            ax.text(
                x, y, f"{y}º",
                ha='center', va='center',
                fontsize=20,
                fontweight=weight_posicion,
                color=color_posicion,
                zorder=4
            )

        # Title and axis
        ax.set_title(plot_title, fontsize=35, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("Posición", fontsize=25)
        ax.set_xticks(keys)
        min_value = min(df_historic_posiciones_pilotos[driver].dropna())
        max_value = max(df_historic_posiciones_pilotos[driver].dropna())
        # print(driver, min_value, max_value)

        ax.set_yticks(range(int(min_value), int(max_value)+2))
        ax.yaxis.set_inverted(True) # Invertir eje y
        ax.grid(which='minor', visible=False)
        ax.grid(which='major', linestyle='--', linewidth=0.4, color='lightgray', alpha=0.3)

        # Borrar algunos bordes para estilo más moderno
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='y', which='major', labelsize=24, colors='white')
        ax.tick_params(axis='y', which='minor', left=False, right=False)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # Colocar posiciones encima o debajo de los valores
        ax.set_xticklabels([""] * (len(keys)))
        delta = ax.get_ylim()[0]
        put_flags_in_xaxis(keys, ax, delta) # Añadir banderas en eje x

        # Save
        plot_filename = os.path.join(path, f"historico_posiciones_{driver}.png")
        plt.savefig(plot_filename, format='png')

        ## Upload to S3
        s3path = "imagenes/historico_pilotos"
        upload_file_s3(plot_filename, s3path)

        print(f"Plotted {driver}")


def graph_mundial_pilotos(session, df_mundial, save=False):
    # PLOT
    fig, ax = plt.subplots(figsize=(12, 7), dpi=700)

    # Modificar tabla
    df_mundial_copy = df_mundial[['posicion', 'piloto', 'equipo', 'puntos']].copy()

    # Añadir tabla
    table_data = df_mundial_copy.values.tolist()
    #column_labels = df_mundial_copy.columns.tolist()
    column_labels = ['Posicion', 'Piloto', 'Equipo', 'Puntos']

    row_colors = ['#2E2E2E', '#1E1C1C']
    table = ax.table(cellText=table_data, colLabels=column_labels,
                      cellLoc='center', loc='right',
                      colColours=["#1E1C1C"] * len(column_labels),  # Fondo de la cabecera
                      cellColours=[[row_colors[i % len(row_colors)] for _ in range(len(column_labels))]
                                   for i in range(len(table_data))],  # Fondo alternado de las celdas
                      # cellColours=[["#1E1C1C"] * len(column_labels)] * len(table_data),  # Fondo de las celdas
                     bbox=[0.0, 0.0, 1.0, 1.0])  # Posición y tamaño de la tabla [xmin, ymin, width, height]

    # Estilo de la tabla
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1.5, 1.5)

    # Eliminar grafico del fondo
    ax.axis('off')

    # Cambiar el color del texto a blanco y eliminar bordes
    for key, cell in table.get_celld().items():
        cell.set_text_props(color="white")
        cell.set_edgecolor('none')  # Eliminar bordes

    # Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        eventname = session.event.EventName.split(" ")[0]
        path = f"{session.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        # Crear el subdirectorio de GP
        if not os.path.exists(path):
            os.makedirs(path)

        # Crear el subdirectorio del nombre del evento
        path = os.path.join(path, session.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar figura
        plot_filename = path + "/Mundial_Pilotos.pdf"
        plt.savefig(plot_filename, format='pdf')
    else:
        plt.show()


def graph_mundial_equipos(session, df_mundial, save=False):
    # PLOT
    fig, ax = plt.subplots(figsize=(12, 7), dpi=700)

    # Modificar tabla
    df_mundial_copy = df_mundial.copy()

    # Añadir tabla
    table_data = df_mundial_copy.values.tolist()
    #column_labels = df_mundial_copy.columns.tolist()
    column_labels = ['Posicion', 'Equipo', 'Puntos']
    row_colors = ['#2E2E2E', '#1E1C1C']
    table = ax.table(cellText=table_data, colLabels=column_labels,
                     cellLoc='center', loc='right',
                     colColours=["#1E1C1C"] * len(column_labels),  # Fondo de la cabecera
                     cellColours=[[row_colors[i % len(row_colors)] for _ in range(len(column_labels))]
                                  for i in range(len(table_data))],  # Fondo alternado de las celdas
                     # cellColours=[["#1E1C1C"] * len(column_labels)] * len(table_data),  # Fondo de las celdas
                     bbox=[0.0, 0.0, 1.0, 1.0])  # Posición y tamaño de la tabla [xmin, ymin, width, height]
    # Estilo de la tabla
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(2.0, 2.0)

    # Añadir imágenes en la columna de 'Equipo'
    list_teams = df_mundial_copy['equipo'].to_list()
    list_teams = [team if team != 'Mercedes' else 'mercedes' for team in list_teams ]
    dict_zooms = {
        "Mclaren": 0.12,
        "mercedes": 0.025,
        "RedBull": 0.05,
        "Alpine": 0.03,
        "Williams": 0.01,
        "Ferrari": 0.02,
        "RB": 0.02,
        "AstonMartin": 0.06,
        "Haas": 0.03,
        "KickSauber": 0.03,
    }

    for i, team in enumerate(list_teams):
        img_path = os.path.join("outputs", f'logo_{team}.png')
        image = plt.imread(img_path)
        imagebox = OffsetImage(image, zoom=dict_zooms[team])
        ab = AnnotationBbox(imagebox, (0.3, 0.86 - i*0.09), frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)

    # Cambiar el color del texto a blanco y eliminar bordes
    for key, cell in table.get_celld().items():
        cell.set_text_props(color="white")
        cell.set_edgecolor('none')  # Eliminar bordes

    # Eliminar grafico del fondo
    ax.axis('off')

    # Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        eventname = session.event.EventName.split(" ")[0]
        path = f"{session.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        # Crear el subdirectorio de GP
        if not os.path.exists(path):
            os.makedirs(path)

        # Crear el subdirectorio del nombre del evento
        path = os.path.join(path, session.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar figura
        plot_filename = path + "/Mundial_Equipos.pdf"
        plt.savefig(plot_filename, format='pdf')
    else:
        plt.show()


def generate_recta_ajuste(x, y, text):
    # Convert to array bth and reshape x to (-1,1)
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    # Calcular los parametros de la recta
    regressor = linear_model.LinearRegression()
    regressor.fit(x, y)
    pendiente = regressor.coef_[0]
    termino_independiente = regressor.intercept_
    coeficientes = [pendiente, termino_independiente]
    text += f"\tRecta Ajuste = {pendiente} * x + {termino_independiente}\n"

    # Predecir los valores ajustados
    vueltas_ajustadas = regressor.predict(x)

    # Calcula la bondad del ajuste
    score = regressor.score(x,y)
    text += "\tScore = {:.2f}\n".format(score)

    return vueltas_ajustadas, coeficientes, text


def calcula_degradacion(session, coeficientes, text):
    '''
        Función que calcula la degradacion por neumaticos restando la
        ganancia que da por quema de gasolina
        Calculo:
        --------------------------------------------------------------------------------------
        Capacidad deposito F1 = 110kg (145L)
        Distancia Carrera (Austin) = 308,45km
        Nº de vueltas en Austin = 56 vueltas
        Ganancia en tiempo por kilo = 0,03 s/kg o 3 decimas por 10Kg

        kilos por km gastados de gasolina = 110kg/308,45km = 0,357 kg/km
        km/vuelta = 308,45Km/56vueltas = 5,513 km/vuelta

        Los kilos por vuelta =  0,357 kg/km * 5,513 km/vuelta = 1,966 kg/vuelta ~ 2kg/vuelta

        Ganancia segundos por vuelta = 0,03 s/kg * 2 kg/vuelta = 0,06 s/vuelta

        Luego, Degradacion medida (pendiente) = Degradacion_neumatico - Ganancia_combustible <=>
        <=> Degradacion_neumatico = Degradacion medida (pendiente) + Ganancia_combustible
        --------------------------------------------------------------------------------------

        :param session: objecto session de fastf1
        :param coeficientes [list]: con la pendiente y termino independiente
        :param text (str): string para agregar texto al resultado de ajuste
        :return:  degradacion_neumatico, text
    '''
    # Estimar la ganancia por vuelta de combustible en este circuito (Lo estime para Austin gastando all fuel)
    consumo_kg_vuelta = 2
    ganancia_seg_vuelta = 0.06

    # Calcualar degradacion de neumaticos
    degradacion_neumatico = coeficientes[0] + ganancia_seg_vuelta
    text += "\tDegradacion = {:.3f} s/vuelta\n\n".format(degradacion_neumatico)

    return degradacion_neumatico, text


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

def graph_ritmos_por_equipos(session, data, stint, save=True):
    # Constantes y variables del metodo
    top_teams = ["Aston Martin", "Ferrari", "Red Bull Racing", "Mercedes", "McLaren"]
    top_teams = top_teams + ["Williams"]
    option = "TopTeams"

    dict_teams = {
        'Alpine': 'Alpine',
        'Aston Martin': 'AstonMartin',
        'Ferrari': 'Ferrari',
        'Haas F1 Team': 'Haas',
        'Kick Sauber': 'KickSauber',
        'McLaren': 'Mclaren',
        'Mercedes': 'Mercedes',
        'RB': 'RB',
        'Red Bull Racing': 'RedBull',
        'Williams': 'Williams'
    }

    # Definir grafico
    fig, ax = plt.subplots(1, 2, figsize=(12, 7), dpi=700)

    # 1) Extraer ritmos por equipo y plotear
    info_equipos = data['info_equipos']
    df_info_equipos = pd.DataFrame(info_equipos)
    df_info_equipos = df_info_equipos.sort_values(by="Ritmo")
    colors = []
    list_laps_topteams = []
    list_teams_order = []
    list_degradacion = []
    text = ""
    if option == "TopTeams":
        # Selecciono solo las filas de equipos Top
        mask_topteams = df_info_equipos["Team"].isin(top_teams)
        df_info_equipos = df_info_equipos.loc[mask_topteams,]

        for i in range(df_info_equipos.shape[0]):
            # Valores
            team = df_info_equipos['Team'].iloc[i]
            team = dict_teams[team]
            driver = df_info_equipos['Driver'].iloc[i]
            laps = df_info_equipos['Laps'].iloc[i]
            list_lapnumber = [j + 1 for j in range(len(laps))]

            color = ff1.plotting.get_team_color(team, session)

            # Calcular recta de ajuste lineal
            text += f"Equipo = {team}\n"
            x = list_lapnumber
            y = laps

            y_ajustada, coeficientes, text = generate_recta_ajuste(x=x, y=y, text=text)
            text += f"\tTiempo Min = {min(y)}, Tiempo Max = {max(y)}\n"

            # Calcular al degradacion de neumaticos
            degradacion, text = calcula_degradacion(session, coeficientes, text)

            # Grafica de tiempos
            ax[0].plot(x, y, "o", label=f"{team} ({driver})", color=color, alpha=0.3)
            ax[0].plot(x, y_ajustada, "-", label="", color=color)

            # Append and Save degradacion
            list_degradacion.append(degradacion)
            list_laps_topteams.append(y)
            colors.append(color)
            list_teams_order.append(team)

    # TODO PONER COMO EN EL CASO TOP TEAM
    else:
        for i in range(df_info_equipos.shape[0]):
            # Valores
            team = df_info_equipos['Team'].iloc[i]
            team = dict_teams[team]
            driver = df_info_equipos['Driver'].iloc[i]
            laps = df_info_equipos['Laps'].iloc[i]
            list_lapnumber = [j + 1 for j in range(len(laps))]
            color = ff1.plotting.get_team_color(team, session)

            # Grafica de tiempos
            ax[0].plot(list_lapnumber[1:], laps[1:], "o", label=f"{team} ({driver})", color=color)

            # Append
            list_laps_topteams.append(laps[1:])
            colors.append(color)
            list_teams_order.append(team)

    # Ejes
    ## Calcular el tiempo minimo y maximo
    min_time = min([min(laps) for laps in list_laps_topteams])
    max_time = max([max(laps) for laps in list_laps_topteams])

    ## Definir ejes
    ax[0].set_xlabel("Vuelta del Stint", fontsize=14)
    ax[0].set_ylabel("Tiempo de Vuelta (s)", fontsize=14)
    ax[0].set_title("Evolución de los Ritmos")
    ax[0].tick_params(axis='y', which='major', labelsize=12, colors='white')
    ax[0].tick_params(axis='x', which='major', labelsize=12, colors='white')

    y_min = 0.99* min_time
    y_max = max_time

    ax[0].set_ylim(y_min, y_max)
    #ax[0].set_ylim(min_time - 2.0 * delta_time_min, 100.0)

    ax[0].legend(loc='upper right', ncols=2, fontsize=12)

    # Añadir logo
    x_logo = 0.58
    y_logo = 0.01
    zoom = 0.06
    add_team_logo(ax=ax[0], x=x_logo, y=y_logo, zoom=zoom)

    # 2) BoxPlot TOP
    bplot = ax[1].boxplot(list_laps_topteams,
                          patch_artist=True,
                          labels=list_teams_order,
                          medianprops={"color": "white", "linewidth": 1.0},
                          whiskerprops={"color": "white", "linewidth": 1.0},
                          boxprops={"color": "white", "linewidth": 1.0},
                          flierprops={
                            "marker": "o",  # Forma de los fliers, puedes cambiarla si deseas
                            "markeredgecolor": "white",  # Color del borde de los fliers
                            "markersize": 4  # Tamaño de los fliers, ajusta si es necesario
                          },
                          capprops={"color": "white", "linewidth": 1.0})

    # Dar color al boxplot
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    # Ejes
    # ax[1].legend(loc='upper right', ncols=3, fontsize=10)
    ax[1].set_ylabel("Tiempo de Vuelta (s)", fontsize=14)
    ax[1].set_xlabel("")
    ax[1].set_title("Boxplot de los ritmos")
    ax[1].set_ylim(y_min, y_max)
    ax[1].tick_params(axis='y', which='major', labelsize=12, colors='white')
    ax[1].tick_params(axis='x', which='major', labelsize=12, colors='white')

    # Añadir logo
    x_logo = 0.58
    y_logo = 0.01
    zoom = 0.06
    add_team_logo(ax=ax[1], x=x_logo, y=y_logo, zoom=zoom)

    # Mostrar resultados
    print(text)

    df_info_equipos['Degradacion'] = list_degradacion
    df_info_equipos = df_info_equipos[['Team', 'Ritmo', 'Degradacion']]
    df_info_equipos['Ritmo'] = df_info_equipos['Ritmo'].apply(lambda i: round(i, 3))
    df_info_equipos['Degradacion'] = df_info_equipos['Degradacion'].apply(lambda i: round(i, 3))

    print(tabulate(df_info_equipos, headers='keys', showindex=False, tablefmt='grid'))

    # 3) Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        eventname = session.event.EventName.split(" ")[0]
        path = f"{session.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        # Crear el subdirectorio de GP
        if not os.path.exists(path):
            os.makedirs(path)

        # Crear el subdirectorio del nombre del evento
        path = os.path.join(path, session.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar figura
        plot_filename = path + f"/Ritmos_Equipos_Stint{stint}.pdf"
        plt.savefig(plot_filename, format='pdf', bbox_inches='tight', pad_inches=0.1)

        # Guardar text
        pathtext = f"Ritmos_degradacion_Equipos_Stint{stint}.txt"
        f = open(os.path.join(path, pathtext), "w")
        f.write(text)
        f.close()
    else:
        plt.show()


def compare_lift_straight(laps_driver_1, laps_driver_2, session, target_lap, drivers, save=True):
    # Extract telemetry from lap in both drivers
    fastest_driver_1 = laps_driver_1[laps_driver_1['LapNumber'] == target_lap].iloc[0]
    fastest_driver_2 = laps_driver_2[laps_driver_2['LapNumber'] == target_lap].iloc[0]
    telemetry_driver_1 = fastest_driver_1.get_telemetry().add_distance()
    telemetry_driver_2 = fastest_driver_2.get_telemetry().add_distance()

    # PLOT - subplot off two axis
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), dpi=700)
    plot_title = "Levantada en la Recta Principal"

    # Subset in main straight by distance
    subset_driver1 = telemetry_driver_1.loc[telemetry_driver_1['Distance'] <= 800,]
    subset_driver2 = telemetry_driver_2.loc[telemetry_driver_2['Distance'] <= 800,]
    team1 = laps_driver_1['Team'].to_list()[0]
    team2 = laps_driver_2['Team'].to_list()[0]
    teams = [team1, team2]

    # ax1 - plot Speed
    colors = [ff1.plotting.get_team_color(team, session) for team in teams]
    ax[0].plot(subset_driver1['Distance'], subset_driver1['Speed'], label=drivers[0],
               color=colors[0], alpha=1.0, linewidth=2, linestyle="-")
    ax[0].plot(subset_driver2['Distance'], subset_driver2['Speed'], label=drivers[1],
               color=colors[1], alpha=1.0, linewidth=2, linestyle="-")

    ## Axis and titles
    ax[0].set_ylabel('Speed (km/h)', fontsize=12)
    ax[0].set_xlim(0, 800)
    ax[0].legend(loc="upper right")
    ax[0].set_title(plot_title, fontdict={'fontsize': 20, 'fontweight': 'bold'})

    ## Circle in the lift part
    mask_lift = subset_driver1['Throttle'] <= 99.0
    mask_limit = subset_driver1['Distance'] <= 500
    subset_lift = subset_driver1.loc[mask_lift & mask_limit,]
    list_distances = subset_lift["Distance"].to_list()
    x_min = min(list_distances)
    x_max = max(list_distances)
    mask1 = subset_driver2['Distance'] >= x_min
    mask2 = subset_driver2['Distance']<= x_max
    list_other = subset_driver2.loc[mask1&mask2, "Speed"].to_list()

    y_max = max(list_other) + 5.0
    y_min = min(list_other) - 5.0
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = 2 * (x_max - x_min)
    height = 2 * (y_max - y_min)

    elipse = Ellipse(
        (center_x, center_y),
        width,
        height,
        color='white',
        alpha=1.0,
        linestyle='--',
        fill=False
    )
    ax[0].add_patch(elipse)

    # ax2 - plot Throttle
    ax[1].plot(subset_driver1['Distance'], subset_driver1['Throttle'], label=drivers[0],
               color=colors[0], alpha=1.0, linewidth=2, linestyle="-")
    ax[1].plot(subset_driver2['Distance'], subset_driver2['Throttle'], label=drivers[1],
               color=colors[1], alpha=1.0, linewidth=2, linestyle="-")

    ## Circle in the lift part
    mask_lift = subset_driver1['Throttle'] <= 99.0
    mask_limit = subset_driver1['Distance'] <= 500
    subset_lift = subset_driver1.loc[mask_lift&mask_limit,]
    list_distances = subset_lift["Distance"].to_list()
    list_throttles = subset_lift["Throttle"].to_list()

    x_min = min(list_distances)
    x_max = max(list_distances)
    y_max = 102.0
    y_min = min(list_throttles) - 2.0
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = 2*(x_max - x_min)
    height = 2*(y_max - y_min)

    elipse = Ellipse(
        (center_x, center_y),
        width,
        height,
        color='white',
        alpha=1.0,
        linestyle='--',
        fill=False
    )
    ax[1].add_patch(elipse)

    ## Axis and titles
    ax[1].set_ylabel('Throttle (%)', fontsize=12)
    ax[1].set_xlim(0, 800)

    # Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        eventname = session.event.EventName.split(" ")[0]
        path = f"{session.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        # Crear el subdirectorio de GP
        if not os.path.exists(path):
            os.makedirs(path)

        # Crear el subdirectorio del nombre del evento
        path = os.path.join(path, session.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar figura
        plot_filename = path + "/Levantada_RectaPrincipal.pdf"
        plt.savefig(plot_filename, format='pdf')
    else:
        plt.show()

def get_reaction_times(df: pd.DataFrame) -> dict:
    """
    Calcula los tiempos de aceleración 0-100 km/h y 0-200 km/h desde los datos de telemetría.
    """

    # Asegúrate de que 'Speed' esté en km/h y 'SessionTime' esté en segundos
    df = df.copy()

    # ReactionLightsTime
    t_reaction = df.loc[df['Speed'] > 0.0, 'Time'].iloc[0].total_seconds() 

    # Tiempo de Reaccion 0-100km/h 
    df['Delta_100'] = df['Speed'].apply(lambda i: abs(i-100.0))
    t_0_100 = df.loc[df['Delta_100'].idxmin(), 'Time'].total_seconds() 

    # Encuentra el tiempo cuando alcanza 200 km/h
    df['Delta_200'] = df['Speed'].apply(lambda i: abs(i-200.0))
    t_0_200 = df.loc[df['Delta_200'].idxmin(), 'Time'].total_seconds() 

    # return {
    #     "time0-100": round(t_100, 3),
    #     "time0-200": round(t_200, 3),
    #     "time100-200": round(t_200 - t_100, 3)
    # }
    return {
        "t_reaction": t_reaction,
        "time0-100": t_0_100,
        "time0-200": t_0_200,
        "time100-200": t_0_200 - t_0_100
    }

def plot_race_start(race, driver1, driver2, tiktok_format, save):
    # Extract data from the times of acceleration 
    race_laps = race.laps
    drivers = np.unique(race_laps['Driver'])
    drivers = [driver for driver in drivers if driver != 'TSU']

    dict_times = {}
    for driver in drivers:
        aux_laps_driver = race_laps.pick_drivers(driver)
        aux_lap1_driver = aux_laps_driver[aux_laps_driver['LapNumber']==1.0].iloc[0]
        aux_telemetry = aux_lap1_driver.get_telemetry().add_distance()
        idx_final = aux_telemetry[aux_telemetry['Speed'] >= 200].index.min()
        aux_telemetry = aux_telemetry.loc[:idx_final,]

        team_driver = aux_lap1_driver['Team']
        color = ff1.plotting.get_team_color(team_driver, race)
        dict_times[driver] = get_reaction_times(aux_telemetry)
        dict_times[driver]["color"] = color

        # Extract telemetry data off driver1 and driver2
        if driver == driver1:
            team_driver_1 = aux_lap1_driver['Team']
            telemetry_driver1 = aux_telemetry
            telemetry_driver1['Time_seconds'] = telemetry_driver1['Time'].apply(lambda i: i.total_seconds())
        
        if driver == driver2:
            team_driver_2 = aux_lap1_driver['Team']
            telemetry_driver2 = aux_telemetry
            telemetry_driver2['Time_seconds'] = telemetry_driver2['Time'].apply(lambda i: i.total_seconds())
        else:
            pass

    # Create dataframe of times
    df_times = pd.DataFrame(dict_times)
    df_times = df_times.T
    df_times.sort_values(by='time0-200', ascending=True, inplace=True)  
    print(tabulate(df_times, headers='keys', showindex=True, tablefmt='grid'))      

    # PLOT 
    columns = ["Speed", "Throttle", "Brake", "RPM"]
    dict_plot_ratios = {
        "Speed": 3,
        "Throttle": 1,
        "Brake": 1,
        "nGear": 1,
        "RPM": 1,
    }
    dict_yaxis = {
        "Speed": "Speed (km/h)",
        "Throttle": "Throttle (%)",
        "Brake": "Brake",
        "nGear": "nGear",
        "RPM": "RPM",
    }
    plot_ratios = [dict_plot_ratios[column] for column in columns]
    if tiktok_format:
        fig, ax = plt.subplots(len(columns), figsize=(9, 16), gridspec_kw={'height_ratios': plot_ratios}, dpi=700, sharex=True)
        plt.subplots_adjust(top=0.92, bottom=0.05, hspace=0.3)
    else:
        fig, ax = plt.subplots(len(columns), figsize=(12, 12), gridspec_kw={'height_ratios': plot_ratios}, dpi=700, sharex=True)
        plt.subplots_adjust(left=0.075, right=0.98, top=0.92, bottom=0.05, hspace=0.3)

    ## Colors
    color_driver_1 = ff1.plotting.get_team_color(team_driver_1, race)
    color_driver_2 = ff1.plotting.get_team_color(team_driver_2, race)
    
    if telemetry_driver1['Time_seconds'].iloc[-1] > telemetry_driver2['Time_seconds'].iloc[-1]:
        max_time = telemetry_driver1['Time_seconds'].iloc[-1] 
    else:
        max_time = telemetry_driver2['Time_seconds'].iloc[-1]

    ## Images
    results = race.results
    url_driver_1 = results.loc[results['Abbreviation']==driver1, 'HeadshotUrl'].iloc[0]
    url_driver_2 = results.loc[results['Abbreviation']==driver2, 'HeadshotUrl'].iloc[0]
    img1 = load_image(url_driver_1)
    img2 = load_image(url_driver_2)

    ## Title
    plot_title = f"{race.event.year} {race.event.EventName} - Race Start {driver1} vs {driver2}"
    ax[0].set_title(plot_title, fontdict={'fontsize': 18, 'fontweight': 'bold'})    
    for i, column in enumerate(columns):
        ax[i].plot(telemetry_driver1['Time_seconds'], telemetry_driver1[column], color=color_driver_1, label=f"{driver1} - 0-200km/h = {df_times.loc[driver1, 'time0-200']}")
        ax[i].plot(telemetry_driver2['Time_seconds'], telemetry_driver2[column], color=color_driver_2, label=f"{driver2} - 0-200km/h = {df_times.loc[driver2, 'time0-200']}")
        ax[i].set_ylabel(dict_yaxis[column], fontsize=14)
        ax[i].set_xlim([0, max_time])
        ax[i].grid(which='minor', visible=False)
        ax[i].tick_params(axis='x', which='minor', left=False, labelsize=14)
        ax[i].tick_params(axis='y', which='minor', left=False, labelsize=14)

        if column == "RPM":
            ax[i].set_xlabel("Tiempo(s)", fontsize=14)
        elif column == "Speed":
            ax[i].legend(loc="lower right", fontsize=14) 
            '''imagebox = OffsetImage(img1, zoom=0.5)
            ab = AnnotationBbox(imagebox, (4.0, 100), frameon=False)
            ax[i].add_artist(ab)'''
            # Añadir logo
            x_logo = 0.41
            y_logo = 0.82
            zoom = 0.07
            add_team_logo(ax=ax[i], x=x_logo, y=y_logo, zoom=zoom)
    # Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        eventname = race.event.EventName.split(" ")[0]
        path = f"{race.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        # Crear el subdirectorio de GP
        if not os.path.exists(path):
            os.makedirs(path)

        # Crear el subdirectorio del nombre del evento
        path = os.path.join(path, race.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar figura
        if tiktok_format:
            plot_filename = path + f"/TiktokRace_Start_{driver1}_{driver2}.png"
            plt.savefig(plot_filename, format='png')
        else:
            plot_filename = path + f"/Race_Start_{driver1}_{driver2}.pdf"
            plt.savefig(plot_filename, format='pdf')
        return df_times
    else:
        plt.show()
        return df_times
    

def plot_race_start_histogram(race, df_times, tiktok_format, save=True):
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 6), dpi=700)
    bar_width = 0.8

    # Graficos de barras
    df_times.sort_values(by='time0-100', ascending=True, inplace=True)
    positions = df_times.index.to_list()
    bar1 = ax.bar(positions, df_times["time0-100"], color=df_times["color"], label="0-100 km/h", width=bar_width, zorder=3)
    bar2 = ax.bar(positions, df_times["time100-200"], bottom=df_times["time0-100"], color=df_times["color"], alpha=0.3, edgecolor='#2a2625', label="0-200 km/h", width=bar_width, hatch='//', zorder=4)
    fig.canvas.draw()

    # Etiquetas
    for pos, total in zip(positions, df_times["time0-100"]):
        ax.text(pos, total + 0.05, f"{total:.2f}", ha="center", va="bottom", fontsize=10, color="white", zorder=5)

    for pos, total in zip(positions, df_times["time0-200"]):
        ax.text(pos, total + 0.05, f"{total:.1f}", ha="center", va="bottom", fontsize=10, color="white", zorder=6)
    
    # Details
    eventname = race.event.EventName.split(" ")[0]
    ax.set_title(f"{race.event.year} {eventname} - Tiempos de Reacción", fontsize=18, weight="bold")
    ax.set_ylabel("Tiempos de Reacción (s)", fontsize=12, weight="bold")
    ax.set_xticks(positions)
    ax.set_ylim(0, 8)
    ax.set_xlim([-0.5, len(positions) - 0.5])
    ax.tick_params(axis='x', which='minor', left=False)
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.grid(axis='x', which='minor', visible=False)
    ax.tick_params(axis='x', which='major', left=False, labelsize=11)
    ax.grid(axis='x', which='major', visible=False)
    ax.set_axisbelow(True)
    ax.legend()

    # Logo 
    x_logo = 0.3
    y_logo = 0.76
    zoom = 0.07

    add_team_logo(ax=ax, x=x_logo, y=y_logo, zoom=zoom)

    # Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        path = f"{race.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        # Crear el subdirectorio de GP
        if not os.path.exists(path):
            os.makedirs(path)

        # Crear el subdirectorio del nombre del evento
        path = os.path.join(path, race.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar figura
        if tiktok_format:
            plot_filename = path + f"/TiktokTiempos_Reaccion.png"
            plt.savefig(plot_filename, format='png')
        else:
            plot_filename = path + f"/Tiempos_Reaccion.pdf"
            plt.savefig(plot_filename, format='pdf')
    else:
        plt.show()

