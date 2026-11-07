# Library
import ast
import os
import json
from matplotlib import patches
import numpy as np
import pandas as pd
import fastf1 as ff1
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import src.driver.utils as openf1_utils

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.legend_handler import HandlerPatch
from matplotlib.text import TextPath
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from fastf1 import plotting
from fastf1 import utils
from datetime import timedelta
from scipy.spatial import ConvexHull
from fastf1.exceptions import DataNotLoadedError

from . import DICT_TEAM_COLOR

from .graphs_race import generate_recta_ajuste, get_flag_image
from src.common.const import (
    DICT_EVENT,
    DICT_RACE_COUNTRYCODE,
    DICT_METEO,
    DICT_LENGTH_CIRCUITS,
)
from statsmodels.tsa.seasonal import seasonal_decompose

from src.utils.functions import (
    DICT_TEAMS,
    add_lap_info,
    format_time,
    calcula_lateralacc,
    calcula_longacc,
    get_corners_df,
    get_lap_stats,
    get_last_year_pole_time,
    find_maxspeed_one_driver,
    find_maxspeed_one_driver_parametrize,
    estimate_unknown_speed_and_time,
    show_ranking_column,
    show_meteo_resume,
    get_circuito,
)



def drs_graphs(telemetry, ax, v_min1, v_max1, color_driver):
    # Funcion para graficar telemetria
    mask_DRS_on = telemetry["DRS"] > 8
    subset_DRS = telemetry.loc[mask_DRS_on, ["Distance", "Speed", "Throttle"]]
    index_DRS_on = subset_DRS.index.tolist()

    delta_index_DRS_on = np.diff(index_DRS_on)
    index_con_cambio_mayor_de_1 = np.where(delta_index_DRS_on > 1)[0]

    vfinales_driver = []
    distances = []
    for j, index in enumerate(index_con_cambio_mayor_de_1):
        if j == 0:
            # Primer caso
            index_inicio = subset_DRS.index[0]
            index_final = index_DRS_on[index]

            subset_DRS_i = subset_DRS.loc[index_inicio:index_final]
            ax[0].fill_between(
                subset_DRS_i["Distance"], v_min1, v_max1, color=color_driver, alpha=0.2
            )

            # Encontrams la velocidad maxima al final de recta
            v_final_driver, distance = find_maxspeed_one_driver(subset_DRS_i)
            vfinales_driver.append(v_final_driver)
            distances.append(distance)

            if j == (len(index_con_cambio_mayor_de_1) - 1):
                index_inicio = index_DRS_on[index_con_cambio_mayor_de_1[j] + 1]
                index_final = subset_DRS.index[-1]

                subset_DRS_i = subset_DRS.loc[index_inicio:index_final]
                ax[0].fill_between(
                    subset_DRS_i["Distance"],
                    v_min1,
                    v_max1,
                    color=color_driver,
                    alpha=0.2,
                )

                # Añadimos la velocidad maxima al final de recta
                v_final_driver, distance = find_maxspeed_one_driver(subset_DRS_i)
                vfinales_driver.append(v_final_driver)
                distances.append(distance)

        else:
            # Caso intermedio desde el punto anterior hasta el siguiente corte
            index_inicio = index_DRS_on[index_con_cambio_mayor_de_1[j - 1] + 1]
            index_final = index_DRS_on[index]

            subset_DRS_i = subset_DRS.loc[index_inicio:index_final]
            ax[0].fill_between(
                subset_DRS_i["Distance"], v_min1, v_max1, color=color_driver, alpha=0.2
            )

            v_final_driver, distance = find_maxspeed_one_driver(subset_DRS_i)
            vfinales_driver.append(v_final_driver)
            distances.append(distance)

            # Vemos si es el ultimo caso y le ponemos otra mas
            if j == (len(index_con_cambio_mayor_de_1) - 1):
                index_inicio = index_DRS_on[index_con_cambio_mayor_de_1[j] + 1]
                index_final = subset_DRS.index[-1]

                subset_DRS_i = subset_DRS.loc[index_inicio:index_final]
                ax[0].fill_between(
                    subset_DRS_i["Distance"],
                    v_min1,
                    v_max1,
                    color=color_driver,
                    alpha=0.2,
                )

                # Añadimos la velocidad maxima al final de recta
                v_final_driver, distance = find_maxspeed_one_driver(subset_DRS_i)
                vfinales_driver.append(v_final_driver)
                distances.append(distance)

    return vfinales_driver, distances


def drs_graphs_merged(telemetry, driver, ax, v_min1, v_max1, color_driver):
    # Control driver
    if driver == "driver_1":
        dist_column = "Distance_d1"
        speed_column = "Speed_d1"
        throttle_column = "Throttle_d1"
        drs_column = "DRS_d1"
    elif driver == "driver_2":
        dist_column = "Distance_d2"
        speed_column = "Speed_d2"
        throttle_column = "Throttle_d2"
        drs_column = "DRS_d2"

    # Funcion para graficar telemetria
    mask_DRS_on = telemetry[drs_column] > 8
    subset_DRS = telemetry.loc[
        mask_DRS_on, [dist_column, speed_column, throttle_column]
    ]
    index_DRS_on = subset_DRS.index.tolist()

    delta_index_DRS_on = np.diff(index_DRS_on)
    index_con_cambio_mayor_de_1 = np.where(delta_index_DRS_on > 1)[0]

    vfinales_driver = []
    distances = []
    for j, index in enumerate(index_con_cambio_mayor_de_1):
        if j == 0:
            # Primer caso
            index_inicio = subset_DRS.index[0]
            index_final = index_DRS_on[index]

            subset_DRS_i = subset_DRS.loc[index_inicio:index_final]
            ax[0].fill_between(
                subset_DRS_i[dist_column], v_min1, v_max1, color=color_driver, alpha=0.2
            )

            # Encontrams la velocidad maxima al final de recta
            v_final_driver, distance = find_maxspeed_one_driver_parametrize(
                subset_DRS_i, dist_column, speed_column, throttle_column
            )
            vfinales_driver.append(v_final_driver)
            distances.append(distance)

            if j == (len(index_con_cambio_mayor_de_1) - 1):
                index_inicio = index_DRS_on[index_con_cambio_mayor_de_1[j] + 1]
                index_final = subset_DRS.index[-1]

                subset_DRS_i = subset_DRS.loc[index_inicio:index_final]
                ax[0].fill_between(
                    subset_DRS_i[dist_column],
                    v_min1,
                    v_max1,
                    color=color_driver,
                    alpha=0.2,
                )

                # Añadimos la velocidad maxima al final de recta
                v_final_driver, distance = find_maxspeed_one_driver_parametrize(
                    subset_DRS_i, dist_column, speed_column, throttle_column
                )
                vfinales_driver.append(v_final_driver)
                distances.append(distance)

        else:
            # Caso intermedio desde el punto anterior hasta el siguiente corte
            index_inicio = index_DRS_on[index_con_cambio_mayor_de_1[j - 1] + 1]
            index_final = index_DRS_on[index]

            subset_DRS_i = subset_DRS.loc[index_inicio:index_final]
            ax[0].fill_between(
                subset_DRS_i[dist_column], v_min1, v_max1, color=color_driver, alpha=0.2
            )

            v_final_driver, distance = find_maxspeed_one_driver_parametrize(
                subset_DRS_i, dist_column, speed_column, throttle_column
            )
            vfinales_driver.append(v_final_driver)
            distances.append(distance)

            # Vemos si es el ultimo caso y le ponemos otra mas
            if j == (len(index_con_cambio_mayor_de_1) - 1):
                index_inicio = index_DRS_on[index_con_cambio_mayor_de_1[j] + 1]
                index_final = subset_DRS.index[-1]

                subset_DRS_i = subset_DRS.loc[index_inicio:index_final]
                ax[0].fill_between(
                    subset_DRS_i[dist_column],
                    v_min1,
                    v_max1,
                    color=color_driver,
                    alpha=0.2,
                )

                # Añadimos la velocidad maxima al final de recta
                v_final_driver, distance = find_maxspeed_one_driver_parametrize(
                    subset_DRS_i, dist_column, speed_column, throttle_column
                )
                vfinales_driver.append(v_final_driver)
                distances.append(distance)

    return vfinales_driver, distances


def add_speed_text_common(
    vfinales_driver1,
    vfinales_driver2,
    distances_1,
    distances_2,
    color_driver1,
    color_driver2,
    ax,
):
    for v_final_driver_1, v_final_driver_2, distance1, distance2 in zip(
        vfinales_driver1, vfinales_driver2, distances_1, distances_2
    ):
        if v_final_driver_1 > v_final_driver_2:
            ax[0].text(
                distance1, v_final_driver_1, str(v_final_driver_1), color=color_driver1
            )
            ax[0].text(
                distance2,
                v_final_driver_2 - 15,
                str(v_final_driver_2),
                color=color_driver2,
            )
        else:
            ax[0].text(
                distance2, v_final_driver_2, str(v_final_driver_2), color=color_driver2
            )
            ax[0].text(
                distance1,
                v_final_driver_1 - 15,
                str(v_final_driver_1),
                color=color_driver1,
            )


def add_speed_text(
    vfinales_driver1,
    vfinales_driver2,
    distances_1,
    distances_2,
    color_driver1,
    color_driver2,
    ax,
):
    if len(distances_1) == len(distances_2):
        add_speed_text_common(
            vfinales_driver1,
            vfinales_driver2,
            distances_1,
            distances_2,
            color_driver1,
            color_driver2,
            ax,
        )
    elif len(distances_1) > len(distances_2):
        add_speed_text_common(
            vfinales_driver1,
            vfinales_driver2,
            distances_1,
            distances_2,
            color_driver1,
            color_driver2,
            ax,
        )
        # Add last d1
        ax[0].text(
            distances_1[-1],
            vfinales_driver1[-1],
            str(vfinales_driver1[-1]),
            color=color_driver1,
        )
    else:
        add_speed_text_common(
            vfinales_driver1,
            vfinales_driver2,
            distances_1,
            distances_2,
            color_driver1,
            color_driver2,
            ax,
        )
        # Add last d2
        ax[0].text(
            distances_2[-1],
            vfinales_driver2[-1],
            str(vfinales_driver2[-1]),
            color=color_driver2,
        )


def graph_telemetry(
    fastest_driver_1,
    fastest_driver_2,
    telemetry_driver_1,
    telemetry_driver_2,
    driver_1,
    driver_2,
    corners,
    event,
    save=True,
):
    """
    Esta función crea un grafico de telemetría comparando los pilotos 1 y 2,
    con opción de guardar la telemetría en el drive o solo mostrar el grafico
    """
    # 1) Definimos parametros de la telemetria
    plot_size = [18, 15]
    plot_title = f"{event.event.year} {event.event.EventName} {event.name} {driver_1} vs {driver_2}"
    plot_ratios = [3, 1, 2, 1, 1, 2, 1]
    plot_filename = plot_title.replace(" ", "_") + ".pdf"
    # plot_filename = './outputs/' + plot_filename

    # 2) Cargamos el color de los pilotos
    team_driver_1 = fastest_driver_1["Team"]
    team_driver_2 = fastest_driver_2["Team"]
    color_driver1 = DICT_TEAM_COLOR[str(team_driver_1)]
    color_driver2 = DICT_TEAM_COLOR[str(team_driver_2)]

    if team_driver_1 == team_driver_2:
        # Las vueltas son con el mismo equipo
        linestyle_driver_1 = "-"
        alpha_driver_1 = 1.0
        linestyle_driver_2 = "--"
        alpha_driver_2 = 0.5
    else:
        # Las vueltas son con de distinto equipo
        linestyle_driver_1 = "-"
        alpha_driver_1 = 1.0
        linestyle_driver_2 = "-"
        alpha_driver_2 = 1.0

    # 3) Obtenemos la telemetria
    # FIXME Optional redudUCION DE DISTANCE
    # telemetry_driver_1 = telemetry_driver_1.loc[telemetry_driver_1['Distance']<=3600,]
    # telemetry_driver_2 = telemetry_driver_2.loc[telemetry_driver_2['Distance']<=3600,]
    # print(telemetry_driver_1.columns.to_list())

    # 3.5) Obtenemos los límites del gráfico en el eje x
    x_lim_ini = 0
    dist_final_driver1 = telemetry_driver_1.loc[
        telemetry_driver_1.index[-1], "Distance"
    ]
    dist_final_driver2 = telemetry_driver_2.loc[
        telemetry_driver_2.index[-1], "Distance"
    ]

    if dist_final_driver1 > dist_final_driver2:
        x_lim_fin = dist_final_driver1
    else:
        x_lim_fin = dist_final_driver2

    # 4) Calculamos el tiempo delta
    try:
        delta_time, ref_tel, compare_tel = utils.delta_time(
            fastest_driver_1, fastest_driver_2
        )
    except AttributeError:
        delta_time, ref_tel, compare_tel = openf1_utils.delta_time(
            telemetry_driver_1, telemetry_driver_2
        )

    # FIXME
    # mask_dist1_reftel = ref_tel['Distance'] >= 0
    # mask_dist2_reftel = ref_tel['Distance'] <= 3600
    # ref_tel = ref_tel[mask_dist1_reftel&mask_dist2_reftel]

    # idx_ini = ref_tel.index.to_list()[0]
    # idx_fin = ref_tel.index.to_list()[-1]
    # delta_time = delta_time[idx_ini:idx_fin+1]

    # 5) PLOTEO DE LA TELEMETRIA
    ## -----------------------
    # Establecer el tamaño de la imagen
    plt.rcParams["figure.figsize"] = plot_size

    # Creamos subplots de diferentes tamaños
    fig, ax = plt.subplots(7, gridspec_kw={"height_ratios": plot_ratios}, dpi=700)
    # plt.subplots_adjust(left=0.05, right=0.84, top=0.95, bottom=0.05, hspace=0.3)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.05, hspace=0.3)

    # Ponemos el titulo al gráfico and check si es Sprint
    if event.name == "Sprint Qualifying":
        ax[0].set_title(plot_title, fontdict={"fontsize": 20, "fontweight": "bold"})
    else:
        ax[0].set_title(plot_title, fontdict={"fontsize": 21, "fontweight": "bold"})

    # Speed
    # ----------------------------------------------------------------------------------
    time_driver1 = format_time(telemetry_driver_1.Time.iloc[-1])
    time_driver2 = format_time(telemetry_driver_2.Time.iloc[-1])

    ax[0].plot(
        telemetry_driver_1["Distance"],
        telemetry_driver_1["Speed"],
        label=driver_1 + " - " + time_driver1,
        color=color_driver1,
        alpha=alpha_driver_1,
        linewidth=2,
        linestyle=linestyle_driver_1,
    )
    ax[0].plot(
        telemetry_driver_2["Distance"],
        telemetry_driver_2["Speed"],
        label=driver_2 + " - " + time_driver2,
        color=color_driver2,
        alpha=alpha_driver_2,
        linewidth=2,
        linestyle=linestyle_driver_2,
    )

    ax[0].set_ylabel("Speed (km/h)", fontsize=12)
    # ax[0].legend(loc="best", bbox_to_anchor=(1.025, 1.0))
    # ax[0].legend(loc="lower left")
    ax[0].legend(loc="best")
    ax[0].set_xlim(x_lim_ini, x_lim_fin)

    ## Calcular cual ha sido la velocidad minima entre los dos pilotos
    v_min1 = telemetry_driver_1["Speed"].min()
    v_min2 = telemetry_driver_2["Speed"].min()
    if v_min1 < v_min2:
        v_min = v_min1
    else:
        v_min = v_min2

    ## Calcular cual ha sido la velocidad maxma entre los dos pilotos<
    v_max1 = telemetry_driver_1["Speed"].max()
    v_max2 = telemetry_driver_2["Speed"].max()

    if v_max1 > v_max2:
        v_max = v_max1
    else:
        v_max = v_max2

    ## Añadir la localización de la curva
    ax[0].vlines(
        x=corners["Distance"],
        ymin=v_min - 20,
        ymax=v_max + 20,
        linestyles="dotted",
        colors="grey",
    )

    ## Añadir el número de curva debajo de las lineas
    for _, corner in corners.iterrows():
        if "Letter" in corner:
            txt = f"{corner['Number']}{corner['Letter']}"
        else:
            txt = f"{corner['Number']}"

        ax[0].text(
            corner["Distance"],
            v_min - 30,
            txt,
            va="center_baseline",
            ha="center",
            fontsize=10,
        )
    # va = 'center_baseline', ha = 'center', size = 'small')

    ## Añadir las zonas de DRS
    vfinales_driver1, distances_1 = drs_graphs(
        telemetry_driver_1, ax, v_min1, v_max1, color_driver1
    )
    vfinales_driver2, distances_2 = drs_graphs(
        telemetry_driver_2, ax, v_min1, v_max1, color_driver2
    )
    add_speed_text(
        vfinales_driver1,
        vfinales_driver2,
        distances_1,
        distances_2,
        color_driver1,
        color_driver2,
        ax,
    )

    ## Añadir imagen del logo
    x_logo = 0.17
    y_logo = 1.01
    zoom = 0.07
    add_team_logo(ax=ax[0], x=x_logo, y=y_logo, zoom=zoom)

    # Delta line
    # ----------------------------------------------------------------------------------
    ## Calcular el deltatime
    delta_d2d1 = telemetry_driver_2.Time.iloc[-1] - telemetry_driver_1.Time.iloc[-1]
    deltatime_d2d1 = delta_time[len(delta_time) - 1]

    ## Segun quien ha ganado dar un color u otro
    if delta_d2d1 > timedelta(days=0, hours=0, minutes=0, seconds=0, milliseconds=0):
        # El piloto 1 ha sido el más rapido, la linea de referencia es con driver1
        if deltatime_d2d1 > 0.0:
            # La delta ha quedado positiva, luego esta bien calculada
            ax[1].plot(
                ref_tel["Distance"],
                delta_time,
                color=color_driver2,
                alpha=alpha_driver_2,
                linewidth=2,
                linestyle=linestyle_driver_2,
            )

            ## Añadir la localización de la curva
            ax[1].vlines(
                x=corners["Distance"],
                ymin=delta_time.min(),
                ymax=delta_time.max(),
                linestyles="dotted",
                colors="grey",
            )

            ## Añadir la linea de referencia y el texto
            ax[1].axhline(
                0,
                color=color_driver1,
                alpha=alpha_driver_1,
                linestyle=linestyle_driver_1,
            )
            ax[1].set_ylabel(f"Gap to {driver_1} (s)", fontsize=12)
            ax[1].set_xlim(x_lim_ini, x_lim_fin)

            # distance_c12 = corners.loc[corners['Number'] == 11, 'Distance'].iloc[0]
            # ref_tel['Distance_diff'] = abs(ref_tel['Distance'] - distance_c12)
            # idx_min = ref_tel['Distance_diff'].idxmin()
            # print(idx_min)
            # print(f"Delta en la Curva 11: {delta_time[idx_min]}")

        else:
            ## La delta ha quedado negativa, luego hay que multiplicarla por -1
            ax[1].plot(
                ref_tel["Distance"],
                delta_time * (-1),
                color=color_driver2,
                alpha=alpha_driver_2,
                linewidth=2,
                linestyle=linestyle_driver_2,
            )

            ## Añadir la localización de la curva
            ax[1].vlines(
                x=corners["Distance"],
                ymin=delta_time.min(),
                ymax=delta_time.max(),
                linestyles="dotted",
                colors="grey",
            )

            ## Añadir la linea de referencia y el texto
            ax[1].axhline(
                0,
                color=color_driver1,
                alpha=alpha_driver_1,
                linestyle=linestyle_driver_1,
            )
            ax[1].set_ylabel(f"Gap to {driver_1} (s)", fontsize=12)
            ax[1].set_xlim(x_lim_ini, x_lim_fin)

    else:
        # El piloto 2 ha sido el más rapido, la linea de referencia es con driver2
        ax[1].plot(
            ref_tel["Distance"],
            delta_time,
            color=color_driver1,
            alpha=alpha_driver_1,
            linewidth=2,
            linestyle=linestyle_driver_1,
        )

        ## Añadir la localización de la curva
        ax[1].vlines(
            x=corners["Distance"],
            ymin=delta_time.min(),
            ymax=delta_time.max(),
            linestyles="dotted",
            colors="grey",
        )

        ## Añadir la linea de referencia y el texto
        ax[1].axhline(
            0, color=color_driver2, alpha=alpha_driver_2, linestyle=linestyle_driver_2
        )
        ax[1].set_ylabel(f"Gap to {driver_2} (s)", fontsize=12)
        ax[1].set_xlim(x_lim_ini, x_lim_fin)

    # Throttle
    # ----------------------------------------------------------------------------------
    ax[2].plot(
        telemetry_driver_1["Distance"],
        telemetry_driver_1["Throttle"],
        label=driver_1,
        color=color_driver1,
        alpha=alpha_driver_1,
        linewidth=2,
        linestyle=linestyle_driver_1,
    )
    ax[2].plot(
        telemetry_driver_2["Distance"],
        telemetry_driver_2["Throttle"],
        label=driver_2,
        color=color_driver2,
        alpha=alpha_driver_2,
        linewidth=2,
        linestyle=linestyle_driver_2,
    )

    ## Añadir la localización de la curva
    # ax[2].vlines(x=corners['Distance'], ymin=0, ymax=100, linestyles='dotted', colors='grey')

    ## Añadir text labels
    ax[2].set_ylabel("Throttle (%)", fontsize=12)
    ax[2].set_xlim(x_lim_ini, x_lim_fin)

    # Brake
    # ----------------------------------------------------------------------------------
    ax[3].plot(
        telemetry_driver_1["Distance"],
        telemetry_driver_1["Brake"],
        label=driver_1,
        color=color_driver1,
        alpha=alpha_driver_1,
        linewidth=2,
        linestyle=linestyle_driver_1,
    )
    ax[3].plot(
        telemetry_driver_2["Distance"],
        telemetry_driver_2["Brake"],
        label=driver_2,
        color=color_driver2,
        alpha=alpha_driver_2,
        linewidth=2,
        linestyle=linestyle_driver_2,
    )

    ## Añadir la localización de la curva
    # ax[3].vlines(x=corners['Distance'], ymin=0, ymax=1, linestyles='dotted', colors='grey')

    ## Añadir text labels
    ax[3].set_ylabel("Brake", fontsize=12)
    ax[3].set_xlim(x_lim_ini, x_lim_fin)

    # Gear
    # ----------------------------------------------------------------------------------
    ax[4].plot(
        telemetry_driver_1["Distance"],
        telemetry_driver_1["nGear"],
        label=driver_1,
        color=color_driver1,
        alpha=alpha_driver_1,
        linewidth=2,
        linestyle=linestyle_driver_1,
    )
    ax[4].plot(
        telemetry_driver_2["Distance"],
        telemetry_driver_2["nGear"],
        label=driver_2,
        color=color_driver2,
        alpha=alpha_driver_2,
        linewidth=2,
        linestyle=linestyle_driver_2,
    )
    ax[4].set_ylabel("nGear", fontsize=12)
    ax[4].set_xlim(x_lim_ini, x_lim_fin)

    # RPM
    # ----------------------------------------------------------------------------------
    ax[5].plot(
        telemetry_driver_1["Distance"],
        telemetry_driver_1["RPM"],
        label=driver_1,
        color=color_driver1,
        alpha=alpha_driver_1,
        linewidth=2,
        linestyle=linestyle_driver_1,
    )
    ax[5].plot(
        telemetry_driver_2["Distance"],
        telemetry_driver_2["RPM"],
        label=driver_2,
        color=color_driver2,
        alpha=alpha_driver_2,
        linewidth=2,
        linestyle=linestyle_driver_2,
    )
    ax[5].set_ylabel("RPM", fontsize=12)
    ax[5].set_xlim(x_lim_ini, x_lim_fin)

    # DRS
    # ----------------------------------------------------------------------------------
    ax[6].plot(
        telemetry_driver_1["Distance"],
        telemetry_driver_1["DRS"],
        label=driver_1,
        color=color_driver1,
        alpha=alpha_driver_1,
        linewidth=2,
        linestyle=linestyle_driver_1,
    )
    ax[6].plot(
        telemetry_driver_2["Distance"],
        telemetry_driver_2["DRS"],
        label=driver_2,
        color=color_driver2,
        alpha=alpha_driver_2,
        linewidth=2,
        linestyle=linestyle_driver_2,
    )
    ax[6].set_ylabel("DRS", fontsize=12)
    ax[6].set_xlabel("Lap distance (m)", fontsize=12)
    ax[6].set_xlim(x_lim_ini, x_lim_fin)

    ## Comprobar el valor de Save
    ## ----------------------------------------------------------------------------------
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        eventname = event.event.EventName.split(" ")[0]
        path = f"{event.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", str(event.event.year), path)

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, event.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar figura
        plot_filename = os.path.join(path, plot_filename)
        plt.savefig(plot_filename, format="pdf")
    else:
        plt.show()


def graph_telemetry_custom(
    fastest_driver_1,
    fastest_driver_2,
    driver_1,
    driver_2,
    corners,
    columns,
    distances,
    event,
    plot_title,
    save=True,
):
    """
    Esta función crea un grafico de telemetría comparando los pilotos 1 y 2,
    con opción de guardar la telemetría en el drive o solo mostrar el grafico
    """
    tiktokformat = True
    # 1) Definimos parametros de la telemetria
    plot_size = [18, 15]
    # plot_title = f"{event.event.year} {event.event.EventName} {event.name}"
    dict_plot_ratios = {
        "Speed": 3,
        "Delta": 1,
        "Throttle": 2,
        "Brake": 1,
        "nGear": 1,
        "RPM": 2,
        "DRS": 1,
    }

    # plot_filename = './outputs/' + plot_filename

    # 2) Cargamos el color de los pilotos
    team_driver_1 = fastest_driver_1["Team"]
    team_driver_2 = fastest_driver_2["Team"]
    color_driver1 = ff1.plotting.get_team_color(team_driver_1, event)
    color_driver2 = ff1.plotting.get_team_color(team_driver_2, event)

    if team_driver_1 == team_driver_2:
        # Las vueltas son con el mismo equipo
        linestyle_driver_1 = "-"
        alpha_driver_1 = 1.0
        linestyle_driver_2 = "--"
        alpha_driver_2 = 0.5
    else:
        # Las vueltas son con de distinto equipo
        linestyle_driver_1 = "-"
        alpha_driver_1 = 1.0
        linestyle_driver_2 = "-"
        alpha_driver_2 = 1.0

    # 3) Obtenemos la telemetria
    telemetry_driver_1 = fastest_driver_1.get_telemetry().add_distance()
    telemetry_driver_2 = fastest_driver_2.get_telemetry().add_distance()

    # 3.1) Filter Distances variables in telemetries
    if distances[0] != "all":
        mask_dist1_driver1 = telemetry_driver_1["Distance"] >= distances[0]
        mask_dist2_driver1 = telemetry_driver_1["Distance"] <= distances[1]
        telemetry_driver_1 = telemetry_driver_1.loc[
            mask_dist1_driver1 & mask_dist2_driver1,
        ]
        mask_dist1_driver2 = telemetry_driver_2["Distance"] >= distances[0]
        mask_dist2_driver2 = telemetry_driver_2["Distance"] <= distances[1]
        telemetry_driver_2 = telemetry_driver_2.loc[
            mask_dist1_driver2 & mask_dist2_driver2,
        ]

        x_lim_ini = distances[0]
        x_lim_fin = distances[1]
    else:
        # 3.5) Obtenemos los límites del gráfico en el eje x
        x_lim_ini = 0
        dist_final_driver1 = telemetry_driver_1.loc[
            telemetry_driver_1.index[-1], "Distance"
        ]
        dist_final_driver2 = telemetry_driver_2.loc[
            telemetry_driver_2.index[-1], "Distance"
        ]

        if dist_final_driver1 > dist_final_driver2:
            x_lim_fin = dist_final_driver1
        else:
            x_lim_fin = dist_final_driver2

    # 3.2) Filter Distances variables in corner
    mask_corner_d1 = corners["Distance"] >= distances[0]
    mask_corner_d2 = corners["Distance"] <= distances[1]
    corners = corners.loc[mask_corner_d1 & mask_corner_d2,]

    # 4) Calculamos el tiempo delta
    delta_time, ref_tel, compare_tel = utils.delta_time(
        fastest_driver_1, fastest_driver_2
    )
    ## Subset de delta entre dist1 - dist2
    mask_dist1_reftel = ref_tel["Distance"] >= distances[0]
    mask_dist2_reftel = ref_tel["Distance"] <= distances[1]
    ref_tel = ref_tel[mask_dist1_reftel & mask_dist2_reftel]

    idx_ini = ref_tel.index.to_list()[0]
    idx_fin = ref_tel.index.to_list()[-1]
    delta_time = delta_time[idx_ini : idx_fin + 1]

    # 5) PLOTEO DE LA TELEMETRIA
    ## -----------------------
    # Establecer el tamaño de la imagen
    plt.rcParams["figure.figsize"] = plot_size

    # Creamos subplots de diferentes tamaños
    plot_ratios = [dict_plot_ratios[column] for column in columns]
    if tiktokformat:
        fig, ax = plt.subplots(
            len(columns),
            figsize=[9, 16],
            gridspec_kw={"height_ratios": plot_ratios},
            dpi=700,
            sharex=True,
        )
        plt.subplots_adjust(top=0.92, bottom=0.05, hspace=0.3, left=0.2, right=0.9)
    else:
        fig, ax = plt.subplots(
            len(columns),
            gridspec_kw={"height_ratios": plot_ratios},
            dpi=700,
            sharex=True,
        )
        plt.subplots_adjust(left=0.075, right=0.98, top=0.92, bottom=0.05, hspace=0.3)

    for i, column in enumerate(columns):
        ax[i].tick_params(axis="x", labelsize=14)
        ax[i].tick_params(axis="y", labelsize=16)

        if column == "Speed":
            # Ponemos el titulo al gráfico and check si es Sprint
            if event.name == "Sprint Qualifying":
                ax[i].set_title(
                    plot_title, fontdict={"fontsize": 20, "fontweight": "bold"}
                )
            else:
                ax[i].set_title(
                    plot_title, fontdict={"fontsize": 20, "fontweight": "bold"}
                )

            # Speed
            # ----------------------------------------------------------------------------------
            nrow1 = telemetry_driver_1.shape[0]
            nrow2 = telemetry_driver_2.shape[0]
            # time_driver1 = format_time(telemetry_driver_1.Time[nrow1 + 1])
            # time_driver2 = format_time(telemetry_driver_2.Time[nrow2 + 1])

            ax[i].plot(
                telemetry_driver_1["Distance"],
                telemetry_driver_1["Speed"],
                label=driver_1,
                color=color_driver1,
                alpha=alpha_driver_1,
                linewidth=3,
                linestyle=linestyle_driver_1,
            )
            ax[i].plot(
                telemetry_driver_2["Distance"],
                telemetry_driver_2["Speed"],
                label=driver_2,
                color=color_driver2,
                alpha=alpha_driver_2,
                linewidth=3,
                linestyle=linestyle_driver_2,
            )

            ax[i].set_ylabel("Speed (km/h)", fontsize=16)
            # ax[0].legend(loc="best", bbox_to_anchor=(1.025, 1.0))
            ax[i].legend(loc="lower left", fontsize=16)
            ax[i].set_xlim(x_lim_ini, x_lim_fin)

            ## Calcular cual ha sido la velocidad minima entre los dos pilotos
            v_min1 = telemetry_driver_1["Speed"].min()
            v_min2 = telemetry_driver_2["Speed"].min()
            if v_min1 < v_min2:
                v_min = v_min1
            else:
                v_min = v_min2

            ## Calcular cual ha sido la velocidad maxma entre los dos pilotos
            v_max1 = telemetry_driver_1["Speed"].max()
            v_max2 = telemetry_driver_2["Speed"].max()

            if v_max1 > v_max2:
                v_max = v_max1
            else:
                v_max = v_max2

            # ylim
            ax[i].set_ylim(v_min - 10, v_max + 5)

            ## Añadir la localización de la curva
            ax[i].vlines(
                x=corners["Distance"],
                ymin=v_min - 20,
                ymax=v_max + 20,
                linestyles="dotted",
                colors="grey",
            )

            ## Añadir el número de curva debajo de las lineas
            for _, corner in corners.iterrows():
                txt = f"{corner['Number']}{corner['Letter']}"
                ax[i].text(
                    corner["Distance"],
                    v_min - 30,
                    txt,
                    va="center_baseline",
                    ha="center",
                    fontsize=12,
                )
            # va = 'center_baseline', ha = 'center', size = 'small')

            ## Añadir las zonas de DRS
            ## Driver 1
            vfinales_driver1, distances_1 = drs_graphs(
                telemetry_driver_1, ax, v_min1, v_max1, color_driver1
            )

            ## Driver 2
            vfinales_driver2, distances_2 = drs_graphs(
                telemetry_driver_2, ax, v_min1, v_max1, color_driver2
            )

            # Añadir el texto de la velocidad, ordenando por el piloto mas rapido
            for i in range(len(vfinales_driver1)):
                v_final_driver_1 = vfinales_driver1[i]
                distance1 = distances_1[i]
                v_final_driver_2 = vfinales_driver2[i]
                distance2 = distances_2[i]

                if v_final_driver_1 > v_final_driver_2:
                    ax[i].text(
                        distance1,
                        v_final_driver_1,
                        str(v_final_driver_1),
                        color=color_driver1,
                    )
                    ax[i].text(
                        distance2,
                        v_final_driver_2 - 15,
                        str(v_final_driver_2),
                        color=color_driver2,
                    )
                else:
                    ax[i].text(
                        distance2,
                        v_final_driver_2,
                        str(v_final_driver_2),
                        color=color_driver2,
                    )
                    ax[i].text(
                        distance1,
                        v_final_driver_1 - 15,
                        str(v_final_driver_1),
                        color=color_driver1,
                    )

            ## Añadir imagen del logo
            x_logo = 0.99
            y_logo = 0.7
            # y_logo = 0.04
            zoom = 0.07
            add_team_logo(ax=ax[i], x=x_logo, y=y_logo, zoom=zoom)

        elif column == "Delta":
            # Delta line
            # ----------------------------------------------------------------------------------
            ## Calcular el deltatime
            delta_d2d1 = (
                telemetry_driver_2.Time.iloc[-1] - telemetry_driver_1.Time.iloc[-1]
            )
            deltatime_d2d1 = delta_time.iloc[len(delta_time) - 1]

            ## Segun quien ha ganado dar un color u otro
            if delta_d2d1 > timedelta(
                days=0, hours=0, minutes=0, seconds=0, milliseconds=0
            ):
                # El piloto 1 ha sido el más rapido, la linea de referencia es con driver1
                if deltatime_d2d1 > 0.0:
                    # print("Entra 1")
                    # La delta ha quedado positiva, luego esta bien calculada
                    ax[i].plot(
                        ref_tel["Distance"],
                        delta_time,
                        color=color_driver2,
                        alpha=alpha_driver_2,
                        linewidth=3,
                        linestyle=linestyle_driver_2,
                    )

                    ## Añadir la localización de la curva
                    ax[i].vlines(
                        x=corners["Distance"],
                        ymin=delta_time.min(),
                        ymax=delta_time.max(),
                        linestyles="dotted",
                        colors="grey",
                    )

                    ## Añadir la linea de referencia y el texto
                    ax[i].axhline(
                        0,
                        color=color_driver1,
                        alpha=alpha_driver_1,
                        linestyle=linestyle_driver_1,
                        linewidth=3,
                    )
                    ax[i].set_ylabel(f"Gap to {driver_1} (s)", fontsize=12)
                    ax[i].set_xlim(x_lim_ini, x_lim_fin)
                    # print(f"min: {min(delta_time)-0.1}, max={max(delta_time)+0.1}")
                    ax[i].set_ylim(min(delta_time) - 0.1, max(delta_time) + 0.1)
                    print(f"Delta final = {deltatime_d2d1}")
                else:
                    # print("Entra 2")
                    ## La delta ha quedado negativa, luego hay que multiplicarla por -1
                    ax[i].plot(
                        ref_tel["Distance"],
                        delta_time * (-1),
                        color=color_driver2,
                        alpha=alpha_driver_2,
                        linewidth=3,
                        linestyle=linestyle_driver_2,
                    )

                    ## Añadir la localización de la curva
                    ax[i].vlines(
                        x=corners["Distance"],
                        ymin=delta_time.min(),
                        ymax=delta_time.max(),
                        linestyles="dotted",
                        colors="grey",
                    )

                    ## Añadir la linea de referencia y el texto
                    ax[i].axhline(
                        0,
                        color=color_driver1,
                        alpha=alpha_driver_1,
                        linestyle=linestyle_driver_1,
                        linewidth=3,
                    )
                    ax[i].set_ylabel(f"Gap to {driver_1} (s)", fontsize=12)
                    ax[i].set_xlim(x_lim_ini, x_lim_fin)
                    # print(f"min: {min(delta_time)-0.1}, max={max(delta_time)+0.1}")
                    ax[i].set_ylim(min(delta_time) - 0.1, max(delta_time) + 0.1)
                    print(f"Delta final = {deltatime_d2d1}")
            else:
                # print("Entra 3")
                # El piloto 2 ha sido el más rapido, la linea de referencia es con driver2
                ax[i].plot(
                    ref_tel["Distance"],
                    delta_time,
                    color=color_driver1,
                    alpha=alpha_driver_1,
                    linewidth=3,
                    linestyle=linestyle_driver_1,
                )

                ## Añadir la localización de la curva
                ax[i].vlines(
                    x=corners["Distance"],
                    ymin=delta_time.min(),
                    ymax=delta_time.max(),
                    linestyles="dotted",
                    colors="grey",
                )

                ## Añadir la linea de referencia y el texto
                ax[i].axhline(
                    0,
                    color=color_driver2,
                    alpha=alpha_driver_2,
                    linestyle=linestyle_driver_2,
                    linewidth=3,
                )
                ax[i].set_ylabel(f"Gap to {driver_2} (s)", fontsize=12)
                ax[i].set_xlim(x_lim_ini, x_lim_fin)
                # print(f"min: {min(delta_time)-0.1}, max={max(delta_time)+0.1}")
                ax[i].set_ylim(min(delta_time) - 0.1, max(delta_time) + 0.1)
                print(f"Delta final = {deltatime_d2d1}")

        elif column == "Throttle":
            # Throttle
            # ----------------------------------------------------------------------------------
            ax[i].plot(
                telemetry_driver_1["Distance"],
                telemetry_driver_1["Throttle"],
                label=driver_1,
                color=color_driver1,
                alpha=alpha_driver_1,
                linewidth=3,
                linestyle=linestyle_driver_1,
            )
            ax[i].plot(
                telemetry_driver_2["Distance"],
                telemetry_driver_2["Throttle"],
                label=driver_2,
                color=color_driver2,
                alpha=alpha_driver_2,
                linewidth=3,
                linestyle=linestyle_driver_2,
            )

            ## Añadir la localización de la curva
            # ax[i].vlines(x=corners['Distance'], ymin=0, ymax=100, linestyles='dotted', colors='grey')

            ## Añadir text labels
            ax[i].set_ylabel("Throttle (%)", fontsize=16)
            ax[i].set_xlim(x_lim_ini, x_lim_fin)
            ax[i].set_ylim(0, 100)

        elif column == "Brake":
            # Brake
            # ----------------------------------------------------------------------------------
            ax[i].plot(
                telemetry_driver_1["Distance"],
                telemetry_driver_1["Brake"],
                label=driver_1,
                color=color_driver1,
                alpha=alpha_driver_1,
                linewidth=3,
                linestyle=linestyle_driver_1,
            )
            ax[i].plot(
                telemetry_driver_2["Distance"],
                telemetry_driver_2["Brake"],
                label=driver_2,
                color=color_driver2,
                alpha=alpha_driver_2,
                linewidth=3,
                linestyle=linestyle_driver_2,
            )

            ## Añadir la localización de la curva
            # ax[i].vlines(x=corners['Distance'], ymin=0, ymax=1, linestyles='dotted', colors='grey')

            ## Añadir text labels
            ax[i].set_ylabel("Brake", fontsize=12)
            ax[i].set_xlim(x_lim_ini, x_lim_fin)

        elif column == "nGear":
            # Gear
            # ----------------------------------------------------------------------------------
            ax[i].plot(
                telemetry_driver_1["Distance"],
                telemetry_driver_1["nGear"],
                label=driver_1,
                color=color_driver1,
                alpha=alpha_driver_1,
                linewidth=3,
                linestyle=linestyle_driver_1,
            )
            ax[i].plot(
                telemetry_driver_2["Distance"],
                telemetry_driver_2["nGear"],
                label=driver_2,
                color=color_driver2,
                alpha=alpha_driver_2,
                linewidth=3,
                linestyle=linestyle_driver_2,
            )
            ax[i].set_ylabel("nGear", fontsize=12)
            ax[i].set_xlim(x_lim_ini, x_lim_fin)

        elif column == "RPM":
            # RPM
            # ----------------------------------------------------------------------------------
            ax[i].plot(
                telemetry_driver_1["Distance"],
                telemetry_driver_1["RPM"],
                label=driver_1,
                color=color_driver1,
                alpha=alpha_driver_1,
                linewidth=3,
                linestyle=linestyle_driver_1,
            )
            ax[i].plot(
                telemetry_driver_2["Distance"],
                telemetry_driver_2["RPM"],
                label=driver_2,
                color=color_driver2,
                alpha=alpha_driver_2,
                linewidth=3,
                linestyle=linestyle_driver_2,
            )
            ax[i].set_ylabel("RPM", fontsize=12)
            ax[i].set_xlim(x_lim_ini, x_lim_fin)

        elif column == "DRS":
            # DRS
            # ----------------------------------------------------------------------------------
            ax[i].plot(
                telemetry_driver_1["Distance"],
                telemetry_driver_1["DRS"],
                label=driver_1,
                color=color_driver1,
                alpha=alpha_driver_1,
                linewidth=3,
                linestyle=linestyle_driver_1,
            )
            ax[i].plot(
                telemetry_driver_2["Distance"],
                telemetry_driver_2["DRS"],
                label=driver_2,
                color=color_driver2,
                alpha=alpha_driver_2,
                linewidth=3,
                linestyle=linestyle_driver_2,
            )
            ax[i].set_ylabel("DRS", fontsize=12)
            ax[i].set_xlabel("Lap distance (m)", fontsize=12)
            ax[i].set_xlim(x_lim_ini, x_lim_fin)

    ## Comprobar el valor de Save
    ## ----------------------------------------------------------------------------------
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        eventname = event.event.EventName.split(" ")[0]
        path = f"{event.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, event.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar figura
        if tiktokformat:
            plot_filename = plot_title.replace(" ", "_") + ".png"
            plot_filename = os.path.join(path, plot_filename)
            plt.savefig(plot_filename, format="png")
        else:
            plot_filename = plot_title.replace(" ", "_") + ".pdf"
            plot_filename = os.path.join(path, plot_filename)
            plt.savefig(plot_filename, format="pdf")
    else:
        plt.show()


def graph_telemetry_custom_merged(
    fastest_driver_1,
    fastest_driver_2,
    driver_1,
    driver_2,
    corners,
    columns,
    distances,
    event,
    save=True,
):
    """
    Esta función crea un grafico de telemetría comparando los pilotos 1 y 2,
    con opción de guardar la telemetría en el drive o solo mostrar el grafico
    """
    tiktokformat = True
    # 1) Definimos parametros de la telemetria
    plot_size = [18, 15]
    # plot_title = f"{event.event.year} {event.event.EventName} {event.name}"
    plot_title = f"{event.event.year} Salida Canada Merged"
    dict_plot_ratios = {
        "Speed": 3,
        "Throttle": 2,
        "Brake": 1,
        "nGear": 1,
        "RPM": 2,
        "DRS": 1,
    }

    # plot_filename = './outputs/' + plot_filename

    # 2) Cargamos el color de los pilotos
    team_driver_1 = fastest_driver_1["Team"]
    team_driver_2 = fastest_driver_2["Team"]
    color_driver1 = ff1.plotting.get_team_color(team_driver_1, event)
    color_driver2 = ff1.plotting.get_team_color(team_driver_2, event)

    if team_driver_1 == team_driver_2:
        # Las vueltas son con el mismo equipo
        linestyle_driver_1 = "-"
        alpha_driver_1 = 1.0
        linestyle_driver_2 = "--"
        alpha_driver_2 = 0.5
    else:
        # Las vueltas son con de distinto equipo
        linestyle_driver_1 = "-"
        alpha_driver_1 = 1.0
        linestyle_driver_2 = "-"
        alpha_driver_2 = 1.0

    # 3) Obtenemos la telemetria
    telemetry_driver_1 = fastest_driver_1.get_telemetry().add_distance()
    telemetry_driver_2 = fastest_driver_2.get_telemetry().add_distance()
    merged = telemetry_driver_1.merge(
        telemetry_driver_2, on="SessionTime", how="inner", suffixes=("_d1", "_d2")
    )

    # 3.1) Filter Distances variables
    if distances[0] != "all":
        mask_dist1 = merged["Distance_d1"] >= distances[0]
        mask_dist2 = merged["Distance_d1"] <= distances[1]
        merged = merged.loc[mask_dist1 & mask_dist2,]

        x_lim_ini = distances[0]
        x_lim_fin = distances[1]
    else:
        # 3.5) Obtenemos los límites del gráfico en el eje x
        x_lim_ini = 0
        x_lim_fin = merged["Distance_d1"].iloc[-1]

    # 4) PLOTEO DE LA TELEMETRIA
    ## -----------------------
    # Establecer el tamaño de la imagen
    plt.rcParams["figure.figsize"] = plot_size

    # Creamos subplots de diferentes tamaños
    plot_ratios = [dict_plot_ratios[column] for column in columns]
    if tiktokformat:
        fig, ax = plt.subplots(
            len(columns),
            figsize=[9, 16],
            gridspec_kw={"height_ratios": plot_ratios},
            dpi=700,
            sharex=True,
        )
        plt.subplots_adjust(top=0.92, bottom=0.05, hspace=0.3, left=0.2, right=0.9)
    else:
        fig, ax = plt.subplots(
            len(columns),
            gridspec_kw={"height_ratios": plot_ratios},
            dpi=700,
            sharex=True,
        )
        plt.subplots_adjust(left=0.075, right=0.98, top=0.92, bottom=0.05, hspace=0.3)

    for i, column in enumerate(columns):
        ax[i].tick_params(axis="x", labelsize=14)
        ax[i].tick_params(axis="y", labelsize=16)

        if column == "Speed":
            # Ponemos el titulo al gráfico and check si es Sprint
            if event.name == "Sprint Qualifying":
                ax[i].set_title(
                    plot_title, fontdict={"fontsize": 20, "fontweight": "bold"}
                )
            else:
                ax[i].set_title(
                    plot_title, fontdict={"fontsize": 25, "fontweight": "bold"}
                )

            # Speed
            # ----------------------------------------------------------------------------------
            ax[i].plot(
                merged["Distance_d1"],
                merged["Speed_d1"],
                label=driver_1,
                color=color_driver1,
                alpha=alpha_driver_1,
                linewidth=2,
                linestyle=linestyle_driver_1,
            )
            ax[i].plot(
                merged["Distance_d2"],
                merged["Speed_d2"],
                label=driver_2,
                color=color_driver2,
                alpha=alpha_driver_2,
                linewidth=2,
                linestyle=linestyle_driver_2,
            )

            ax[i].set_ylabel("Speed (km/h)", fontsize=16)
            # ax[0].legend(loc="best", bbox_to_anchor=(1.025, 1.0))
            ax[i].legend(loc="lower left", fontsize=16)
            ax[i].set_xlim(x_lim_ini, x_lim_fin)

            ## Calcular cual ha sido la velocidad minima entre los dos pilotos
            v_min1 = merged["Speed_d1"].min()
            v_min2 = merged["Speed_d2"].min()
            if v_min1 < v_min2:
                v_min = v_min1
            else:
                v_min = v_min2

            ## Calcular cual ha sido la velocidad maxma entre los dos pilotos
            v_max1 = merged["Speed_d1"].max()
            v_max2 = merged["Speed_d2"].max()

            if v_max1 > v_max2:
                v_max = v_max1
            else:
                v_max = v_max2

            # ylim
            ax[i].set_ylim(v_min - 5, v_max + 5)

            ## Añadir la localización de la curva
            ax[i].vlines(
                x=corners["Distance"],
                ymin=v_min - 20,
                ymax=v_max + 20,
                linestyles="dotted",
                colors="grey",
            )

            ## Añadir el número de curva debajo de las lineas
            for _, corner in corners.iterrows():
                txt = f"{corner['Number']}{corner['Letter']}"
                ax[i].text(
                    corner["Distance"],
                    v_min - 30,
                    txt,
                    va="center_baseline",
                    ha="center",
                    fontsize=12,
                )
            # va = 'center_baseline', ha = 'center', size = 'small')

            ## Añadir las zonas de DRS
            vfinales_driver1, distances_1 = drs_graphs_merged(
                merged, "driver_1", ax, v_min1, v_max1, color_driver1
            )
            vfinales_driver2, distances_2 = drs_graphs_merged(
                merged, "driver_2", ax, v_min1, v_max1, color_driver2
            )

            # Añadir el texto de la velocidad, ordenando por el piloto mas rapido
            for i in range(len(vfinales_driver1)):
                v_final_driver_1 = vfinales_driver1[i]
                distance1 = distances_1[i]
                v_final_driver_2 = vfinales_driver2[i]
                distance2 = distances_2[i]

                if v_final_driver_1 > v_final_driver_2:
                    ax[i].text(
                        distance1,
                        v_final_driver_1,
                        str(v_final_driver_1),
                        color=color_driver1,
                    )
                    ax[i].text(
                        distance2,
                        v_final_driver_2 - 15,
                        str(v_final_driver_2),
                        color=color_driver2,
                    )
                else:
                    ax[i].text(
                        distance2,
                        v_final_driver_2,
                        str(v_final_driver_2),
                        color=color_driver2,
                    )
                    ax[i].text(
                        distance1,
                        v_final_driver_1 - 15,
                        str(v_final_driver_1),
                        color=color_driver1,
                    )

            ## Añadir imagen del logo
            x_logo = 0.96
            y_logo = 0.04
            zoom = 0.07
            add_team_logo(ax=ax[i], x=x_logo, y=y_logo, zoom=zoom)

        elif column == "Throttle":
            # Throttle
            # ----------------------------------------------------------------------------------
            ax[i].plot(
                merged["Distance_d1"],
                merged["Throttle_d1"],
                label=driver_1,
                color=color_driver1,
                alpha=alpha_driver_1,
                linewidth=2,
                linestyle=linestyle_driver_1,
            )
            ax[i].plot(
                merged["Distance_d2"],
                merged["Throttle_d2"],
                label=driver_2,
                color=color_driver2,
                alpha=alpha_driver_2,
                linewidth=2,
                linestyle=linestyle_driver_2,
            )

            ## Añadir la localización de la curva
            # ax[i].vlines(x=corners['Distance'], ymin=0, ymax=100, linestyles='dotted', colors='grey')

            ## Añadir text labels
            ax[i].set_ylabel("Throttle (%)", fontsize=16)
            ax[i].set_xlim(x_lim_ini, x_lim_fin)
            ax[i].set_ylim(0, 100)

        elif column == "Brake":
            # Brake
            # ----------------------------------------------------------------------------------
            ax[i].plot(
                merged["Distance_d1"],
                merged["Brake_d1"],
                label=driver_1,
                color=color_driver1,
                alpha=alpha_driver_1,
                linewidth=2,
                linestyle=linestyle_driver_1,
            )
            ax[i].plot(
                merged["Distance_d2"],
                merged["Brake_d2"],
                label=driver_2,
                color=color_driver2,
                alpha=alpha_driver_2,
                linewidth=2,
                linestyle=linestyle_driver_2,
            )

            ## Añadir la localización de la curva
            # ax[i].vlines(x=corners['Distance'], ymin=0, ymax=1, linestyles='dotted', colors='grey')

            ## Añadir text labels
            ax[i].set_ylabel("Brake", fontsize=12)
            ax[i].set_xlim(x_lim_ini, x_lim_fin)

        elif column == "nGear":
            # Gear
            # ----------------------------------------------------------------------------------
            ax[i].plot(
                merged["Distance_d1"],
                merged["nGear_d1"],
                label=driver_1,
                color=color_driver1,
                alpha=alpha_driver_1,
                linewidth=2,
                linestyle=linestyle_driver_1,
            )
            ax[i].plot(
                merged["Distance_d2"],
                merged["nGear_d2"],
                label=driver_2,
                color=color_driver2,
                alpha=alpha_driver_2,
                linewidth=2,
                linestyle=linestyle_driver_2,
            )
            ax[i].set_ylabel("nGear", fontsize=12)
            ax[i].set_xlim(x_lim_ini, x_lim_fin)

        elif column == "RPM":
            # RPM
            # ----------------------------------------------------------------------------------
            ax[i].plot(
                merged["Distance_d1"],
                merged["RPM_d1"],
                label=driver_1,
                color=color_driver1,
                alpha=alpha_driver_1,
                linewidth=2,
                linestyle=linestyle_driver_1,
            )
            ax[i].plot(
                merged["Distance_d2"],
                merged["RPM_d2"],
                label=driver_2,
                color=color_driver2,
                alpha=alpha_driver_2,
                linewidth=2,
                linestyle=linestyle_driver_2,
            )
            ax[i].set_ylabel("RPM", fontsize=12)
            ax[i].set_xlim(x_lim_ini, x_lim_fin)

        elif column == "DRS":
            # DRS
            # ----------------------------------------------------------------------------------
            ax[i].plot(
                merged["Distance_d1"],
                merged["DRS_d1"],
                label=driver_1,
                color=color_driver1,
                alpha=alpha_driver_1,
                linewidth=2,
                linestyle=linestyle_driver_1,
            )
            ax[i].plot(
                merged["Distance_d2"],
                merged["DRS_d2"],
                label=driver_2,
                color=color_driver2,
                alpha=alpha_driver_2,
                linewidth=2,
                linestyle=linestyle_driver_2,
            )
            ax[i].set_ylabel("DRS", fontsize=12)
            ax[i].set_xlabel("Lap distance (m)", fontsize=12)
            ax[i].set_xlim(x_lim_ini, x_lim_fin)

    ## Comprobar el valor de Save
    ## ----------------------------------------------------------------------------------
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        eventname = event.event.EventName.split(" ")[0]
        path = f"{event.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, event.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar figura
        if tiktokformat:
            plot_filename = plot_title.replace(" ", "_") + ".png"
            plot_filename = os.path.join(path, plot_filename)
            plt.savefig(plot_filename, format="png")
        else:
            plot_filename = plot_title.replace(" ", "_") + ".pdf"
            plot_filename = os.path.join(path, plot_filename)
            plt.savefig(plot_filename, format="pdf")
    else:
        plt.show()


def graph_comparaison_telemetry_twolaps(
    fastest_driver_1,
    fastest_driver_2,
    telemetry_driver_1,
    telemetry_driver_2,
    driver_1,
    driver_2,
    corners,
    event,
    save=True,
):
    """
    Esta función crea un grafico de telemetría comparando los pilotos 1 y 2,
    con opción de guardar la telemetría en el drive o solo mostrar el grafico
    """
    # 1) Definimos parametros de la telemetria
    plot_size = [16, 14]
    plot_title = f"{event.event.year} {event.event.EventName} {event.name} {driver_1} vs {driver_2}"
    plot_ratios = [3, 1, 2, 1, 1, 2, 1]
    plot_filename = plot_title.replace(" ", "_") + ".pdf"
    plot_filename = "./outputs/" + plot_filename

    # 2) Cargamos el color de los pilotos
    team_driver_1 = fastest_driver_1.loc[fastest_driver_1.index[0], "Team"]
    team_driver_2 = fastest_driver_2.loc[fastest_driver_2.index[0], "Team"]

    color_driver1 = ff1.plotting.get_team_color(team_driver_1, event)
    color_driver2 = ff1.plotting.get_team_color(team_driver_2, event)

    if team_driver_1 == team_driver_2:
        # Las vueltas son con el mismo equipo
        linestyle_driver_1 = "-"
        linestyle_driver_2 = "--"
    else:
        # Las vueltas son con de distinto equipo
        linestyle_driver_1 = "-"
        linestyle_driver_2 = "-"

    # 4) Calculamos el tiempo delta
    delta_time, ref_tel, compare_tel = utils.delta_time(
        fastest_driver_1, fastest_driver_2
    )

    # 5) PLOTEO DE LA TELEMETRIA
    ## -----------------------
    # Establecer el tamaño de la imagen
    plt.rcParams["figure.figsize"] = plot_size

    # Creamos subplots de diferentes tamaños
    fig, ax = plt.subplots(7, gridspec_kw={"height_ratios": plot_ratios})

    # Ponemos el titulo al gráfico
    ax[0].title.set_text(plot_title)

    # Speed
    # ----------------------------------------------------------------------------------
    telemetry_1_index = telemetry_driver_1.index.tolist()
    telemetry_2_index = telemetry_driver_2.index.tolist()

    nrow1 = telemetry_1_index[len(telemetry_1_index) - 1]
    n0 = telemetry_1_index[0]
    time_driver1 = format_time(
        telemetry_driver_1.Time[nrow1] - telemetry_driver_1.Time[n0]
    )
    nrow2 = telemetry_2_index[len(telemetry_2_index) - 1]
    n0 = telemetry_2_index[0]
    time_driver2 = format_time(
        telemetry_driver_2.Time[nrow2] - telemetry_driver_2.Time[n0]
    )

    ax[0].plot(
        telemetry_driver_1["DistanceLap"],
        telemetry_driver_1["Speed"],
        label=driver_1 + " - " + time_driver1,
        color=color_driver1,
        linewidth=2,
        linestyle=linestyle_driver_1,
    )
    ax[0].plot(
        telemetry_driver_2["DistanceLap"],
        telemetry_driver_2["Speed"],
        label=driver_2 + " - " + time_driver2,
        color=color_driver2,
        linewidth=2,
        linestyle=linestyle_driver_2,
    )
    ax[0].set(ylabel="Speed (km/h)")
    ax[0].legend(loc="best")

    ## Calcular cual ha sido la velocidad minima entre los dos pilotos
    v_min1 = telemetry_driver_1["Speed"].min()
    v_min2 = telemetry_driver_2["Speed"].min()
    if v_min1 < v_min2:
        v_min = v_min1
    else:
        v_min = v_min2

    ## Calcular cual ha sido la velocidad maxma entre los dos pilotos
    v_max1 = telemetry_driver_1["Speed"].max()
    v_max2 = telemetry_driver_2["Speed"].max()

    if v_max1 > v_max2:
        v_max = v_max1
    else:
        v_max = v_max2

    ## Añadir la localización de la curva
    ax[0].vlines(
        x=corners["Distance"],
        ymin=v_min - 20,
        ymax=v_max + 20,
        linestyles="dotted",
        colors="grey",
    )

    ## Añadir el número de curva debajo de las lineas
    for _, corner in corners.iterrows():
        txt = f"{corner['Number']}{corner['Letter']}"
        ax[0].text(
            corner["Distance"],
            v_min - 30,
            txt,
            va="center_baseline",
            ha="center",
            size="small",
        )

    ## Añadir las zonas de DRS
    """mask_DRS_on = telemetry_driver_1['DRS'] > 8
    subset_DRS = telemetry_driver_1.loc[mask_DRS_on, 'Distance']
    index_DRS_on = subset_DRS.index.tolist()

    delta_index_DRS_on = np.diff(index_DRS_on)
    index_con_cambio_mayor_de_1 = np.where(delta_index_DRS_on > 1)[0]

    for i, index in enumerate(index_con_cambio_mayor_de_1):
        if i == 0:
            # Primer caso
            index_inicio = subset_DRS.index[0]
            index_final = index_DRS_on[index]

            subset_DRS_i = subset_DRS.loc[index_inicio:index_final]
            ax[0].fill_between(subset_DRS_i, v_min1, v_max1, color='yellow', alpha=0.2)

            # Añadimos la velocidad maxima al final de recta
            paint_maxspeed(subset_DRS_i, telemetry_driver_1, telemetry_driver_2,
                           ax, color_driver1, color_driver2)

        else:
            # Caso intermedio desde el punto anterior hasta el siguiente corte
            index_inicio = index_DRS_on[index_con_cambio_mayor_de_1[i-1] + 1]
            index_final = index_DRS_on[index]

            subset_DRS_i = subset_DRS.loc[index_inicio:index_final]
            ax[0].fill_between(subset_DRS_i, v_min1, v_max1, color='yellow', alpha=0.2)

            # Añadimos la velocidad maxima al final de recta
            paint_maxspeed(subset_DRS_i, telemetry_driver_1, telemetry_driver_2,
                           ax, color_driver1, color_driver2)

            # Vemos si es el ultimo caso y le ponemos otra mas
            if i == (len(index_con_cambio_mayor_de_1) - 1):
                index_inicio = index_DRS_on[index_con_cambio_mayor_de_1[i] + 1]
                index_final = subset_DRS.index[-1]

                subset_DRS_i = subset_DRS.loc[index_inicio:index_final]
                ax[0].fill_between(subset_DRS_i, v_min1, v_max1, color='yellow', alpha=0.2)

                # Añadimos la velocidad maxima al final de recta
                paint_maxspeed(subset_DRS_i, telemetry_driver_1, telemetry_driver_2,
                               ax, color_driver1, color_driver2)

    # Delta line
    # ----------------------------------------------------------------------------------
    ## Calcular el deltatime
    delta_d2d1 = telemetry_driver_2.Time[nrow2] - telemetry_driver_1.Time[nrow1]
    deltatime_d2d1 = delta_time[len(delta_time)-1]

    ## Segun quien ha ganado dar un color u otro
    if delta_d2d1 > timedelta(days=0,hours=0,minutes=0, seconds=0, milliseconds=0):
        # El piloto 1 ha sido el más rapido, la linea de referencia es con driver1
        if deltatime_d2d1 > 0.0:
            # La delta ha quedado positiva, luego esta bien calculada
            ax[1].plot(ref_tel['Distance'], delta_time,  color=color_driver2, linewidth=2, linestyle=linestyle_driver_2)

            ## Añadir la localización de la curva
            #ax[1].vlines(x=corners['Distance'], ymin=delta_time.min(), ymax=delta_time.max(), linestyles='dotted', colors='grey')

            ## Añadir la linea de referencia y el texto
            ax[1].axhline(0, color=color_driver1, linestyle=linestyle_driver_1)
            ax[1].set(ylabel=f"Gap to {driver_1} (s)")

        else:
            ## La delta ha quedado negativa, luego hay que multiplicarla por -1
            ax[1].plot(ref_tel['Distance'], delta_time*(-1), color=color_driver2, linewidth=2, linestyle=linestyle_driver_2)

            ## Añadir la localización de la curva
            #ax[1].vlines(x=corners['Distance'], ymin=delta_time.min(), ymax=delta_time.max(), linestyles='dotted', colors='grey')

            ## Añadir la linea de referencia y el texto
            ax[1].axhline(0,  color=color_driver1, linestyle=linestyle_driver_1)
            ax[1].set(ylabel=f"Gap to {driver_1} (s)")

    else:
        # El piloto 2 ha sido el más rapido, la linea de referencia es con driver2
        ax[1].plot(ref_tel['Distance'], delta_time,  color=color_driver1, linewidth=2, linestyle=linestyle_driver_1)

        ## Añadir la localización de la curva
        #ax[1].vlines(x=corners['Distance'], ymin=delta_time.min(), ymax=delta_time.max(), linestyles='dotted', colors='grey')

        ## Añadir la linea de referencia y el texto
        ax[1].axhline(0,  color=color_driver2, linestyle=linestyle_driver_2)
        ax[1].set(ylabel=f"Gap to {driver_2} (s)")"""

    # Throttle
    # ----------------------------------------------------------------------------------
    ax[2].plot(
        telemetry_driver_1["DistanceLap"],
        telemetry_driver_1["Throttle"],
        label=driver_1,
        color=color_driver1,
        linewidth=2,
        linestyle=linestyle_driver_1,
    )
    ax[2].plot(
        telemetry_driver_2["DistanceLap"],
        telemetry_driver_2["Throttle"],
        label=driver_2,
        color=color_driver2,
        linewidth=2,
        linestyle=linestyle_driver_2,
    )

    ## Añadir la localización de la curva
    # ax[2].vlines(x=corners['Distance'], ymin=0, ymax=100, linestyles='dotted', colors='grey')

    ## Añadir text labels
    ax[2].set(ylabel="Throttle (%)")

    # Brake
    # ----------------------------------------------------------------------------------
    ax[3].plot(
        telemetry_driver_1["DistanceLap"],
        telemetry_driver_1["Brake"],
        label=driver_1,
        color=color_driver1,
        linewidth=2,
        linestyle=linestyle_driver_1,
    )
    ax[3].plot(
        telemetry_driver_2["DistanceLap"],
        telemetry_driver_2["Brake"],
        label=driver_2,
        color=color_driver2,
        linewidth=2,
        linestyle=linestyle_driver_2,
    )

    ## Añadir la localización de la curva
    # ax[3].vlines(x=corners['Distance'], ymin=0, ymax=1, linestyles='dotted', colors='grey')

    ## Añadir text labels
    ax[3].set(ylabel="Brake")

    # Gear
    # ----------------------------------------------------------------------------------
    ax[4].plot(
        telemetry_driver_1["DistanceLap"],
        telemetry_driver_1["nGear"],
        label=driver_1,
        color=color_driver1,
        linewidth=2,
        linestyle=linestyle_driver_1,
    )
    ax[4].plot(
        telemetry_driver_2["DistanceLap"],
        telemetry_driver_2["nGear"],
        label=driver_2,
        color=color_driver2,
        linewidth=2,
        linestyle=linestyle_driver_2,
    )
    ax[4].set(ylabel="nGear")

    # RPM
    # ----------------------------------------------------------------------------------
    ax[5].plot(
        telemetry_driver_1["DistanceLap"],
        telemetry_driver_1["RPM"],
        label=driver_1,
        color=color_driver1,
        linewidth=2,
        linestyle=linestyle_driver_1,
    )
    ax[5].plot(
        telemetry_driver_2["DistanceLap"],
        telemetry_driver_2["RPM"],
        label=driver_2,
        color=color_driver2,
        linewidth=2,
        linestyle=linestyle_driver_2,
    )
    ax[5].set(ylabel="RPM")

    # DRS
    # ----------------------------------------------------------------------------------
    ax[6].plot(
        telemetry_driver_1["DistanceLap"],
        telemetry_driver_1["DRS"],
        label=driver_1,
        color=color_driver1,
        linewidth=2,
        linestyle=linestyle_driver_1,
    )
    ax[6].plot(
        telemetry_driver_2["DistanceLap"],
        telemetry_driver_2["DRS"],
        label=driver_2,
        color=color_driver2,
        linewidth=2,
        linestyle=linestyle_driver_2,
    )
    ax[6].set(ylabel="DRS")
    ax[6].set(xlabel="Lap distance (m)")

    ## Comprobar el valor de Save
    ## ----------------------------------------------------------------------------------
    if save is True:
        plt.savefig(plot_filename, format="pdf")

    else:
        plt.show()


def rotate(xy, *, angle):
    rot_mat = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    return np.matmul(xy, rot_mat)


class HandlerFancyArrow(HandlerPatch):
    def __init__(self, **kw):
        HandlerPatch.__init__(self, patch_func=self.make_arrow, **kw)

    def make_arrow(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize
    ):
        """
        Crea una flecha con las mismas proporciones que en el mapa
        pero adaptada al recuadro de la leyenda.
        """
        # Centro de la leyenda
        width = orig_handle.get_linewidth()

        center_x = width / 2
        center_y = height / 2

        # Longitud escalada a la caja de la leyenda
        dx = -width * 0.75
        dy = 0

        return FancyArrowPatch(
            posA=(center_x, center_y),
            posB=(center_x + dx, center_y + dy),
            arrowstyle="simple,head_length=36,head_width=14,tail_width=2",
            color=orig_handle.get_facecolor(),
            alpha=orig_handle.get_alpha(),
        )


def create_minisectors_graph(
    telemetry_driver_1,
    telemetry_driver_2,
    fastest_driver_1,
    fastest_driver_2,
    driver_1,
    driver_2,
    event,
    n_minisectors=25,
    plot_wind=False,
    save=True,
):
    # =================== INFO ===========================================
    lap_driver1_info = add_lap_info(fastest_driver_1, event)
    lap_driver2_info = add_lap_info(fastest_driver_2, event)

    # =================== DATA ===========================================
    # 1) Crear el objeto telemetria con los pilotos
    telemetry_driver_1["Driver"] = driver_1
    telemetry_driver_2["Driver"] = driver_2

    # telemetry = telemetry_driver_1.append(telemetry_driver_2)
    telemetry = pd.concat([telemetry_driver_1, telemetry_driver_2], axis=0)

    # 2) Crear los Minisectores
    num_minisectors = n_minisectors

    ## Obtener la distancia maxima de la telemetria, longitud de la vuelta
    total_distance = max(telemetry["Distance"])

    ## Generar la longitud de un minisector
    minisector_length = total_distance / num_minisectors

    ## Generar un vector con las distancias limite de los minisectors
    minisectors_dist = [0]

    for i in range(0, num_minisectors - 1):
        minisectors_dist.append(minisector_length * (i + 1))

    ## Añadimos una columna a la telemetria que indica en que minisector esta
    telemetry["Minisector"] = telemetry["Distance"].apply(
        lambda dist: int((dist // minisector_length) + 1)
    )

    ## Calcular la velocidad media en el minisector
    average_speed = (
        telemetry.groupby(["Minisector", "Driver"])["Speed"].mean().reset_index()
    )

    # 3) Creamos un df con el piloto más rapido en cada minisector
    ## Tomamos el piloto con mayor velocidad
    fastest_driver = average_speed.loc[
        average_speed.groupby(["Minisector"])["Speed"].idxmax()
    ]

    ## Renombramos la columna Driver a Fastest_driver
    fastest_driver = fastest_driver.rename(columns={"Driver": "Fastest_driver"})

    # 4) Juntamos fastest driver per minisector con telemetry
    telemetry = telemetry.merge(fastest_driver, on=["Minisector"])

    ## Ordenamos los valores de telemetria por Distancia
    telemetry = telemetry.sort_values(by=["Distance"])

    ## Convertirmos la columna Fastest_driver a entero
    telemetry.loc[telemetry["Fastest_driver"] == driver_1, "Fastest_driver_int"] = 1
    telemetry.loc[telemetry["Fastest_driver"] == driver_2, "Fastest_driver_int"] = 2

    # 5) Obtenemos el porcentaje de vuelta de cada piloto en que es más rápido
    count_minisectors_driver1 = sum(fastest_driver["Fastest_driver"] == driver_1)
    porcentaje_driver1 = (count_minisectors_driver1 / num_minisectors) * 100

    count_minisectors_driver2 = sum(fastest_driver["Fastest_driver"] == driver_2)
    porcentaje_driver2 = (count_minisectors_driver2 / num_minisectors) * 100

    # 6) PLOT
    # ----------------------------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=[10, 10], width_ratios=[6, 1])
    plt.subplots_adjust(top=0.90, bottom=0.03, wspace=0.01, left=0.03, right=0.95)

    ## ============= Circuito =========
    x = np.array(telemetry["X"].values)
    y = np.array(telemetry["Y"].values)

    # Create una malla de puntos
    x_min = np.min(x)
    y_min = np.min(y)
    x_max = np.max(x)
    y_max = np.max(y)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fastest_driver_array = telemetry["Fastest_driver_int"].to_numpy().astype(float)

    # Definir un colormap personalizado con dos colores
    team_driver_1 = fastest_driver_1["Team"]
    team_driver_2 = fastest_driver_2["Team"]
    color_1 = ff1.plotting.get_team_color(team_driver_1, event)
    color_2 = ff1.plotting.get_team_color(team_driver_2, event)
    colors = ListedColormap([color_1, color_2])
    driver1_patch = mpatches.Patch(
        color=color_1, label=f"{driver_1} - {format_time(fastest_driver_1.LapTime)}"
    )
    driver2_patch = mpatches.Patch(
        color=color_2, label=f"{driver_2} - {format_time(fastest_driver_2.LapTime)}"
    )
    # colors = ListedColormap([ff1.plotting.get_team_color(team_driver_1), 'yellow'])

    ### Crear la colección de líneas y asignar colores
    lc_comp = LineCollection(segments, cmap=colors, norm=plt.Normalize(1, 2))
    lc_comp.set_array(fastest_driver_array)
    lc_comp.set_linewidth(10)

    ax[0].add_collection(lc_comp)

    ### Después de crear ax[0]
    pos = ax[0].get_position()  # posición actual [left, bottom, width, height]
    ax[0].set_position(
        [pos.x0, pos.y0, pos.width, pos.height * 0.85]
    )  # reduce altura un 10%

    ### Añadir circulo con la curva y su numero
    circuit_info = event.get_circuit_info()
    corners = circuit_info.corners
    offset_vector = [500, 0]  # offset length is chosen arbitrarily to 'look good'

    for _, corner in corners.iterrows():
        # Create a string from corner number and letter
        txt = f"{corner['Number']}{corner['Letter']}"

        # Convert the angle from degrees to radian.
        offset_angle = corner["Angle"] / 180 * np.pi

        # Rotate the offset vector so that it points sideways from the track.
        offset_x, offset_y = rotate(offset_vector, angle=offset_angle)

        # Add the offset to the position of the corner
        text_x = corner["X"] + offset_x
        text_y = corner["Y"] + offset_y

        # Rotate the center of the corner equivalently to the rest of the track map
        track_x, track_y = [corner["X"], corner["Y"]]

        # Draw a circle next to the track.
        ax[0].scatter(text_x, text_y, color="grey", s=500)

        # Draw a line from the track to this circle.
        ax[0].plot([track_x, text_x], [track_y, text_y], color="grey", linewidth=5)

        # Finally, print the corner number inside the circle.
        ax[0].text(
            text_x,
            text_y,
            txt,
            va="center_baseline",
            ha="center",
            fontsize=15,
            color="white",
        )

    ### Añadir logo
    x_logo = 0.49
    y_logo = 1.01
    zoom = 0.087
    add_team_logo(ax=ax[0], x=x_logo, y=y_logo, zoom=zoom)

    ### Añadir dirección viento
    if plot_wind:
        # Extraer la dirección del viento
        wind_dir_deg = (
            lap_driver1_info["WindDirection"].iloc[0]
            + lap_driver2_info["WindDirection"].iloc[0]
        ) / 2
        wind_dir_deg = (90 - wind_dir_deg) % 360
        wind_dir_rad = np.radians(wind_dir_deg)
        u_unit = np.cos(wind_dir_rad)
        v_unit = np.sin(wind_dir_rad)

        x_wind, y_wind = np.meshgrid(
            np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10)
        )
        u = np.ones_like(x_wind) * u_unit  # wind componente x
        v = np.ones_like(y_wind) * v_unit  # wind componente y

        # Plot wind
        factor = 100
        for x0, y0, dx, dy in zip(
            x_wind.flatten(), y_wind.flatten(), u.flatten(), v.flatten()
        ):
            arrow = FancyArrowPatch(
                posA=(x0, y0),
                posB=(x0 + dx * factor, y0 + dy * factor),
                arrowstyle="simple,head_length=30,head_width=12,tail_width=2",
                color="royalblue",
                alpha=0.6,
                zorder=2,
            )
            ax[0].add_patch(arrow)

        ### Añadir leyenda
        arrow_style = dict(
            width=0.1,
            head_width=0.15,
            head_length=0.25,
            length_includes_head=True,
            color="royalblue",
            alpha=0.6,
        )

        flecha_viento = mpatches.FancyArrow(0, 0, -1, 0, **arrow_style)
        ax[0].legend(
            handles=[driver1_patch, driver2_patch, flecha_viento],
            labels=[
                f"{driver_1} - {format_time(fastest_driver_1.LapTime)}",
                f"{driver_2} - {format_time(fastest_driver_2.LapTime)}",
                f"Viento ({wind_dir_deg})º",
            ],
            handler_map={mpatches.FancyArrow: HandlerFancyArrow()},
            loc="upper right",
            bbox_to_anchor=(0.94, 1.19),
            frameon=False,
            fontsize=18,
        )

    else:
        ### Añadir leyenda
        ax[0].legend(
            handles=[driver1_patch, driver2_patch],
            loc="upper right",
            bbox_to_anchor=(0.94, 1.19),  # esquina superior derecha del eje
            frameon=False,
            fontsize=19,
            borderpad=0.4,
        )

    # ax[0].axis('equal')
    ax[0].axis("off")
    ax[0].tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    # ======== COLOR BAR ======================================
    ax[1].bar(
        0, porcentaje_driver1 / 100, bottom=0, width=0.1, color=color_1, alpha=1.0
    )

    ax[1].bar(
        0,
        porcentaje_driver2 / 100,
        bottom=porcentaje_driver1 / 100,
        width=0.1,
        color=color_2,
        alpha=1.0,
    )

    txt1 = driver_1 + f"\n{porcentaje_driver1:.1f}%"
    text_y1 = porcentaje_driver1 / 200
    text_y2 = (1.0 + (porcentaje_driver1 / 100)) / 2
    txt2 = driver_2 + f"\n{porcentaje_driver2:.1f}%"
    ax[1].text(
        0.15,
        text_y1,
        txt1,
        va="center_baseline",
        ha="center",
        fontsize=16,
        color="white",
    )
    ax[1].text(
        0.15,
        text_y2,
        txt2,
        va="center_baseline",
        ha="center",
        fontsize=16,
        color="white",
    )
    ax[1].set_xlim([-0.1, 0.2])
    ax[1].set_ylim([0.0, 1.0])
    ax[1].axis("off")

    ## Crear el titulo del gráfico
    ## ---------------------------
    plt.suptitle(
        f"{driver_1} vs {driver_2} - Minisectores\n"
        # f"{event.event.year} {event.event.EventName} - {event.name}",
        f"{event.event.year} {event.event.EventName} - Q3",
        fontsize=24,
        weight="bold",
        y=0.99,
    )

    ## ========== SAVE ===============================================
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        plot_title = f"Minisectores {driver_1} vs {driver_2}"
        plot_filename = plot_title.replace(" ", "_") + ".png"

        eventname = event.event.EventName.split(" ")[0]
        path = f"{event.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, event.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar figura
        plot_filename = os.path.join(path, plot_filename)
        plt.savefig(plot_filename, format="png")
    else:
        plt.show()


def graph_long_acc(
    telemetry1,
    telemetry2,
    fastest_driver_1,
    fastest_driver_2,
    driver_1,
    driver_2,
    session,
    save=True,
):
    # Cogemos solo los datos de un tipo car (car)
    mask1 = telemetry1["Source"] == "car"
    mask2 = telemetry2["Source"] == "car"
    telemetry1 = telemetry1.loc[mask1]
    telemetry2 = telemetry2.loc[mask2]

    # Aplicamos la función longacc y lateralacc
    telemetry1 = calcula_longacc(telemetry1)
    telemetry2 = calcula_longacc(telemetry2)

    # Calcular los colores del equipo
    team_driver_1 = fastest_driver_1["Team"]
    team_driver_2 = fastest_driver_2["Team"]

    # Visualizar el resultado
    plt.figure(figsize=(9, 3))
    plt.plot(
        telemetry1["Distance"],
        telemetry1["Long Acc"],
        label=driver_1,
        color=ff1.plotting.get_team_color(team_driver_1, session),
        linewidth=2,
    )
    plt.plot(
        telemetry2["Distance"],
        telemetry2["Long Acc"],
        label=driver_2,
        color=ff1.plotting.get_team_color(team_driver_2, session),
        linewidth=2,
    )
    plt.legend()
    plt.title("Aceleración Longitudinal (g)", loc="left")
    plt.xlabel("Distancia (m)")
    plt.ylabel("Aceleración (g)")

    if save:
        # Construir el titulo de la figura
        plot_filename = "LongAcc_{}_{}.pdf".format(driver_1, driver_2)
        plot_filename = "./outputs/" + plot_filename

        # Guardar figura
        plt.savefig(plot_filename, format="pdf")
    else:
        plt.show()


def graph_lat_acc(
    telemetry1,
    telemetry2,
    fastest_driver_1,
    fastest_driver_2,
    driver_1,
    driver_2,
    session,
    save=True,
):
    # Cogemos solo los datos de un tipo pos (pos)
    mask1 = telemetry1["Source"] == "car"
    mask2 = telemetry2["Source"] == "car"
    telemetry1 = telemetry1.loc[mask1]
    telemetry2 = telemetry2.loc[mask2]

    # Aplicamos la función longacc y lateralacc
    telemetry1 = calcula_lateralacc(telemetry1)
    telemetry2 = calcula_lateralacc(telemetry2)

    # Calcular los colores del equipo
    team_driver_1 = fastest_driver_1["Team"]
    team_driver_2 = fastest_driver_2["Team"]

    # Visualizar el resultado
    plt.figure(figsize=(9, 3))
    plt.plot(
        telemetry1["Distance"],
        telemetry1["Lat Acc"],
        label=driver_1,
        color=ff1.plotting.get_team_color(team_driver_1, session),
        linewidth=2,
    )
    plt.plot(
        telemetry2["Distance"],
        telemetry2["Lat Acc"],
        label=driver_2,
        color=ff1.plotting.get_team_color(team_driver_2, session),
        linewidth=2,
    )
    plt.legend()
    plt.title("Aceleración Lateral (g)", loc="left")
    plt.xlabel("Distancia (m)")
    plt.ylabel("Aceleración (g)")

    if save:
        # Construir el titulo de la figura
        plot_filename = f"LatAcc_graph_{driver_1}_{driver_2}.pdf"
        plot_filename = "./outputs/" + plot_filename

        # Guardar figura
        plt.savefig(plot_filename, format="pdf")
    else:
        plt.show()


def graph_comparaison_firststraight(lapfastest, lapanterior, corners, save=True):
    # Calcular delta entre las vueltas
    delta_time, ref_tel, compare_tel = utils.delta_time(lapfastest, lapanterior)

    # Obtener telemetrias
    telemetry_fastest = lapfastest.get_telemetry().add_distance()
    telemetry_anterior = lapanterior.get_telemetry().add_distance()

    team_driver_1 = lapfastest["Team"]
    team_driver_2 = lapanterior["Team"]
    color_driver1 = ff1.plotting.get_team_color(team_driver_1, session)
    color_driver2 = ff1.plotting.get_team_color(team_driver_2, session)

    ## Determinar indice inicio de vuelta
    index_telemetryfastest = telemetry_fastest.index.to_list()
    index_ini = index_telemetryfastest[0]

    # Determinar indice curva 2
    ## Creamos una columna llamada distancia a la curva en valor absoluto
    corner_distance = corners.loc[corners["Number"] == 2, "Distance"].values[0]
    telemetry_fastest["Distancia_curva"] = abs(
        telemetry_fastest["Distance"] - corner_distance
    )
    index_secondcorner = telemetry_fastest["Distancia_curva"].idxmin()

    ## Subset curva 2
    subsetcurva2_fastest = telemetry_fastest.loc[index_ini:index_secondcorner,].copy()

    ## Buscar en el subconjunto el indice donde levanta,
    mask = subsetcurva2_fastest["Throttle"] < 99
    aux = subsetcurva2_fastest.loc[mask,].index.to_list()
    indice_fincurva2 = aux[0]

    subsetfinal2_fastest = subsetcurva2_fastest.loc[:indice_fincurva2,].copy()
    dist_final = subsetfinal2_fastest.loc[indice_fincurva2, "Distance"]

    ## Buscamos el indice equivalente en la vuelta anterior (mas cercano en distancia)
    index_ini2 = telemetry_anterior.index.to_list()[0]
    telemetry_anterior["Distancia_punto"] = abs(
        telemetry_anterior["Distance"] - dist_final
    )
    index2_final = telemetry_anterior["Distancia_punto"].idxmin()
    subsetfinal2_anterior = telemetry_anterior.loc[index_ini2:index2_final,].copy()

    ## Obtenemos la delta de tiempo en la curva 2
    nrow2 = subsetfinal2_anterior.shape[0]
    # delta_frenada_curva2 = delta_time[nrow2]

    index = subsetfinal2_anterior.index.to_list()
    distance_curva2 = subsetfinal2_anterior.loc[index[nrow2 - 1], "Distance"]
    ref_tel["Distance_ini"] = abs(ref_tel["Distance"] - distance_curva2)
    index_curva2 = ref_tel["Distance_ini"].idxmin()

    delta_frenada_curva2 = delta_time[index_curva2]
    print("Delta en la Frenada curva 2 = {:.3f}s".format(delta_frenada_curva2))

    # Determinar indice curva 4 ---------------------------------------
    ## Creamos una columna llamada distancia a la curva en valor absoluto
    corner_distance4 = corners.loc[corners["Number"] == 4, "Distance"].values[0]
    telemetry_fastest["Distancia_curva4"] = abs(
        telemetry_fastest["Distance"] - corner_distance4
    )
    index_fourthcorner1 = telemetry_fastest["Distancia_curva4"].idxmin()

    ## Subset curva 4 (fastest)
    subsetfinal4_fastest = telemetry_fastest.loc[index_ini:index_fourthcorner1,].copy()

    ## Subset curva 4 (anterior
    telemetry_anterior["Distancia_curva4"] = abs(
        telemetry_anterior["Distance"] - corner_distance4
    )
    index_fourthcorner2 = telemetry_anterior["Distancia_curva4"].idxmin()
    subsetfinal4_anterior = telemetry_anterior.loc[
        index_ini2:index_fourthcorner2,
    ].copy()

    ## Obtenemos la distancia en la curva 4
    nrow4 = subsetfinal4_anterior.shape[0]
    index = subsetfinal4_anterior.index.to_list()
    distance_curva4 = subsetfinal4_anterior.loc[index[nrow4 - 1], "Distance"]
    ref_tel["Distance_fin"] = abs(ref_tel["Distance"] - distance_curva4)
    index_curva4 = ref_tel["Distance_fin"].idxmin()
    delta_frenada_curva4 = delta_time[index_curva4]
    print("Delta en la curva 4 = {:.3f}s".format(delta_frenada_curva4))

    # -------------------- PLOT ------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})

    # 0. Speed
    ax[0].plot(
        subsetfinal4_fastest["Distance"],
        subsetfinal4_fastest["Speed"],
        label="Verstappen",
        color=color_driver1,
        alpha=1.0,
        linewidth=1,
        linestyle="-",
    )
    ax[0].plot(
        subsetfinal4_anterior["Distance"],
        subsetfinal4_anterior["Speed"],
        label="Norris",
        color=color_driver2,
        alpha=1.0,
        linewidth=1,
        linestyle="-",
    )

    ax[0].set_ylabel("Speed (km/h)")
    ax[0].legend(loc="lower left", fontsize="8")
    ax[0].title.set_text("Comparativa con Norris - Hasta curva 4")

    ## Añadir la localización de la curva y numero
    vmin = subsetfinal4_anterior["Speed"].min()
    vmax = subsetfinal4_fastest["Speed"].max()
    subset_corners = corners.loc[corners["Number"] <= 3,].copy()
    ax[0].vlines(
        x=subset_corners["Distance"],
        ymin=vmin - 20,
        ymax=vmax,
        linestyles="dotted",
        colors="grey",
    )

    ## Añadir el número de curva debajo de las lineas
    for _, corner in subset_corners.iterrows():
        txt = f"{corner['Number']}{corner['Letter']}"
        ax[0].text(
            corner["Distance"],
            vmin - 30,
            txt,
            va="center_baseline",
            ha="center",
            size="small",
        )

    # 1. Delta
    ## La delta ha quedado positiva, luego esta bien calculada
    # ax[1].set_ylim((0.0, 1.0))
    index = subsetfinal4_anterior.index.to_list()
    distance_fin = subsetfinal4_anterior.loc[index[-1], "Distance"]

    ref_tel["Distance_final"] = abs(ref_tel["Distance"] - distance_fin)
    index_ini = ref_tel.index.to_list()[0]
    index_fin = ref_tel["Distance_final"].idxmin()

    ax[1].plot(
        ref_tel.loc[index_ini:index_fin, "Distance"],
        delta_time[: index_fin + 1],
        color=color_driver2,
        alpha=1.0,
        linewidth=1,
        linestyle="-",
    )

    ## Anotaciones del delta - curva2
    index = subsetfinal2_anterior.index.to_list()
    distance_curva2 = subsetfinal2_anterior.loc[index[-1], "Distance"]

    ref_tel["Distance_curva2"] = abs(ref_tel["Distance"] - distance_curva2)
    index_min = ref_tel["Distance_curva2"].idxmin()

    xp = ref_tel.loc[index_min, "Distance"]
    yp = delta_frenada_curva2

    ax[1].plot([xp], [yp], "o", color="lightblue", alpha=1.0, linewidth=0.25)
    ax[1].annotate(
        f"Delta en la frenada = {round(delta_frenada_curva2, 3)}s",
        (xp, yp),
        (xp, yp * 0.5),
        arrowprops=dict(facecolor="lightblue", headwidth=8, width=2),
        fontsize=8,
    )

    ## Anotaciones del delta - curva4
    xp = ref_tel.loc[index_fin, "Distance"]
    yp = delta_frenada_curva4

    ax[1].plot([xp], [yp], "o", color="orange", alpha=1.0, linewidth=0.5)
    ax[1].annotate(
        f"Delta en la curva4 = {round(delta_frenada_curva4, 3)}s",
        (xp, yp),
        (xp, yp + 0.05),
        arrowprops=dict(facecolor="orange", headwidth=8, width=2),
        fontsize=8,
        ha="right",
    )

    ## Añadir la linea de referencia y el texto
    ax[1].axhline(0, color=color_driver1, alpha=1.0, linestyle="-")
    ax[1].set(ylabel=f"Gap (s)")
    ax[1].set_xlabel("Lap distance (m)")

    ## Añadir la linea de las curvas
    ax[1].vlines(
        x=subset_corners["Distance"],
        ymin=delta_time[: nrow4 - 1].min(),
        ymax=delta_time[: nrow4 - 1].max(),
        linestyles="dotted",
        colors="grey",
    )

    # Comprobar el valor de Save
    # ----------------------------------------------------------------------------------
    if save is True:
        plot_filename = "Comparativa_Norris_inicial.pdf"
        plt.savefig(plot_filename, format="pdf")

    else:
        plt.show()


def graph_comparaison_finalpart(lapfastest, lapanterior, corners, session, save=True):
    # Calcular delta entre las vueltas
    delta_time, ref_tel, compare_tel = utils.delta_time(lapfastest, lapanterior)

    # Obtener telemetrias
    telemetry_fastest = lapfastest.get_telemetry().add_distance()
    telemetry_anterior = lapanterior.get_telemetry().add_distance()

    team_driver_1 = lapfastest["Team"]
    team_driver_2 = lapanterior["Team"]
    color_driver1 = ff1.plotting.get_team_color(team_driver_1, session)
    color_driver2 = ff1.plotting.get_team_color(team_driver_2, session)

    # Crear el subset del fastestlap
    ## Determinar indice curva 16
    corner_distance16 = corners.loc[corners["Number"] == 16, "Distance"].values[0]
    telemetry_fastest["Distancia_curva16"] = abs(
        telemetry_fastest["Distance"] - corner_distance16
    )
    index_16corner1 = telemetry_fastest["Distancia_curva16"].idxmin()

    subset_fastest = telemetry_fastest.loc[index_16corner1:,].copy()

    # Crear el subset del anteriorlap
    ## Determinar indice curva 16
    telemetry_anterior["Distancia_curva16"] = abs(
        telemetry_anterior["Distance"] - corner_distance16
    )
    index_16corner2 = telemetry_anterior["Distancia_curva16"].idxmin()

    subset_anterior = telemetry_anterior.loc[index_16corner2:,].copy()

    # -------------------- PLOT ------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})

    # 0. Speed
    ax[0].plot(
        subset_fastest["Distance"],
        subset_fastest["Speed"],
        label="Verstappen",
        color=color_driver1,
        alpha=1.0,
        linewidth=1,
        linestyle="-",
    )
    ax[0].plot(
        subset_anterior["Distance"],
        subset_anterior["Speed"],
        label="Norris",
        color=color_driver2,
        alpha=1.0,
        linewidth=1,
        linestyle="-",
    )

    ax[0].set_ylabel("Speed (km/h)")
    ax[0].legend(loc="lower right", fontsize="8")
    ax[0].title.set_text("Parte Final contra Norris")

    # 1. Delta
    ## La delta ha quedado positiva, luego esta bien calculada
    index = subset_anterior.index.to_list()
    distance_ini = subset_anterior.loc[index[0], "Distance"]

    ref_tel["Distance_ini"] = abs(ref_tel["Distance"] - distance_ini)
    index_ini = ref_tel["Distance_ini"].idxmin()
    index_fin = ref_tel.index.to_list()[-1]

    ax[1].plot(
        ref_tel.loc[index_ini:index_fin, "Distance"],
        delta_time[index_ini : index_fin + 1],
        color=color_driver2,
        alpha=1.0,
        linewidth=1,
        linestyle="-",
    )

    ## Añadir linea de referencia
    ax[1].axhline(0, color=color_driver1, alpha=1.0, linestyle="-")

    """## Añadir flechas de inicio
    dist_ini = ref_tel.loc[index_ini, 'Distance']
    dist_fin = ref_tel.loc[index_fin, 'Distance']
    delta_rebufo = 0.090
    delta_ini = delta_time[index_ini]
    delta_fin = delta_time[index_fin]
    offset = 0.01

    arrow1 = FancyArrowPatch(posA=[dist_ini, 0.0-offset], posB=[dist_ini, delta_rebufo+offset], color='orange',
                             arrowstyle='<->', mutation_scale=10)
    arrow2 = FancyArrowPatch(posA=[dist_ini, delta_rebufo-offset], posB=[dist_ini, delta_ini+offset], color='purple',
                             arrowstyle='<->', mutation_scale=10)
    arrow3 = FancyArrowPatch(posA=[dist_fin, 0.0 - offset], posB=[dist_fin, delta_fin + offset],
                             color='yellow',arrowstyle='<->', mutation_scale=10)
    arrow4 = FancyArrowPatch(posA=[dist_fin+10, 0.0 - offset], posB=[dist_fin+10, delta_rebufo + offset],
                             color='orange', arrowstyle='<->', mutation_scale=10)
    arrow5 = FancyArrowPatch(posA=[dist_fin+10, delta_rebufo - offset], posB=[dist_fin+10, delta_fin + offset],
                             color='purple', arrowstyle='<->', mutation_scale=10)

    ax[1].add_patch(arrow1)
    ax[1].add_patch(arrow2)
    ax[1].add_patch(arrow3)
    ax[1].add_patch(arrow4)
    ax[1].add_patch(arrow5)

    ## Añadir los textos
    ax[1].text(dist_ini+10, delta_rebufo/2, 'Delta Vuelta = {:.3f}s'.format(delta_rebufo),
                color='orange', fontsize=8)
    ax[1].text(dist_ini+10, (delta_ini+delta_rebufo)/2, 'Delta Rebufo = {:.3f}s'.format(delta_ini-delta_rebufo),
               color='purple', fontsize=8)
    ax[1].text(dist_fin - 200, delta_fin/2, 'Delta Final = {:.3f}s'.format(delta_fin),
               color='yellow', fontsize=8)

    ax[1].annotate("Error de {:.3f}s".format(delta_fin-delta_ini), (dist_fin, delta_fin),
                   (dist_fin - 250, delta_fin + 0.02),
                   arrowprops=dict(arrowstyle='->', color='white'), fontsize=8, color='white')"""

    # Flechas
    delta_ini = delta_time[index_ini]
    delta_fin = delta_time[index_fin]

    xp = ref_tel.loc[index_ini, "Distance"]
    yp = delta_ini

    ax[1].plot([xp], [yp], "o", color="orange", alpha=1.0, linewidth=0.5)
    ax[1].annotate(
        f"Delta Inicial = {round(delta_ini, 3)}s",
        (xp, yp),
        (xp, yp * 0.5),
        arrowprops=dict(facecolor="orange", headwidth=8, width=2),
        fontsize=8,
    )

    xp = ref_tel.loc[index_fin, "Distance"]
    yp = delta_fin

    ax[1].plot([xp], [yp], "o", color="lightblue", alpha=1.0, linewidth=0.5)
    ax[1].annotate(
        f"Delta final = {round(delta_fin, 3)}s",
        (xp, yp),
        (xp, yp + 0.1),
        arrowprops=dict(facecolor="lightblue", headwidth=8, width=2),
        fontsize=8,
        ha="right",
    )

    ## Añadir ejes
    ax[1].set(ylabel=f"Gap (s)")
    ax[1].set_xlabel("Lap distance (m)")

    # Comprobar el valor de Save
    # ----------------------------------------------------------------------------------
    if save is True:
        plot_filename = "Comparativa_Norris_final.pdf"
        plt.savefig(plot_filename, format="pdf")

    else:
        plt.show()


def obten_subset(
    telemetry, corners, corner_start, corner_stop, corner_start_type, corner_stop_type
):
    # Copy and indexes
    telemetrycopy = telemetry.copy()
    indexes = telemetrycopy.index.to_list()

    # Check corner_start_type
    if corner_start_type == "Inicio":
        index_inicial = indexes[0]

    elif corner_start_type == "FullThrottle":
        # Defino subset primordial
        corner_distance_inicial = corners.loc[
            corners["Number"] == corner_start, "Distance"
        ].values[0]
        corner_distance_final = corners.loc[
            corners["Number"] == corner_stop, "Distance"
        ].values[0]

        mask_ini = telemetrycopy["Distance"] >= corner_distance_inicial
        mask_fin = telemetrycopy["Distance"] <= corner_distance_final

        subset = telemetrycopy.loc[mask_ini & mask_fin,]
        # Busco el inicio de acelerador a fondo
        mask_fullthrottle = subset["Throttle"] >= 99
        index_inicial = subset.loc[mask_fullthrottle,].index.to_list()[0]

    else:
        # Defino subset primordial
        corner_distance_inicial = corners.loc[
            corners["Number"] == corner_start, "Distance"
        ].values[0]
        corner_distance_final = corners.loc[
            corners["Number"] == corner_stop, "Distance"
        ].values[0]

        mask_ini = telemetrycopy["Distance"] >= corner_distance_inicial
        mask_fin = telemetrycopy["Distance"] <= corner_distance_final

        subset = telemetrycopy.loc[mask_ini & mask_fin,]

        # Busco el indice inicial
        index_inicial = subset.index.to_list()[0]

    # Check corner_stop_type
    if corner_stop_type == "Final":
        index_final = indexes[-1]
    elif corner_stop_type == "Levantada":
        subset_aux = telemetrycopy.loc[index_inicial:,]

        corner_distance_final = corners.loc[
            corners["Number"] == corner_stop, "Distance"
        ].values[0]
        mask_fin = subset_aux["Distance"] <= corner_distance_final

        subset = subset_aux.loc[mask_fin,]

        # Busco el inicio de acelerador a fondo
        mask_levantada = subset["Throttle"] < 99
        index_final = subset.loc[mask_levantada,].index.to_list()[0]

    # Creo subset
    subset_final = telemetrycopy.loc[index_inicial:index_final,].copy()
    return subset_final


def plot_one_recta(
    subset_driver1,
    subset_driver2,
    driver1,
    driver2,
    color_driver1,
    color_driver2,
    corners,
    delta_time,
    ref_tel,
    save=False,
):
    fig, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})

    ax[0].plot(
        subset_driver1["Distance"],
        subset_driver1["Speed"],
        label=driver1,
        color=color_driver1,
    )
    ax[0].plot(
        subset_driver2["Distance"],
        subset_driver2["Speed"],
        label=driver2,
        color=color_driver2,
    )

    ax[0].set_ylabel("Speed (km/h)")
    ax[0].legend(loc="lower right")
    ax[0].set_title(
        f"Recta entre curva {corners[0]}<->{corners[1]}",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )

    ### Recta DRS delta
    index = subset_driver2.index.to_list()
    distance_ini = subset_driver2.loc[index[0], "Distance"]
    distance_fin = subset_driver2.loc[index[-1], "Distance"]

    ref_tel["Distance_ini"] = abs(ref_tel["Distance"] - distance_ini)
    ref_tel["Distance_fin"] = abs(ref_tel["Distance"] - distance_fin)
    index_ini = ref_tel["Distance_ini"].idxmin()
    index_fin = ref_tel["Distance_fin"].idxmin()

    ax[1].plot(
        ref_tel.loc[index_ini:index_fin, "Distance"],
        delta_time[index_ini : index_fin + 1],
        color=color_driver2,
        alpha=1.0,
        linewidth=1,
        linestyle="-",
    )
    ax[1].set_xlabel("Lap distance (m)")
    ax[1].set_ylabel("Gap (s)")
    ax[1].axhline(0, color=color_driver1, alpha=1.0, linestyle="-")

    # Flechas
    delta_ini = delta_time[index_ini]
    delta_fin = delta_time[index_fin]

    xp = ref_tel.loc[index_ini, "Distance"]
    yp = delta_ini

    ax[1].plot([xp], [yp], "o", color="orange", alpha=1.0, linewidth=0.5)
    ax[1].annotate(
        f"Delta en la curva{corners[0]} = {round(delta_ini, 3)}s",
        (xp, yp),
        (xp, yp * 0.5),
        arrowprops=dict(facecolor="orange", headwidth=8, width=2),
        fontsize=8,
    )

    xp = ref_tel.loc[index_fin, "Distance"]
    yp = delta_fin

    ax[1].plot([xp], [yp], "o", color="lightblue", alpha=1.0, linewidth=0.5)
    ax[1].annotate(
        f"Delta en la curva{corners[1]} = {round(delta_fin, 3)}s",
        (xp, yp),
        (xp, yp * 0.5),
        arrowprops=dict(facecolor="lightblue", headwidth=8, width=2),
        fontsize=8,
        ha="right",
    )

    if save is True:
        plot_filename = f"Recta entre curva {corners[0]}<->{corners[1]}.png"
        plt.savefig(plot_filename, format="png")

    else:
        plt.show()


def rectas_imola(lap_driver1, lap_driver2, corners, driver1, driver2, session, save):
    # 1) Obterner telemetria
    telemetry_driver1 = lap_driver1.get_telemetry().add_distance()
    telemetry_driver2 = lap_driver2.get_telemetry().add_distance()

    team_driver_1 = lap_driver1["Team"]
    team_driver_2 = lap_driver2["Team"]
    color_driver1 = ff1.plotting.get_team_color(team_driver_1, session)
    color_driver2 = ff1.plotting.get_team_color(team_driver_2, session)

    # 2) Obtener delta time
    delta_time, ref_tel, compare_tel = utils.delta_time(lap_driver1, lap_driver2)

    # 3) RECTAS DRS (entre 0m y la levantada de la curva 2)
    subset_rectadrs_driver1 = obten_subset(
        telemetry=telemetry_driver1,
        corners=corners,
        corner_start=0,
        corner_stop=2,
        corner_start_type="Inicio",
        corner_stop_type="Levantada",
    )
    subset_rectadrs_driver2 = obten_subset(
        telemetry=telemetry_driver2,
        corners=corners,
        corner_start=0,
        corner_stop=2,
        corner_start_type="Inicio",
        corner_stop_type="Levantada",
    )

    ## Guardar delta
    nrow2 = subset_rectadrs_driver2.shape[0]
    index = subset_rectadrs_driver2.index.to_list()
    distance_curva2 = subset_rectadrs_driver2.loc[index[nrow2 - 1], "Distance"]
    ref_tel["Distance_ini"] = abs(ref_tel["Distance"] - distance_curva2)
    index_curva2 = ref_tel["Distance_ini"].idxmin()

    delta_frenada_curva2 = delta_time[index_curva2]

    # 4) RECTAS SIN DRS
    ## Recta1 (entre la curva 4 y curva 5)
    subset_recta1_driver1 = obten_subset(
        telemetry=telemetry_driver1,
        corners=corners,
        corner_start=4,
        corner_stop=5,
        corner_start_type="Curva",
        corner_stop_type="Levantada",
    )
    subset_recta1_driver2 = obten_subset(
        telemetry=telemetry_driver2,
        corners=corners,
        corner_start=4,
        corner_stop=5,
        corner_start_type="Curva",
        corner_stop_type="Levantada",
    )

    ## Recta 2 (entre la salida de la 6 -fullthrottle y levantada de la 7)
    subset_recta2_driver1 = obten_subset(
        telemetry=telemetry_driver1,
        corners=corners,
        corner_start=6,
        corner_stop=7,
        corner_start_type="FullThrottle",
        corner_stop_type="Levantada",
    )
    subset_recta2_driver2 = obten_subset(
        telemetry=telemetry_driver2,
        corners=corners,
        corner_start=6,
        corner_stop=7,
        corner_start_type="FullThrottle",
        corner_stop_type="Levantada",
    )

    ## Recta 3 (entre la salida de la 13 -fullthrottle y levantada de la 14)
    subset_recta3_driver1 = obten_subset(
        telemetry=telemetry_driver1,
        corners=corners,
        corner_start=13,
        corner_stop=14,
        corner_start_type="FullThrottle",
        corner_stop_type="Levantada",
    )
    subset_recta3_driver2 = obten_subset(
        telemetry=telemetry_driver2,
        corners=corners,
        corner_start=13,
        corner_stop=14,
        corner_start_type="FullThrottle",
        corner_stop_type="Levantada",
    )

    ## Recta 4 (entre la salida de la 15 -fullthrottle y levantada de la 17)
    subset_recta4_driver1 = obten_subset(
        telemetry=telemetry_driver1,
        corners=corners,
        corner_start=15,
        corner_stop=17,
        corner_start_type="FullThrottle",
        corner_stop_type="Levantada",
    )
    subset_recta4_driver2 = obten_subset(
        telemetry=telemetry_driver2,
        corners=corners,
        corner_start=15,
        corner_stop=17,
        corner_start_type="FullThrottle",
        corner_stop_type="Levantada",
    )

    # 5) PLOTS
    corner_numbers = [4, 5]
    plot_one_recta(
        subset_recta1_driver1,
        subset_recta1_driver2,
        driver1,
        driver2,
        color_driver1,
        color_driver2,
        corner_numbers,
        delta_time,
        ref_tel,
        save=save,
    )

    corner_numbers = [6, 7]
    plot_one_recta(
        subset_recta2_driver1,
        subset_recta2_driver2,
        driver1,
        driver2,
        color_driver1,
        color_driver2,
        corner_numbers,
        delta_time,
        ref_tel,
        save=save,
    )

    corner_numbers = [13, 14]
    plot_one_recta(
        subset_recta3_driver1,
        subset_recta3_driver2,
        driver1,
        driver2,
        color_driver1,
        color_driver2,
        corner_numbers,
        delta_time,
        ref_tel,
        save=save,
    )

    corner_numbers = [15, 17]
    plot_one_recta(
        subset_recta4_driver1,
        subset_recta4_driver2,
        driver1,
        driver2,
        color_driver1,
        color_driver2,
        corner_numbers,
        delta_time,
        ref_tel,
        save=save,
    )


def obten_subset_curva(telemetry, corners, corner_num):
    # Parte Inicial
    corner_distance_anterior = corners.loc[
        corners["Number"] == corner_num - 1, "Distance"
    ].values[0]
    corner_distance = corners.loc[corners["Number"] == corner_num, "Distance"].values[0]

    mask1 = telemetry["Distance"] >= corner_distance_anterior
    mask2 = telemetry["Distance"] <= corner_distance
    subset_inicial = telemetry.loc[mask1 & mask2,]

    maskfull = subset_inicial["Throttle"] >= 99
    index_ini = subset_inicial.loc[maskfull,].index.to_list()[-1]
    index_ini = index_ini - 5

    # Parte final
    corner_distance_siguiente = corners.loc[
        corners["Number"] == corner_num + 1, "Distance"
    ].values[0]

    mask1 = telemetry["Distance"] >= corner_distance
    mask2 = telemetry["Distance"] <= corner_distance_siguiente
    subset_final = telemetry.loc[mask1 & mask2,]

    maskfull = subset_final["Throttle"] >= 99
    index_fin = subset_final.loc[maskfull,].index.to_list()[0]
    index_fin = index_fin + 5

    # Fenerar subset
    subset = telemetry.loc[index_ini:index_fin,].copy()

    return subset


def obten_subset_chicane(telemetry, corners, corner_num):
    # Parte Inicial
    corner_distance_anterior = corners.loc[
        corners["Number"] == corner_num - 1, "Distance"
    ].values[0]
    corner_distance = corners.loc[corners["Number"] == corner_num, "Distance"].values[0]

    mask1 = telemetry["Distance"] >= corner_distance_anterior
    mask2 = telemetry["Distance"] <= corner_distance
    subset_inicial = telemetry.loc[mask1 & mask2,]

    maskfull = subset_inicial["Throttle"] >= 99
    index_ini = subset_inicial.loc[maskfull,].index.to_list()[-1]
    index_ini = index_ini - 5

    # Parte final
    corner_distance_siguiente = corners.loc[
        corners["Number"] == corner_num + 1, "Distance"
    ].values[0]
    corner_distance_2siguiente = corners.loc[
        corners["Number"] == corner_num + 2, "Distance"
    ].values[0]

    mask1 = telemetry["Distance"] >= corner_distance_siguiente
    mask2 = telemetry["Distance"] <= corner_distance_2siguiente
    subset_final = telemetry.loc[mask1 & mask2,]

    maskfull = subset_final["Throttle"] >= 99
    index_fin = subset_final.loc[maskfull,].index.to_list()[0]
    index_fin = index_fin + 5

    # Fenerar subset
    subset = telemetry.loc[index_ini:index_fin,].copy()

    return subset


def plot_one_curva(
    subset_driver1,
    subset_driver2,
    driver1,
    driver2,
    color_driver1,
    color_driver2,
    corner_num,
    delta_time,
    ref_tel,
    save=False,
):
    fig, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})

    ax[0].plot(
        subset_driver1["Distance"],
        subset_driver1["Speed"],
        label=driver1,
        color=color_driver1,
    )
    ax[0].plot(
        subset_driver2["Distance"],
        subset_driver2["Speed"],
        label=driver2,
        color=color_driver2,
    )

    ax[0].set_ylabel("Speed (km/h)")
    ax[0].legend(loc="lower right")
    ax[0].set_title(f"Curva {corner_num}", fontsize=14, fontweight="bold", loc="left")

    ### Recta DRS delta
    index = subset_driver2.index.to_list()
    distance_ini = subset_driver2.loc[index[0], "Distance"]
    distance_fin = subset_driver2.loc[index[-1], "Distance"]

    ref_tel["Distance_ini"] = abs(ref_tel["Distance"] - distance_ini)
    ref_tel["Distance_fin"] = abs(ref_tel["Distance"] - distance_fin)
    index_ini = ref_tel["Distance_ini"].idxmin()
    index_fin = ref_tel["Distance_fin"].idxmin()

    ax[1].plot(
        ref_tel.loc[index_ini:index_fin, "Distance"],
        delta_time[index_ini : index_fin + 1],
        color=color_driver2,
        alpha=1.0,
        linewidth=1,
        linestyle="-",
    )
    ax[1].set_xlabel("Lap distance (m)")
    ax[1].set_ylabel("Gap (s)")
    ax[1].axhline(0, color=color_driver1, alpha=1.0, linestyle="-")

    # Flechas
    delta_ini = delta_time[index_ini]
    delta_fin = delta_time[index_fin]

    xp = ref_tel.loc[index_ini, "Distance"]
    yp = delta_ini

    ax[1].plot([xp], [yp], "o", color="orange", alpha=1.0, linewidth=0.5)
    ax[1].annotate(
        f"Delta Inicial = {round(delta_ini, 3)}s",
        (xp, yp),
        (xp, yp * 0.5),
        arrowprops=dict(facecolor="orange", headwidth=8, width=2),
        fontsize=8,
    )

    xp = ref_tel.loc[index_fin, "Distance"]
    yp = delta_fin

    ax[1].plot([xp], [yp], "o", color="lightblue", alpha=1.0, linewidth=0.5)
    ax[1].annotate(
        f"Delta final = {round(delta_fin, 3)}s",
        (xp, yp),
        (xp, yp * 0.5),
        arrowprops=dict(facecolor="lightblue", headwidth=8, width=2),
        fontsize=8,
        ha="right",
    )

    if save is True:
        plot_filename = f"curva {corner_num}.png"
        plt.savefig(plot_filename, format="png")

    else:
        plt.show()


def obten_subset_curvas(
    lap_driver1, lap_driver2, corners, driver1, driver2, session, save
):
    # 1) Obterner telemetria
    telemetry_driver1 = lap_driver1.get_telemetry().add_distance()
    telemetry_driver2 = lap_driver2.get_telemetry().add_distance()

    team_driver_1 = lap_driver1["Team"]
    team_driver_2 = lap_driver2["Team"]
    color_driver1 = ff1.plotting.get_team_color(team_driver_1, session)
    color_driver2 = ff1.plotting.get_team_color(team_driver_2, session)

    # 2) Obtener delta time
    delta_time, ref_tel, compare_tel = utils.delta_time(lap_driver1, lap_driver2)

    # 3) Curvas
    # Obtener subsets
    corner_num = 7
    subset_driver1 = obten_subset_curva(telemetry_driver1, corners, corner_num)
    subset_driver2 = obten_subset_curva(telemetry_driver2, corners, corner_num)

    # Plotear
    plot_one_curva(
        subset_driver1,
        subset_driver2,
        driver1,
        driver2,
        color_driver1,
        color_driver2,
        corner_num,
        delta_time,
        ref_tel,
        save=save,
    )

    # Obtener subsets
    corner_num = 9
    subset_driver1 = obten_subset_curva(telemetry_driver1, corners, corner_num)
    subset_driver2 = obten_subset_curva(telemetry_driver2, corners, corner_num)

    # Plotear
    plot_one_curva(
        subset_driver1,
        subset_driver2,
        driver1,
        driver2,
        color_driver1,
        color_driver2,
        corner_num,
        delta_time,
        ref_tel,
        save=save,
    )

    # Obtener subsets
    corner_num = 14
    subset_driver1 = obten_subset_chicane(telemetry_driver1, corners, corner_num)
    subset_driver2 = obten_subset_chicane(telemetry_driver2, corners, corner_num)

    # Plotear
    plot_one_curva(
        subset_driver1,
        subset_driver2,
        driver1,
        driver2,
        color_driver1,
        color_driver2,
        corner_num,
        delta_time,
        ref_tel,
        save=save,
    )


def plot_time_by_distance(
    lap_driver1, lap_driver2, distance_ini, distance_fin, driver1, driver2, save
):
    # 1) Obterner telemetria
    telemetry_driver1 = lap_driver1.get_telemetry().add_distance()
    telemetry_driver2 = lap_driver2.get_telemetry().add_distance()

    team_driver_1 = lap_driver1["Team"]
    team_driver_2 = lap_driver2["Team"]
    color_driver1 = ff1.plotting.get_team_color(team_driver_1, session)
    color_driver2 = ff1.plotting.get_team_color(team_driver_2, session)

    # 2) Obtener delta time
    delta_time, ref_tel, compare_tel = utils.delta_time(lap_driver1, lap_driver2)

    # 3) Crear los subsets
    # Create subset of distance driver 1
    mask_min = telemetry_driver1["Distance"] >= distance_ini
    mask_max = telemetry_driver1["Distance"] <= distance_fin

    subset_driver1 = telemetry_driver1.loc[mask_min & mask_max,].copy()

    # Create subset of distance driver 2
    mask_min = telemetry_driver2["Distance"] >= distance_ini
    mask_max = telemetry_driver2["Distance"] <= distance_fin

    subset_driver2 = telemetry_driver2.loc[mask_min & mask_max,].copy()

    # 4) Plot
    fig, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})

    ax[0].plot(
        subset_driver1["Distance"],
        subset_driver1["Speed"],
        label="Vuelta Anterior",
        color=color_driver1,
    )
    ax[0].plot(
        subset_driver2["Distance"],
        subset_driver2["Speed"],
        label="Vuelta Tráfico",
        color="red",
    )

    ax[0].set_ylabel("Speed (km/h)")
    ax[0].legend(loc="lower left")
    ax[0].set_title(
        f"Tráfico Alonso - Curva 11", fontsize=14, fontweight="bold", loc="left"
    )

    ### Recta DRS delta
    index = subset_driver2.index.to_list()
    distance_ini = subset_driver2.loc[index[0], "Distance"]
    distance_fin = subset_driver2.loc[index[-1], "Distance"]

    ref_tel["Distance_ini"] = abs(ref_tel["Distance"] - distance_ini)
    ref_tel["Distance_fin"] = abs(ref_tel["Distance"] - distance_fin)
    index_ini = ref_tel["Distance_ini"].idxmin()
    index_fin = ref_tel["Distance_fin"].idxmin()

    ax[1].plot(
        ref_tel.loc[index_ini:index_fin, "Distance"],
        delta_time[index_ini : index_fin + 1],
        color="red",
        alpha=1.0,
        linewidth=1,
        linestyle="-",
    )
    ax[1].set_xlabel("Lap distance (m)")
    ax[1].set_ylabel("Gap (s)")
    ax[1].axhline(0, color=color_driver1, alpha=1.0, linestyle="-")

    # Flechas
    delta_ini = delta_time[index_ini]
    delta_fin = delta_time[index_fin]

    xp = ref_tel.loc[index_ini, "Distance"]
    yp = delta_ini

    ax[1].plot([xp], [yp], "o", color="orange", alpha=1.0, linewidth=0.5)
    ax[1].annotate(
        f"Delta inicial = {round(delta_ini, 3)}s",
        (xp, yp),
        (xp, yp * 0.5),
        arrowprops=dict(facecolor="orange", headwidth=8, width=2),
        fontsize=8,
    )

    xp = ref_tel.loc[index_fin, "Distance"]
    yp = delta_fin

    ax[1].plot([xp], [yp], "o", color="lightblue", alpha=1.0, linewidth=0.5)
    ax[1].annotate(
        f"Delta final = {round(delta_fin, 3)}s",
        (xp, yp),
        (xp, yp * 0.5),
        arrowprops=dict(facecolor="lightblue", headwidth=8, width=2),
        fontsize=8,
        ha="right",
    )

    if save is True:
        plot_filename = f"Monaco_trafico_Alonso_Inicial.pdf"
        plt.savefig(plot_filename, format="pdf")

    else:
        plt.show()


def graph_given_corner(event, corners, drivers, list_corners, save=True):
    # 0) Setting del grafico
    # plot_title = f"Comparativa pilotos en las curvas {list_corners[0]} a {list_corners[-1]}"
    plot_title = f"Comparativa pilotos en Eau Rouge"
    plot_filename = plot_title.replace(" ", "_") + ".pdf"
    plot_filename = "./outputs/" + plot_filename

    fig, ax = plt.subplots(1, figsize=(12, 7))
    ax.set_title(plot_title)

    # 1) Parametros de referencia from list_corners
    if len(list_corners) == 1:
        # Hay una curva
        dist_referencia1 = corners.loc[
            corners["Number"] == list_corners[0], "Distance"
        ].iloc[0]
        dist_referencia2 = dist_referencia1
    elif len(list_corners) > 1:
        # Hay mas de una
        dist_referencia1 = corners.loc[
            corners["Number"] == list_corners[0], "Distance"
        ].iloc[0]
        dist_referencia2 = corners.loc[
            corners["Number"] == list_corners[-1], "Distance"
        ].iloc[0]
    else:
        # No hay curvas
        raise "No apartada ninguna curva de referencia"

    # 1) Loop a cada driver para obtener el subset de la curva de analisis, las velocidades clave y plotearla
    for driver in drivers:
        # 1.0) Obtener la telemetria del piloto
        laps_driver = event.laps.pick_drivers(driver)
        fastest_lap = laps_driver.pick_fastest()
        telemetry = fastest_lap.get_telemetry().add_distance()

        # 1.1) Obtener distancia inicial de la curva
        mask_lessdistance = telemetry["Distance"] < dist_referencia1
        mask_throttle = telemetry["Throttle"] >= 99

        dist_ini = telemetry.loc[
            mask_lessdistance & mask_throttle, "Distance"
        ].to_list()[-1]
        v_entry = telemetry.loc[mask_lessdistance & mask_throttle, "Speed"].to_list()[
            -1
        ]
        dist_ini = dist_ini - 50

        # 1.2) Obtener la distancia final de la curva
        mask_moredistance = telemetry["Distance"] > dist_referencia2

        dist_fin = telemetry.loc[
            mask_moredistance & mask_throttle, "Distance"
        ].to_list()[0]
        dist_fin = dist_fin + 50

        # 1.3) Obtener el subset
        mask_ini = telemetry["Distance"] >= dist_ini
        mask_fin = telemetry["Distance"] <= dist_fin

        subset = telemetry.loc[mask_ini & mask_fin,].copy()

        # 1.4) PLOT subset
        team_driver = laps_driver["Team"].iloc[0]
        color_driver = ff1.plotting.get_team_color(team_driver, event)

        ax.plot(
            subset["Distance"],
            subset["Speed"],
            label=driver,
            color=color_driver,
            alpha=1.0,
            linewidth=2,
            linestyle="-",
        )

    # 2) Añadir la localización de las curvas
    v_min = min(subset["Speed"])
    v_max = max(subset["Speed"])

    for corner in list_corners:
        dist = corners.loc[corners["Number"] == corner, "Distance"]
        ax.vlines(
            x=dist, ymin=v_min - 20, ymax=v_max + 20, linestyles="dotted", colors="grey"
        )

        ## Añadir el número de curva debajo de las lineas
        txt = f"{corner}"
        ax.text(dist, v_min - 15, txt, va="center_baseline", ha="center", size="medium")

    # 3) Ejes del gráfico
    ax.set_xlabel("Distancia (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_ylim([v_min - 20, v_max + 20])
    plt.legend()
    plt.grid(True)

    # 4) Save
    if save is True:
        eventname = event.event.EventName.split(" ")[0]
        path = f"{event.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, event.name)
        if not os.path.exists(path):
            os.makedirs(path)

        plot_filename = path + plot_filename
        plt.savefig(plot_filename, format="pdf")
    else:
        plt.show()


def graph_given_distances(event, name, drivers, list_distances, save=True):
    # 0) Setting del grafico
    plot_title = f"Comparativa pilotos en {name}"
    plot_filename = plot_title.replace(" ", "_") + ".pdf"

    fig = plt.figure(figsize=(12, 7), dpi=700)
    gs0 = gridspec.GridSpec(3, 3, figure=fig)

    # 1) Loop a cada driver para obtener el subset de la curva de analisis, las velocidades clave y plotearla
    final_times = []
    ultimas_distancias = []
    primeras_distancias = []

    ax1 = fig.add_subplot(gs0[:, :-1])
    ax1.set_title(plot_title, fontsize=25)
    for driver in drivers:
        # 1.0) Obtener la telemetria del piloto
        laps_driver = event.laps.pick_drivers(driver)
        """if driver == 'VER':
            fastest_lap = laps_driver[laps_driver['LapNumber']==17.0].iloc[0]
        else:
            fastest_lap = laps_driver.pick_fastest()"""
        fastest_lap = laps_driver.pick_fastest()
        telemetry = fastest_lap.get_telemetry().add_distance()

        # 1.1) Obtener distancia inicial de la curva
        mask_firstdistance = telemetry["Distance"] >= list_distances[0]
        mask_seconddistance = telemetry["Distance"] <= list_distances[1]

        subset = telemetry.loc[mask_firstdistance & mask_seconddistance,].copy()
        v0, t0 = estimate_unknown_speed_and_time(telemetry, x=list_distances[0])
        v1, t1 = estimate_unknown_speed_and_time(telemetry, x=list_distances[1])

        # 1.2) Redefinir subset con estoy nuevos valores
        speeds = [v0] + subset["Speed"].to_list() + [v1]
        times = [t0] + subset["Time"].to_list() + [t1]
        distances = (
            [list_distances[0]] + subset["Distance"].to_list() + [list_distances[1]]
        )

        d = {"Time": times, "Distance": distances, "Speed": speeds}
        df_subset = pd.DataFrame(d)

        # 1.2) Añadir columna de tiempo relativo
        initial_time = df_subset["Time"].iloc[0]
        df_subset["DeltaTime"] = df_subset["Time"].apply(lambda i: i - initial_time)
        final_time = format_time(df_subset["DeltaTime"].iloc[-1])
        final_times.append(final_time)

        primera_distancia = df_subset["Distance"].iloc[0]
        primeras_distancias.append(primera_distancia)

        ultima_distancia = df_subset["Distance"].iloc[-1]
        ultimas_distancias.append(ultima_distancia)

        # 1.3) PLOT subset
        team_driver = laps_driver["Team"].iloc[0]
        color_driver = ff1.plotting.get_team_color(team_driver, event)

        ax1.plot(
            df_subset["Distance"],
            df_subset["Speed"],
            label=driver,
            color=color_driver,
            alpha=1.0,
            linewidth=2,
            linestyle="-",
        )

    # 2) Calcular la tabla de tiempos en la curva
    d = {"Piloto": drivers, "Tiempo": final_times}
    df_times = pd.DataFrame(d)
    df_times = df_times.sort_values(by="Tiempo")
    df_times.reset_index(inplace=True)
    df_times["Delta"] = df_times["Tiempo"].apply(
        lambda i: round(float(i) - float(df_times["Tiempo"].iloc[0]), 3)
    )
    df_times["Delta"] = df_times["Delta"].apply(lambda i: "+{:.3f}s".format(i))
    df_times.loc[0, "Delta"] = "Leader"
    df_times["Tiempo"] = df_times["Tiempo"].apply(lambda i: str(i) + "s")
    df_times["Posicion"] = [str(i + 1) + "º" for i in range(df_times.shape[0])]
    df_times = df_times[["Posicion", "Piloto", "Tiempo", "Delta"]]
    print(df_times)

    # 3) Ejes del gráfico
    ax1.set_facecolor("#1E1C1C")
    plt.minorticks_on()
    ax1.xaxis.set_minor_locator(MultipleLocator(20))
    ax1.xaxis.grid(True, color="#4F4844", linestyle="-", linewidth=1.2)
    ax1.xaxis.grid(True, which="minor", color="#4F4844", linestyle="--", linewidth=0.45)

    # Determinar el valor del locator
    yticks = [label for label in ax1.get_yticklabels(which="major")]
    major_step = float(yticks[1].get_text()) - float(yticks[0].get_text())
    minor_step = major_step / 5
    ax1.yaxis.set_minor_locator(MultipleLocator(minor_step))
    ax1.yaxis.grid(True, color="#4F4844", linestyle="-", linewidth=1.2)
    ax1.yaxis.grid(True, which="minor", color="#4F4844", linestyle="--", linewidth=0.45)

    ax1.set_xlabel("Distancia (m)")
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_xlim([list_distances[0], list_distances[1]])
    ax1.legend(loc="upper right")

    # Añadir imagen del logo
    path = os.path.join(os.getcwd(), "src", "outputs", "logo_graficos.png")
    img = mpimg.imread(path)
    imagebox = OffsetImage(img, zoom=0.062, interpolation="bicubic", resample=True)
    ab = AnnotationBbox(
        imagebox,
        (0.005, 0.01),  # Coordenadas (x, y) para la esquina inferior izquierda
        xycoords="axes fraction",  # Coordenadas relativas al área de dibujo
        box_alignment=(
            0,
            0,
        ),  # Alinear la esquina inferior izquierda de la imagen con las coordenadas
        frameon=False,
    )  # Sin marco alrededor de la imagen
    ax1.add_artist(ab)

    # 3) Añadir la tabla de tiempos a la derecha del grafico
    table_data = df_times.values.tolist()
    column_labels = df_times.columns.tolist()
    row_colors = ["#2E2E2E", "#1E1C1C"]
    table = plt.table(
        cellText=table_data,
        colLabels=column_labels,
        cellLoc="center",
        loc="right",
        colColours=["#1E1C1C"] * len(column_labels),  # Fondo de la cabecera
        cellColours=[
            [row_colors[i % len(row_colors)] for _ in range(len(column_labels))]
            for i in range(len(table_data))
        ],  # Fondo alternado de las celdas
        # cellColours=[["#1E1C1C"] * len(column_labels)] * len(table_data),  # Fondo de las celdas
        bbox=[1.1, 0.55, 0.4, 0.45],
    )  # Posición y tamaño de la tabla [xmin, ymin, width, height]
    # Estilo de la tabla
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Cambiar el color del texto a blanco y eliminar bordes
    for key, cell in table.get_celld().items():
        cell.set_text_props(color="white")
        cell.set_edgecolor("none")  # Eliminar bordes

    plt.subplots_adjust(right=0.6)  # Evita que se solapen

    # 4) Generar Mapa
    gs1 = gs0[:, -1].subgridspec(2, 1, hspace=0.1)
    ax3 = fig.add_subplot(gs1[1])
    test_mapgraph(event, list_distances, ax3)

    # Margin
    plt.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.1, hspace=0.3)

    # 4) Save
    if save is True:
        eventname = event.event.EventName.split(" ")[0]
        path = f"{event.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, event.name)
        if not os.path.exists(path):
            os.makedirs(path)

        plot_filename = os.path.join(path, plot_filename)
        plt.savefig(plot_filename, format="pdf")
    else:
        plt.show()


def determine_sector(aux, sector_times):
    # Obtener ultimo tiempo
    last_time = aux["Time"].iloc[0]

    # Acumulated times
    acumulated_times = []
    acumulated_times.append(sector_times[0])
    acumulated_times.append(sector_times[0] + sector_times[1])
    acumulated_times.append(sector_times[0] + sector_times[1] + sector_times[2])

    # Check
    if last_time <= acumulated_times[0]:
        sector = 1
    else:
        if last_time <= acumulated_times[1]:
            sector = 2
        else:
            sector = 3

    return sector


def test_mapgraph(session, list_distances, ax):
    # 1) DATA
    circuit_info = session.get_circuit_info()
    corners = circuit_info.corners

    lap = session.laps.pick_fastest()
    aux = lap.get_telemetry().add_distance()

    sector_times = [lap.Sector1Time, lap.Sector2Time, lap.Sector3Time]

    # 1.1) Get pos data filtering aux biy list of distances
    mask_1 = aux["Distance"] >= list_distances[0]
    mask_2 = aux["Distance"] <= list_distances[1]
    pos = aux.loc[mask_1 & mask_2,]

    # 1.2) Get corner in this range of distances
    mask_1 = corners["Distance"] >= list_distances[0]
    mask_2 = corners["Distance"] <= list_distances[1]
    corners = corners.loc[mask_1 & mask_2,]

    # Get an array of shape [n, 2] where n is the number of points and the second
    # axis is x and y.
    track = pos.loc[:, ("X", "Y")].to_numpy()

    # Convert the rotation angle from degrees to radian.
    track_angle = circuit_info.rotation / 180 * np.pi

    # Rotate to the track_angle
    rotated_track = rotate(track, angle=track_angle)

    # rotated_track = rotate(rotated_track, angle=-np.pi/2)
    # Plot track map
    ax.plot(rotated_track[:, 0], rotated_track[:, 1], color="white", linewidth=15)

    sector = determine_sector(pos, sector_times)
    if sector == 1:
        ax.plot(rotated_track[:, 0], rotated_track[:, 1], color="#E83323", linewidth=9)
    elif sector == 2:
        ax.plot(rotated_track[:, 0], rotated_track[:, 1], color="#50AFDF", linewidth=9)
    elif sector == 3:
        ax.plot(rotated_track[:, 0], rotated_track[:, 1], color="#F6D548", linewidth=9)

    offset_vector = [300, 0]  # offset length is chosen arbitrarily to 'look good'
    text_inicial = []
    text_final = []

    i = 0
    for _, corner in corners.iterrows():
        # Create a string from corner number and letter
        txt = f"{corner['Number']}{corner['Letter']}"

        # Convert the angle from degrees to radian.
        offset_angle = corner["Angle"] / 180 * np.pi

        # Rotate the offset vector so that it points sideways from the track.
        offset_x, offset_y = rotate(offset_vector, angle=offset_angle)

        # Add the offset to the position of the corner
        text_x = corner["X"] + offset_x
        text_y = corner["Y"] + offset_y

        # Rotate the text position equivalently to the rest of the track map
        text_x, text_y = rotate([text_x, text_y], angle=track_angle)

        # Rotate the center of the corner equivalently to the rest of the track map
        track_x, track_y = rotate([corner["X"], corner["Y"]], angle=track_angle)

        # Draw a circle next to the track.
        ax.scatter(text_x, text_y, color="grey", s=400)

        # Draw a line from the track to this circle.
        # ax.plot([track_x, text_x], [track_y, text_y], color='grey')

        # Finally, print the corner number inside the circle.
        ax.text(
            text_x,
            text_y,
            txt,
            va="center_baseline",
            ha="center",
            size=10,
            color="white",
        )

        # Save the initial and final
        if i == 0:
            text_inicial.append([text_x, text_y])
        if i == (len(corners) - 1):
            text_final.append([text_x, text_y])

        i += 1

    ax.set_facecolor("#1E1C1C")
    ax.set_xticks([])
    ax.set_yticks([])

    # Ver el valor de las coordenadas x para invertir o no el ejex
    """if text_inicial[0][0] > text_final[0][0]:
        ax.invert_xaxis()
    if text_inicial[0][1] > text_final[0][1]:
        ax.invert_yaxis()"""
    # ax.axis('equal')


def get_lap(time, lap_times):
    for i in range(len(lap_times) - 1):
        if lap_times[i] <= time < lap_times[i + 1]:
            return float(i + 1)
    if time >= lap_times[-1]:
        return float(len(lap_times))
    return 0.0


def make_descomposition(series):
    type = "multiplicative"

    if type == "multiplicative":
        multiplicative_decomposition = seasonal_decompose(
            series, model="multiplicative", period=5
        )
        return multiplicative_decomposition
    elif type == "additive":
        additive_decomposition = seasonal_decompose(series, model="additive", period=5)
        return additive_decomposition


def graph_weather(df_weather, session, save=True):
    ################################# COMMON ########################################################################
    # Calcular el listado de objetos para delimitar las sesiones
    df_session_status = session.session_status
    finished_list = df_session_status.loc[
        df_session_status["Status"] == "Finished", "Time"
    ].to_list()

    started_list = []
    started_list.append(
        df_session_status.loc[
            df_session_status["Status"] == "Started", "Time"
        ].to_list()[0]
    )
    for i in range(2, len(df_session_status)):
        if (
            df_session_status.at[i, "Status"] == "Started"
            and df_session_status.at[i - 2, "Status"] != "Aborted"
        ):
            started_list.append(df_session_status.at[i, "Time"])

    # Columnas del gráfico
    columnas = [
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Pressure",
        "WindSpeed",
        "WindDirection",
        "Rainfall",
    ]

    # Checkear el Tipo de la sesion
    if session.session_info["Type"] == "Qualifying":
        # ----------------------------------------- PLOT ------------------------------------------------------------
        # Creamos subplots de diferentes tamaños
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        # plt.rcParams['axes.titlesize'] = 8
        plt.subplots_adjust(hspace=0.5, bottom=0.1)

        # Loop to plot
        meteo_resume = []
        for i in range(0, 2):
            for j in range(0, 4):
                if 4 * i + j != 7:
                    # Plot Time series and media
                    columna = columnas[4 * i + j]

                    label = DICT_METEO[columna]["nombre"]
                    ylabel = DICT_METEO[columna]["ylabel"]
                    color = DICT_METEO[columna]["color"]

                    if columna != "Rainfall":
                        ax[i, j].plot(
                            df_weather["Time"],
                            df_weather[columna],
                            label=label,
                            color=color,
                            alpha=0.5,
                            linewidth=2,
                        )
                        ax[i, j].plot(
                            df_weather["Time"],
                            df_weather[columna].rolling(window=5).mean(),
                            label="Media",
                            color=color,
                            linewidth=3,
                        )
                        # Valores extremos
                        val_min = min(df_weather[columna])
                        val_max = max(df_weather[columna])
                        delta_axis = (val_max - val_min) * 0.1

                    else:
                        ax[i, j].plot(
                            df_weather["Time"],
                            df_weather[columna],
                            label=label,
                            color=color,
                            linewidth=3,
                        )
                        # Valores extremos
                        val_min = 0
                        val_max = 1
                        delta_axis = 0.05

                    ## Añadir franja de color para delimitar la session
                    aux_columna = {}
                    aux_columna["name"] = columna
                    aux_columna["units"] = DICT_METEO[columna]["units"]

                    for iter, (time_start, time_finish) in enumerate(
                        zip(started_list, finished_list)
                    ):
                        # Plot the rectangle of SC
                        mask_start = df_weather["Time"] >= time_start
                        mask_finish = df_weather["Time"] <= time_finish
                        subset = df_weather.loc[mask_start & mask_finish,]

                        ax[i, j].fill_between(
                            subset["Time"],  # Usamos los tiempos del DataFrame filtrado
                            val_min - 3 * delta_axis,  # Limite inferior
                            val_max + 2 * delta_axis,  # Limite superior
                            color="gray",
                            alpha=0.25,
                            label="",
                        )

                        # Plot a vertical line in the extremes
                        ax[i, j].axvline(
                            x=time_start,
                            linestyle="dotted",
                            linewidth=2.0,
                            color="green",
                        )
                        ax[i, j].axvline(
                            x=time_finish,
                            linestyle="dotted",
                            linewidth=2.0,
                            color="red",
                        )

                        # Añadir texto de la sesion
                        x = (time_start + time_finish) / 2
                        if columna != "Rainfall":
                            y = val_max
                        else:
                            y = val_max - 0.04
                        text = f"Q{iter + 1}"
                        ax[i, j].text(
                            x,
                            y,
                            text,
                            fontdict={"fontsize": 16},
                            ha="center",
                            va="center",
                        )

                        # Save data
                        aux_columna[f"Q{iter + 1}min"] = subset[columna].min()
                        aux_columna[f"Q{iter + 1}max"] = subset[columna].max()
                        aux_columna[f"Q{iter + 1}med"] = subset[columna].mean()
                    meteo_resume.append(aux_columna)

                    ## Ponemos el titulo al gráfico
                    # ax[i, j].set_xlabel("Hora", fontsize=17)
                    ax[i, j].set_ylabel(ylabel, fontsize=17)
                    ax[i, j].set_xlim(
                        [
                            started_list[0] - timedelta(seconds=10),
                            finished_list[-1] + timedelta(seconds=10),
                        ]
                    )
                    ax[i, j].set_ylim([val_min - 3 * delta_axis, val_max + delta_axis])

                    if columna == "Rainfall":
                        y_pos = np.arange(0, 2, 1)
                        ax[i, j].set_yticks(y_pos, labels=["No", "Si"])

                    ax[i, j].tick_params(axis="both", which="major", labelsize=16)
                    ax[i, j].legend(loc="lower left", fontsize=16)
                    ax[i, j].set_title(f"Ev. {label}", fontsize=22, fontweight="bold")

        # ----------------------------------------- RESUMEN ---------------------------------------------------------
        print("\nValores meteorologicos de la session:")
        show_meteo_resume(meteo_resume)
        # print(meteo_resume)
    elif session.session_info["Type"] == "Race":
        # Traducir 'Time' to 'LapNumber'
        df_laps = session.laps
        total_laps = session.total_laps

        tiempos_inicio_list = []
        for i in range(1, total_laps + 1):
            # Determinar el Array de vueltas
            vuelta = float(i)
            tiempo_inicio_vuelta = min(
                df_laps.loc[df_laps["LapNumber"] == vuelta, "Time"]
            )
            tiempos_inicio_list.append(tiempo_inicio_vuelta)

        df_weather["Laps"] = df_weather["Time"].apply(
            lambda x: get_lap(x, tiempos_inicio_list)
        )

        # ----------------------------------------- PLOT ------------------------------------------------------------
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        plt.rcParams["axes.titlesize"] = 8

        # Loop to plot
        meteo_resume = []
        for i in range(0, 2):
            for j in range(0, 4):
                if 4 * i + j != 7:
                    ## Plot Time series and media
                    columna = columnas[4 * i + j]

                    label = DICT_METEO[columna]["nombre"]
                    ylabel = DICT_METEO[columna]["ylabel"]
                    color = DICT_METEO[columna]["color"]

                    if columna != "Rainfall":
                        ax[i, j].plot(
                            df_weather["Laps"],
                            df_weather[columna],
                            label=label,
                            color=color,
                            alpha=0.5,
                            linewidth=2,
                        )
                        ax[i, j].plot(
                            df_weather["Laps"],
                            df_weather[columna].rolling(window=5).mean(),
                            label="Media",
                            color=color,
                            linewidth=3,
                        )
                        # Valores extremos
                        val_min = min(df_weather[columna])
                        val_max = max(df_weather[columna])
                        delta_axis = (val_max - val_min) * 0.1
                    else:
                        ax[i, j].plot(
                            df_weather["Laps"],
                            df_weather[columna],
                            label=label,
                            color=color,
                            linewidth=3,
                        )
                        # Valores extremos
                        val_min = 0
                        val_max = 1
                        delta_axis = 0.05

                    ## Ponemos el titulo al gráfico
                    ax[i, j].set_xlabel("Vuelta", fontsize=17)
                    ax[i, j].set_ylabel(ylabel, fontsize=17)
                    ax[i, j].set_xlim([1, total_laps])
                    ax[i, j].set_ylim([val_min - 3 * delta_axis, val_max + delta_axis])
                    ax[i, j].legend(fontsize=16)
                    if columna == "Rainfall":
                        y_pos = np.arange(0, 2, 1)
                        ax[i, j].set_yticks(y_pos, labels=["No", "Si"])

                    ax[i, j].tick_params(axis="both", which="major", labelsize=16)
                    ax[i, j].set_title(f"Ev. {label}", fontsize=22, fontweight="bold")

    ## Quitar el último gráfico y añadir logo ahí
    x_logo = 1.0
    y_logo = 0.6
    zoom = 0.11
    add_team_logo(ax=ax[1, 3], x=x_logo, y=y_logo, zoom=zoom)
    ax[1, 3].axis("off")

    plt.tight_layout()

    # Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        eventname = session.event.EventName.split(" ")[0]
        path = f"{session.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, session.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar figura
        sesion_type = session.session_info["Type"]
        plot_filename = path + f"/Meteorologia_{eventname}_{sesion_type}.pdf"
        plt.savefig(plot_filename, format="pdf")
    else:
        plt.show()


def graph_weather_in_rows(df_weather, session, save=True):
    # Checkear el Tipo de la sesion
    if session.session_info["Type"] == "Qualy":
        # Calcular el listado de objetos para delimitar las sesiones
        df_session_status = session.session_status
        finished_list = df_session_status.loc[
            df_session_status["Status"] == "Finished", "Time"
        ].to_list()

        started_list = []
        started_list.append(
            df_session_status.loc[
                df_session_status["Status"] == "Started", "Time"
            ].to_list()[0]
        )
        for i in range(2, len(df_session_status)):
            if (
                df_session_status.at[i, "Status"] == "Started"
                and df_session_status.at[i - 2, "Status"] != "Aborted"
            ):
                started_list.append(df_session_status.at[i, "Time"])

        # Creamos subplots de diferentes tamaños
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))

        # Parametros del grafico
        plt.rcParams["axes.titlesize"] = 8

        # 1) Temperatura Ambiente
        ax[0, 0].plot(
            df_weather["Time"],
            df_weather["AirTemp"],
            label="Temp. Ambiente",
            color="orange",
        )

        # Añadir la linea discontinua de la sesion
        t_min = min(df_weather["AirTemp"])
        t_max = max(df_weather["AirTemp"])

        ## Finalizar sesion
        for i, time in enumerate(finished_list):
            ax[0, 0].vlines(
                x=time,
                ymin=t_min - 20,
                ymax=t_max + 20,
                linestyles="dotted",
                colors="red",
            )

        ## Iniciar sesion
        for i, time in enumerate(started_list):
            ax[0, 0].vlines(
                x=time,
                ymin=t_min - 20,
                ymax=t_max + 20,
                linestyles="dotted",
                colors="green",
            )

        ## Ponemos el titulo al gráfico
        ax[0, 0].set_xlabel("Hora")
        ax[0, 0].set_ylabel("Temperatura (ºC)")
        ax[0, 0].set_ylim([t_min - 0.1, t_max + 0.1])
        ax[0, 0].legend()
        ax[0, 0].title.set_text("Ev. de la Temp. Ambiente")

        # 2) Temperatura Pista
        ax[0, 1].plot(
            df_weather["Time"],
            df_weather["TrackTemp"],
            label="Temp. Pista",
            color="red",
            linewidth=2,
        )

        ## Ponemos el titulo al gráfico
        ax[0, 1].set_xlabel("Hora")
        ax[0, 1].set_ylabel("Temperatura (ºC)")
        ax[0, 1].legend()
        ax[0, 1].title.set_text("Ev. de la Temp. Pista")

        # 3) Humedad
        ax[0, 2].plot(
            df_weather["Time"],
            df_weather["Humidity"],
            label="Humedad",
            color="green",
            linewidth=2,
        )

        ## Ponemos el titulo al gráfico
        ax[0, 2].set_xlabel("Hora")
        ax[0, 2].set_ylabel("Humedad (%)")
        ax[0, 2].legend()
        ax[0, 2].title.set_text("Ev. de la Humedad")

        # 4) Presion atmosferica
        ax[0, 3].plot(
            df_weather["Time"],
            df_weather["Pressure"],
            label="Presión Atmosférica",
            color="saddlebrown",
            linewidth=2,
        )

        ## Ponemos el titulo al gráfico
        ax[0, 3].set_xlabel("Hora")
        ax[0, 3].set_ylabel("Presion (mbar)")
        ax[0, 3].legend()
        ax[0, 3].title.set_text("Ev. de la Presión Atm.")

        # 5) Velocidad del viento
        ax[1, 0].plot(
            df_weather["Time"],
            df_weather["WindSpeed"],
            label="Velocidad",
            color="royalblue",
            linewidth=2,
        )

        ## Ponemos el titulo al gráfico
        ax[1, 0].set_xlabel("Hora")
        ax[1, 0].set_ylabel("Velocidad del Viento (m/s)")
        ax[1, 0].legend()
        ax[1, 0].title.set_text("Ev. de la Velocidad del viento")

        # 6) Direccion del viento
        ax[1, 1].plot(
            df_weather["Time"],
            df_weather["WindDirection"],
            label="Dirección",
            color="lightblue",
            linewidth=2,
        )

        ## Ponemos el titulo al gráfico
        ax[1, 1].set_xlabel("Hora")
        ax[1, 1].set_ylabel("Velocidad del Dirección (º)")
        ax[1, 1].set_ylim([0, 360])
        ax[1, 1].legend()
        ax[1, 1].title.set_text("Ev. de la Dirección del viento")

        # 7) Lluvia
        ax[1, 2].plot(
            df_weather["Time"],
            df_weather["Rainfall"],
            label="Lluvia",
            color="yellow",
            linewidth=2,
        )

        ## Ponemos el titulo al gráfico
        ax[1, 2].set_xlabel("Hora")
        ax[1, 2].set_ylabel("Cae lluvia")
        y_pos = np.arange(0, 2, 1)
        ax[1, 2].set_yticks(y_pos, labels=["No", "Si"])
        ax[1, 2].legend()
        ax[1, 2].title.set_text("Ev. de la Lluvia")

    elif session.session_info["Type"] == "Race":
        # 0.1 Calcular el listado de objetos para delimitar las sesiones
        df_session_status = session.session_status
        finished_list = df_session_status.loc[
            df_session_status["Status"] == "Finished", "Time"
        ].to_list()

        started_list = []
        started_list.append(
            df_session_status.loc[
                df_session_status["Status"] == "Started", "Time"
            ].to_list()[0]
        )
        for i in range(2, len(df_session_status)):
            if (
                df_session_status.at[i, "Status"] == "Started"
                and df_session_status.at[i - 2, "Status"] != "Aborted"
            ):
                started_list.append(df_session_status.at[i, "Time"])

        # 0.2) Hacer un listado de la equivalencia de 'Time' con 'LapNumber' para usarlos como eje X
        df_laps = session.laps
        total_laps = session.total_laps

        tiempos_inicio_list = []
        for i in range(1, total_laps + 1):
            # Determinar el Array de vueltas
            vuelta = float(i)
            tiempo_inicio_vuelta = min(
                df_laps.loc[df_laps["LapNumber"] == vuelta, "Time"]
            )
            tiempos_inicio_list.append(tiempo_inicio_vuelta)

        # 0.3) Crear una columna en df_weather como una lista que indique la vuelta a la que se tomo el tiempo
        df_weather["Laps"] = df_weather["Time"].apply(
            lambda x: get_lap(x, tiempos_inicio_list)
        )

        # Creamos subplots de diferentes tamaños
        fig, ax = plt.subplots(7, figsize=(7, 20))

        # Parametros del grafico
        plt.rcParams["axes.titlesize"] = 8

        # 1) Temperatura Ambiente
        ax[0].plot(
            df_weather["Laps"],
            df_weather["AirTemp"],
            label="Temp. Ambiente",
            color="orange",
        )

        # Añadir la linea discontinua de la sesion
        t_min = min(df_weather["AirTemp"])
        t_max = max(df_weather["AirTemp"])

        ## Ponemos el titulo al gráfico
        ax[0].set_xlabel("Vuelta")
        ax[0].set_ylabel("Temperatura (ºC)")
        ax[0].set_xlim([1, total_laps])
        ax[0].legend()
        ax[0].title.set_text("Ev. de la Temp. Ambiente")

        # 2) Temperatura Pista
        ax[1].plot(
            df_weather["Laps"],
            df_weather["TrackTemp"],
            label="Temp. Pista",
            color="red",
            linewidth=2,
        )

        ## Ponemos el titulo al gráfico
        ax[1].set_xlabel("Vueltsa")
        ax[1].set_ylabel("Temperatura (ºC)")
        ax[1].set_xlim([1, total_laps])
        ax[1].legend()
        ax[1].title.set_text("Ev. de la Temp. Pista")

        # 3) Lluvia
        ax[2].plot(
            df_weather["Laps"],
            df_weather["Rainfall"],
            label="Lluvia",
            color="yellow",
            linewidth=2,
        )

        ## Ponemos el titulo al gráfico
        ax[2].set_xlabel("Vuelta")
        ax[2].set_ylabel("Cae lluvia")
        ax[2].set_xlim([1, total_laps])
        y_pos = np.arange(0, 2, 1)
        ax[2].set_yticks(y_pos, labels=["No", "Si"])
        ax[2].legend()
        ax[2].title.set_text("Ev. de la Lluvia")

        # 4) Humedad
        ax[3].plot(
            df_weather["Laps"],
            df_weather["Humidity"],
            label="Humedad",
            color="green",
            linewidth=2,
        )

        ## Ponemos el titulo al gráfico
        ax[3].set_xlabel("Vuelta")
        ax[3].set_ylabel("Humedad (%)")
        ax[3].set_xlim([1, total_laps])
        ax[3].legend()
        ax[3].title.set_text("Ev. de la Humedad")

        # 5) Presion atmosferica
        ax[4].plot(
            df_weather["Laps"],
            df_weather["Pressure"],
            label="Presión Atmosférica",
            color="saddlebrown",
            linewidth=2,
        )

        ## Ponemos el titulo al gráfico
        ax[4].set_xlabel("Vuelta")
        ax[4].set_ylabel("Presion (mbar)")
        ax[4].set_xlim([1, total_laps])
        ax[4].legend()
        ax[4].title.set_text("Ev. de la Presión Atm.")

        # 6) Velocidad del viento
        ax[5].plot(
            df_weather["Laps"],
            df_weather["WindSpeed"],
            label="Velocidad",
            color="royalblue",
            linewidth=2,
        )

        ## Ponemos el titulo al gráfico
        ax[5].set_xlabel("Vuelta")
        ax[5].set_ylabel("Velocidad del Viento (m/s)")
        ax[5].set_xlim([1, total_laps])
        ax[5].legend()
        ax[5].title.set_text("Ev. de la Velocidad del viento")

        # 7) Direccion del viento
        ax[6].plot(
            df_weather["Laps"],
            df_weather["WindDirection"],
            label="Dirección",
            color="lightblue",
            linewidth=2,
        )

        ## Ponemos el titulo al gráfico
        ax[6].set_xlabel("Vuelta")
        ax[6].set_ylabel("Velocidad del Dirección (º)")
        ax[6].set_xlim([1, total_laps])
        ax[6].set_ylim([0, 360])
        ax[6].legend()
        ax[6].title.set_text("Ev. de la Dirección del viento")

    plt.tight_layout()

    # Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        eventname = session.event.EventName.split(" ")[0]
        path = f"{session.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, session.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar figura
        sesion_type = session.session_info["Type"]
        plot_filename = path + f"/Meterologia_{eventname}_{sesion_type}.pdf"
        plt.savefig(plot_filename, format="pdf")
    else:
        plt.show()


def add_team_logo(ax, x, y, zoom):
    # Load logo
    path = os.path.join(os.getcwd(), "src", "outputs", "logo_graficos.png")
    img = mpimg.imread(path)

    # Crear objeto imagen
    imagebox = OffsetImage(img, zoom=zoom, interpolation="bicubic", resample=True)

    # Crear la anotacion
    ab = AnnotationBbox(
        imagebox,
        (x, y),  # Coordenadas (x, y) para la esquina inferior izquierda
        xycoords="axes fraction",  # Coordenadas relativas al área de dibujo
        box_alignment=(1, 0),
        # Alinear la esquina inferior izquierda de la imagen con las coordenadas
        frameon=False,
    )  # Sin marco alrededor de la imagen

    # Añadir al eje
    ax.add_artist(ab)


def add_teaam_logo_in_bars(df_final, dict_zooms, barplot, minimo, ax):
    # Leer logos y equipos
    logos = [
        mpimg.imread(os.path.join("src", "outputs", f"logo_{team}.png"))
        for team in df_final["Team"]
    ]
    teams = df_final["Team"].to_list()

    i = 0
    for bar, logo in zip(barplot, logos):
        # Obtener el centro de la barra
        x = bar.get_x() + bar.get_width() / 2
        y = (bar.get_height() + minimo) / 2

        # Calcular el factor de escala para ajustar la imagen a la altura de la barra
        team = teams[i]
        if team in dict_zooms:
            zoom = dict_zooms[team]
        else:
            zoom = 0.02

        # Crear la imagen offset
        im = OffsetImage(logo, zoom=zoom)  # Ajusta el zoom si es necesario
        ab = AnnotationBbox(im, (x, y), xycoords="data", frameon=False)

        # Añadir la imagen al gráfico
        ax.add_artist(ab)
        i += 1


def find_maximum_minimum_axis(maximo, minimo):
    delta = maximo - minimo

    if delta >= 0.5:
        # maximo_axis = round(maximo, 0) + 1.0
        maximo_axis = maximo + 1.5
        # maximo_axis = round(maximo, 0) + 1.0
        minimo_axis = minimo - 1.0
    else:
        # maximo_axis = round(maximo, 0) + 1.5
        maximo_axis = maximo + 1.5
        # minimo_axis = round(minimo, 0) - 1.0
        minimo_axis = minimo - 1.0

    return maximo_axis, minimo_axis


def find_maximum_minimum_axis_tiktok(maximo, minimo):
    delta = maximo - minimo

    if delta >= 0.5:
        maximo_axis = maximo + 0.25
        minimo_axis = minimo - 0.25
    else:
        maximo_axis = maximo + 0.15
        minimo_axis = minimo - 0.25

    return maximo_axis, minimo_axis


def format_timedelta(td):
    total_seconds = int(td.total_seconds())
    milliseconds = int(td.microseconds / 1000)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}.{milliseconds:03d}"


def make_countplot_one_sector(
    df_final, sector_time, ax, session, width, logo_vector, plot_title, tiktok_format
):
    # Order df by sector_time
    df_final.sort_values(by=sector_time, ascending=True, inplace=True)

    barplot = ax.bar(
        df_final["Xlabel"], df_final[sector_time], color=df_final["colors"], width=width
    )
    ax.bar_label(
        barplot, df_final[sector_time], padding=5, color="white", fontweight="bold"
    )

    ## Axis and titles
    translate_dict = {
        "Sector1Time": "Setor 1",
        "Sector2Time": "Setor 2",
        "Sector3Time": "Setor 3",
    }
    ax.set_ylabel(f"{translate_dict[sector_time]} (s)", fontsize=12)

    maximo = max(df_final[sector_time])
    minimo = min(df_final[sector_time])

    if tiktok_format:
        maximo_axis, minimo_axis = find_maximum_minimum_axis_tiktok(
            maximo=maximo, minimo=minimo
        )
    else:
        maximo_axis, minimo_axis = find_maximum_minimum_axis(
            maximo=maximo, minimo=minimo
        )

    ax.set_ylim([minimo_axis, maximo_axis])

    ax.set_xlim([-0.5, len(df_final["Team"]) - 0.5])

    eventname = session.event.EventName.split(" ")[0]
    sesion_type = session.session_info["Type"]

    if plot_title:
        ax.set_title(
            f"Mejores Sectores {eventname} {sesion_type} (Equipos)",
            fontsize=20,
            fontweight="bold",
        )

    ## Añadir imagen de logos de equipos
    if tiktok_format:
        dict_zooms = {
            "McLaren": 0.05,
            "Mclaren": 0.05,
            "Mercedes": 0.022,
            "RedBull": 0.03,
            "Alpine": 0.015,
            "Williams": 0.006,
            "Ferrari": 0.01,  # 0.03
            "RB": 0.015,
            "AstonMartin": 0.03,
            "Haas": 0.025,
            "KickSauber": 0.02,
        }
    else:
        dict_zooms = {
            "McLaren": 0.06,
            "Mclaren": 0.06,
            "Mercedes": 0.045,
            "RedBull": 0.06,
            "Alpine": 0.03,
            "Williams": 0.012,
            "Ferrari": 0.02,  # 0.03
            "RB": 0.03,
            "AstonMartin": 0.06,
            "Haas": 0.05,
            "KickSauber": 0.04,
        }

    add_teaam_logo_in_bars(
        df_final=df_final,
        dict_zooms=dict_zooms,
        barplot=barplot,
        minimo=minimo_axis,
        ax=ax,
    )

    ## Añadir imagen del logo
    add_team_logo(ax=ax, x=logo_vector[0], y=logo_vector[1], zoom=logo_vector[2])


def make_countplot(concepto, session, type, tiktok_format, save):
    # 1) EXTRACT DATA
    # Extract lap data
    df_laps = session.laps

    # Subset isPersonalBest
    mask_personalBest = df_laps["IsPersonalBest"] == True
    subset = df_laps.loc[mask_personalBest,]

    # 2) EVALUATE TYPE
    if type == "Driver":
        # 2) Extract the concept value from the best time of each driver
        list_drivers = np.unique(subset["Driver"].to_list())

        if concepto == "Sectors":
            list_best_S1 = []
            list_best_S2 = []
            list_best_S3 = []
            team_colors = []
            for driver in list_drivers:
                mask_driver = subset["Driver"] == driver
                subset_driver = subset.loc[mask_driver,]
                best_value = subset_driver.loc[
                    subset_driver["LapTime"] == min(subset_driver["LapTime"]),
                    ["Sector1Time", "Sector2Time", "Sector3Time"],
                ]

                list_best_S1.append(
                    float(format_time(best_value["Sector1Time"].iloc[0]))
                )
                list_best_S2.append(
                    float(format_time(best_value["Sector2Time"].iloc[0]))
                )
                list_best_S3.append(
                    float(format_time(best_value["Sector3Time"].iloc[0]))
                )
                team_colors.append(
                    ff1.plotting.get_team_color(subset_driver["Team"].iloc[0], session)
                )

            d = {
                "Driver": list_drivers,
                "Sector1Time": list_best_S1,
                "Sector2Time": list_best_S2,
                "Sector3Time": list_best_S3,
                "colors": team_colors,
            }
            df_final = pd.DataFrame(d)

            # 3) PLOT
            fig, ax = plt.subplots(3, figsize=(16, 10), dpi=700)
            width = 0.7

            # 3.1) S1
            df_final.sort_values(by="Sector1Time", ascending=True, inplace=True)

            barplot = ax[0].bar(
                df_final["Driver"],
                df_final["Sector1Time"],
                color=df_final["colors"],
                width=width,
            )
            ax[0].bar_label(
                barplot,
                df_final["Sector1Time"],
                padding=5,
                color="white",
                fontweight="bold",
            )

            ax[0].set_ylabel("Sector 1 (s)", fontsize=12)

            maximo = round(max(df_final["Sector1Time"]), 0) + 1.0
            minimo = round(min(df_final["Sector1Time"]), 0) - 1
            ax[0].set_ylim([minimo, maximo])
            ax[0].set_xlim([-0.5, len(df_final["Driver"]) - 0.5])

            eventname = session.event.EventName.split(" ")[0]
            sesion_type = session.session_info["Type"]
            ax[0].set_title(
                f"Mejores Sectores {eventname} {sesion_type}",
                fontsize=20,
                fontweight="bold",
            )

            ## Añadir imagen del logo
            path = os.path.join(os.getcwd(), "src", "outputs", "logo_graficos.png")
            img = mpimg.imread(path)
            imagebox = OffsetImage(
                img, zoom=0.09, interpolation="bicubic", resample=True
            )
            ab = AnnotationBbox(
                imagebox,
                (0.2, 0.60),  # Coordenadas (x, y) para la esquina inferior izquierda
                xycoords="axes fraction",  # Coordenadas relativas al área de dibujo
                box_alignment=(1, 0),
                # Alinear la esquina inferior izquierda de la imagen con las coordenadas
                frameon=False,
            )  # Sin marco alrededor de la imagen
            ax[0].add_artist(ab)

            # 3.2) S2
            df_final.sort_values(by="Sector2Time", ascending=True, inplace=True)

            barplot = ax[1].bar(
                df_final["Driver"],
                df_final["Sector2Time"],
                color=df_final["colors"],
                width=width,
            )
            ax[1].bar_label(
                barplot,
                df_final["Sector2Time"],
                padding=5,
                color="white",
                fontweight="bold",
            )

            ax[1].set_ylabel("Sector 2 (s)", fontsize=12)

            maximo = round(max(df_final["Sector2Time"]), 0) + 1.0
            minimo = round(min(df_final["Sector2Time"]), 0) - 1
            ax[1].set_ylim([minimo, maximo])
            ax[1].set_xlim([-0.5, len(df_final["Driver"]) - 0.5])

            ## Añadir imagen del logo
            path = os.path.join(os.getcwd(), "src", "outputs", "logo_graficos.png")
            img = mpimg.imread(path)
            imagebox = OffsetImage(
                img, zoom=0.09, interpolation="bicubic", resample=True
            )
            ab = AnnotationBbox(
                imagebox,
                (0.2, 0.60),  # Coordenadas (x, y) para la esquina inferior izquierda
                xycoords="axes fraction",  # Coordenadas relativas al área de dibujo
                box_alignment=(1, 0),
                # Alinear la esquina inferior izquierda de la imagen con las coordenadas
                frameon=False,
            )  # Sin marco alrededor de la imagen
            ax[1].add_artist(ab)

            # 3.3) S3
            df_final.sort_values(by="Sector3Time", ascending=True, inplace=True)

            barplot = ax[2].bar(
                df_final["Driver"],
                df_final["Sector3Time"],
                color=df_final["colors"],
                width=width,
            )
            ax[2].bar_label(
                barplot,
                df_final["Sector3Time"],
                padding=5,
                color="white",
                fontweight="bold",
            )

            ax[2].set_xlabel("Pilotos", fontsize=12)
            ax[2].set_ylabel("Sector 3 (s)", fontsize=12)

            maximo = round(max(df_final["Sector3Time"]), 0) + 1.0
            minimo = round(min(df_final["Sector3Time"]), 0) - 1
            ax[2].set_ylim([minimo, maximo])
            ax[2].set_xlim([-0.5, len(df_final["Driver"]) - 0.5])

            ## Añadir imagen del logo
            path = os.path.join(os.getcwd(), "src", "outputs", "logo_graficos.png")
            img = mpimg.imread(path)
            imagebox = OffsetImage(
                img, zoom=0.09, interpolation="bicubic", resample=True
            )
            ab = AnnotationBbox(
                imagebox,
                (0.2, 0.60),  # Coordenadas (x, y) para la esquina inferior izquierda
                xycoords="axes fraction",  # Coordenadas relativas al área de dibujo
                box_alignment=(1, 0),
                # Alinear la esquina inferior izquierda de la imagen con las coordenadas
                frameon=False,
            )  # Sin marco alrededor de la imagen
            ax[2].add_artist(ab)

        elif concepto == "LapTime":
            list_best_concept = []
            team_colors = []

            for driver in list_drivers:
                mask_driver = subset["Driver"] == driver
                subset_driver = subset.loc[mask_driver,]
                best_value = subset_driver.loc[
                    subset_driver["LapTime"] == min(subset_driver["LapTime"]), concepto
                ].iloc[0]

                list_best_concept.append(best_value)
                team_colors.append(
                    ff1.plotting.get_team_color(subset_driver["Team"].iloc[0], session)
                )

            d = {
                "Driver": list_drivers,
                concepto: list_best_concept,
                "colors": team_colors,
            }
            df_final = pd.DataFrame(d)
            df_final.sort_values(by=concepto, ascending=True, inplace=True)

            # 3) PLOT
            if tiktok_format:
                fig, ax = plt.subplots(1, figsize=(9, 16), dpi=700)
            else:
                fig, ax = plt.subplots(1, figsize=(12, 8), dpi=700)

            barplot = ax.barh(
                df_final["Driver"], df_final[concepto], color=df_final["colors"]
            )
            formatted_labels = df_final[concepto].apply(format_timedelta)
            ax.bar_label(
                barplot,
                labels=formatted_labels,
                padding=5,
                color="white",
                fontweight="bold",
            )

            ax.set_ylabel("Piloto")
            ax.set_xlabel(concepto)
            maximo = max(df_final[concepto]) + timedelta(seconds=2)
            minimo = min(df_final[concepto]) - timedelta(seconds=5)
            ax.set_xlim([minimo, maximo])
            ax.invert_yaxis()

            eventname = session.event.EventName.split(" ")[0]
            sesion_type = session.session_info["Type"]
            ax.set_title(
                f"{concepto} {eventname} {sesion_type} ({type})",
                fontsize=20,
                fontweight="bold",
            )

            ## Añadir imagen del logo
            path = os.path.join(os.getcwd(), "src", "outputs", "logo_graficos.png")
            img = mpimg.imread(path)
            if tiktok_format:
                ax.set_ylim(-0.5, len(df_final["Driver"].to_list()) - 0.5)
                ax.invert_yaxis()

                imagebox = OffsetImage(
                    img, zoom=0.062, interpolation="bicubic", resample=True
                )
                ab = AnnotationBbox(
                    imagebox,
                    (
                        1.02,
                        0.93,
                    ),  # Coordenadas (x, y) para la esquina inferior izquierda
                    xycoords="axes fraction",  # Coordenadas relativas al área de dibujo
                    box_alignment=(1, 0),
                    # Alinear la esquina inferior izquierda de la imagen con las coordenadas
                    frameon=False,
                )  # Sin marco alrededor de la imagen
            else:
                imagebox = OffsetImage(
                    img, zoom=0.062, interpolation="bicubic", resample=True
                )
                ab = AnnotationBbox(
                    imagebox,
                    (
                        1.0,
                        0.845,
                    ),  # Coordenadas (x, y) para la esquina inferior izquierda
                    xycoords="axes fraction",  # Coordenadas relativas al área de dibujo
                    box_alignment=(1, 0),
                    # Alinear la esquina inferior izquierda de la imagen con las coordenadas
                    frameon=False,
                )  # Sin marco alrededor de la imagen

            ax.add_artist(ab)

        else:
            list_best_concept = []
            team_colors = []
            for driver in list_drivers:
                mask_driver = subset["Driver"] == driver
                subset_driver = subset.loc[mask_driver,]
                best_value = subset_driver.loc[
                    subset_driver["LapTime"] == min(subset_driver["LapTime"]), concepto
                ].iloc[0]

                list_best_concept.append(int(best_value))
                team_colors.append(
                    ff1.plotting.get_team_color(subset_driver["Team"].iloc[0], session)
                )

            d = {
                "Driver": list_drivers,
                concepto: list_best_concept,
                "colors": team_colors,
            }
            df_final = pd.DataFrame(d)

            df_final.sort_values(by=concepto, ascending=False, inplace=True)

            # 3) PLOT
            fig, ax = plt.subplots(1, figsize=(12, 8), dpi=700)

            barplot = ax.bar(
                df_final["Driver"], df_final[concepto], color=df_final["colors"]
            )
            ax.bar_label(
                barplot, df_final[concepto], padding=5, color="white", fontweight="bold"
            )

            ax.set_xlabel("Piloto", fontsize=12)
            ax.set_ylabel(f"{concepto} (km/h)", fontsize=12)
            maximo = max(df_final[concepto]) + 10
            minimo = min(df_final[concepto]) - 50
            ax.set_ylim([minimo, maximo])
            ax.set_xlim([-0.5, len(df_final["Driver"]) - 0.5])

            eventname = session.event.EventName.split(" ")[0]
            sesion_type = session.session_info["Type"]
            ax.set_title(
                f"{concepto} {eventname} {sesion_type}", fontsize=20, fontweight="bold"
            )

            ## Añadir imagen del logo
            path = os.path.join(os.getcwd(), "src", "outputs", "logo_graficos.png")
            img = mpimg.imread(path)
            imagebox = OffsetImage(
                img, zoom=0.062, interpolation="bicubic", resample=True
            )
            ab = AnnotationBbox(
                imagebox,
                (1.0, 0.845),  # Coordenadas (x, y) para la esquina inferior izquierda
                xycoords="axes fraction",  # Coordenadas relativas al área de dibujo
                box_alignment=(1, 0),
                # Alinear la esquina inferior izquierda de la imagen con las coordenadas
                frameon=False,
            )  # Sin marco alrededor de la imagen
            ax.add_artist(ab)

    elif type == "Team":
        # Variables
        if tiktok_format:
            zoom = 0.061
            x_logo = 0.36
            y_logo = 0.75
        else:
            zoom = 0.06
            x_logo = 0.25
            y_logo = 0.53

        # 2) Extract the concept value from the best time of each driver
        list_teams = np.unique(subset["Team"].to_list())

        if concepto == "Sectors":
            list_best_S1 = []
            list_best_S2 = []
            list_best_S3 = []
            team_colors = []
            list_xlabel = []
            for team in list_teams:
                mask_team = subset["Team"] == team
                subset_team = subset.loc[mask_team,]
                best_value = subset_team.loc[
                    subset_team["LapTime"] == min(subset_team["LapTime"]),
                    ["Sector1Time", "Sector2Time", "Sector3Time", "Driver"],
                ]
                driver = best_value["Driver"].iloc[0]

                list_best_S1.append(
                    float(format_time(best_value["Sector1Time"].iloc[0]))
                )
                list_best_S2.append(
                    float(format_time(best_value["Sector2Time"].iloc[0]))
                )
                list_best_S3.append(
                    float(format_time(best_value["Sector3Time"].iloc[0]))
                )
                list_xlabel.append(f"{team}\n({driver})")
                team_colors.append(ff1.plotting.get_team_color(team, session))

            list_teams = [DICT_TEAMS[team] for team in list_teams]

            d = {
                "Team": list_teams,
                "Xlabel": list_xlabel,
                "Sector1Time": list_best_S1,
                "Sector2Time": list_best_S2,
                "Sector3Time": list_best_S3,
                "colors": team_colors,
            }
            df_final = pd.DataFrame(d)

            # 3) PLOT
            # ----------------------------------------------------------------------------------------------------------
            if tiktok_format:
                fig, ax = plt.subplots(3, figsize=(9, 16), dpi=700)
            else:
                fig, ax = plt.subplots(3, figsize=(12, 8), dpi=700)
                # Ajustar los márgenes
                fig.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.94)

            width = 0.7
            logo_vector = [x_logo, y_logo, zoom]

            # 3.1) S1
            # ----------------------------------------------------------------------------------------------------------
            plot_title = True
            make_countplot_one_sector(
                df_final,
                "Sector1Time",
                ax[0],
                session,
                width,
                logo_vector,
                plot_title,
                tiktok_format,
            )

            # 3.2) S2
            # ----------------------------------------------------------------------------------------------------------
            plot_title = False
            make_countplot_one_sector(
                df_final,
                "Sector2Time",
                ax[1],
                session,
                width,
                logo_vector,
                plot_title,
                tiktok_format,
            )

            # 3.3) S3
            # ----------------------------------------------------------------------------------------------------------
            plot_title = False
            make_countplot_one_sector(
                df_final,
                "Sector3Time",
                ax[2],
                session,
                width,
                logo_vector,
                plot_title,
                tiktok_format,
            )

            # Ajustar subplot
            plt.subplots_adjust(hspace=0.3)

        elif concepto == "LapTime":
            list_best_concept = []
            team_colors = []

            for team in list_teams:
                mask_team = subset["Team"] == team
                subset_team = subset.loc[mask_team,]
                best_value = subset_team.loc[
                    subset_team["LapTime"] == min(subset_team["LapTime"]), concepto
                ].iloc[0]

                list_best_concept.append(best_value)
                team_colors.append(ff1.plotting.get_team_color(team, session))

            # Reduce Team values
            list_teams = [DICT_TEAMS[team] for team in list_teams]
            d = {"Team": list_teams, concepto: list_best_concept, "colors": team_colors}
            df_final = pd.DataFrame(d)

            df_final.sort_values(by=concepto, ascending=True, inplace=True)

            # 3) PLOT
            fig, ax = plt.subplots(1, figsize=(12, 8), dpi=700)

            barplot = ax.barh(
                df_final["Team"], df_final[concepto], color=df_final["colors"]
            )

            # ax.bar_label(barplot, df_final[concepto], padding=5, color='white', fontweight='bold')
            formatted_labels = df_final[concepto].apply(format_timedelta)
            ax.bar_label(
                barplot,
                labels=formatted_labels,
                padding=5,
                color="white",
                fontweight="bold",
            )

            ax.set_ylabel("Team")
            ax.set_xlabel(concepto)
            maximo = max(df_final[concepto]) + timedelta(seconds=2)
            minimo = min(df_final[concepto]) - timedelta(seconds=5)
            ax.set_xlim([minimo, maximo])
            ax.invert_yaxis()

            eventname = session.event.EventName.split(" ")[0]
            sesion_type = session.session_info["Type"]
            ax.set_title(
                f"{concepto} {eventname} {sesion_type} ({type})",
                fontsize=20,
                fontweight="bold",
            )

            ## Añadir imagen del logo
            path = os.path.join(os.getcwd(), "src", "outputs", "logo_graficos.png")
            img = mpimg.imread(path)
            if tiktok_format:
                imagebox = OffsetImage(
                    img, zoom=0.062, interpolation="bicubic", resample=True
                )
                ab = AnnotationBbox(
                    imagebox,
                    (
                        1.0,
                        0.845,
                    ),  # Coordenadas (x, y) para la esquina inferior izquierda
                    xycoords="axes fraction",  # Coordenadas relativas al área de dibujo
                    box_alignment=(1, 0),
                    # Alinear la esquina inferior izquierda de la imagen con las coordenadas
                    frameon=False,
                )  # Sin marco alrededor de la imagen
            else:
                imagebox = OffsetImage(
                    img, zoom=0.062, interpolation="bicubic", resample=True
                )
                ab = AnnotationBbox(
                    imagebox,
                    (
                        1.0,
                        0.845,
                    ),  # Coordenadas (x, y) para la esquina inferior izquierda
                    xycoords="axes fraction",  # Coordenadas relativas al área de dibujo
                    box_alignment=(1, 0),
                    # Alinear la esquina inferior izquierda de la imagen con las coordenadas
                    frameon=False,
                )  # Sin marco alrededor de la imagen
            ax.add_artist(ab)

        else:
            list_best_concept = []
            list_xlabel = []
            team_colors = []

            for team in list_teams:
                mask_team = subset["Team"] == team
                subset_team = subset.loc[mask_team,]
                df_best_value = subset_team.loc[
                    subset_team["LapTime"] == min(subset_team["LapTime"]),
                ]
                best_value = df_best_value[concepto].iloc[0]
                driver = df_best_value["Driver"].iloc[0]

                list_best_concept.append(int(best_value))
                list_xlabel.append(f"{team}\n({driver})")
                team_colors.append(ff1.plotting.get_team_color(team, session))

            # Reduce Team values
            list_teams = [DICT_TEAMS[team] for team in list_teams]
            d = {"Team": list_teams, concepto: list_best_concept, "colors": team_colors}
            df_final = pd.DataFrame(d)

            df_final.sort_values(by=concepto, ascending=False, inplace=True)

            # 3) PLOT
            fig, ax = plt.subplots(1, figsize=(12, 8), dpi=700)

            barplot = ax.bar(
                df_final["Team"], df_final[concepto], color=df_final["colors"]
            )
            ax.bar_label(
                barplot, df_final[concepto], padding=5, color="white", fontweight="bold"
            )

            ax.set_xlabel("Team")
            ax.set_ylabel(concepto)
            maximo = max(df_final[concepto]) + 10
            minimo = min(df_final[concepto]) - 50
            ax.set_ylim([minimo, maximo])
            ax.set_xlim([-0.5, len(df_final["Team"]) - 0.5])

            eventname = session.event.EventName.split(" ")[0]
            sesion_type = session.session_info["Type"]
            ax.set_title(
                f"{concepto} {eventname} {sesion_type} ({type})",
                fontsize=20,
                fontweight="bold",
            )

            ## Añadir imagen del logo
            path = os.path.join(os.getcwd(), "src", "outputs", "logo_graficos.png")
            img = mpimg.imread(path)
            imagebox = OffsetImage(
                img, zoom=0.062, interpolation="bicubic", resample=True
            )
            ab = AnnotationBbox(
                imagebox,
                (1.0, 0.845),  # Coordenadas (x, y) para la esquina inferior izquierda
                xycoords="axes fraction",  # Coordenadas relativas al área de dibujo
                box_alignment=(1, 0),
                # Alinear la esquina inferior izquierda de la imagen con las coordenadas
                frameon=False,
            )  # Sin marco alrededor de la imagen
            ax.add_artist(ab)
    # Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        eventname = session.event.EventName.split(" ")[0]
        path = f"{session.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, session.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar figura
        if tiktok_format:
            sesion_type = session.session_info["Type"]
            plot_filename = (
                path + f"/Tiktok{concepto}_{eventname}_{sesion_type}_{type}.png"
            )
            plt.savefig(plot_filename, format="png")
        else:
            sesion_type = session.session_info["Type"]
            plot_filename = path + f"/{concepto}_{eventname}_{sesion_type}_{type}.pdf"
            plt.savefig(plot_filename, format="pdf")
    else:
        plt.show()


def make_countplot_for_corners(corner_distance: list, session, type, save):
    # 1) EXTRACT DATA
    # Extract lap data
    df_laps = session.laps

    # Subset isPersonalBest
    mask_personalBest = df_laps["IsPersonalBest"] == True
    subset = df_laps.loc[mask_personalBest,]

    # Check type
    if type == "Team":
        # Extract data
        # Variables
        zoom = 0.08
        x_logo = 0.22
        y_logo = 0.55

        # 2) Extract the list of teams unique
        list_teams = np.unique(subset["Team"].to_list())

        min_speed_teams = []
        team_colors = []
        for team in list_teams:
            # Extract the telemetry of the fastest lap of the team
            mask_team = subset["Team"] == team
            fastest_lap = min(subset.loc[mask_team, "LapTime"])
            mask_fastest_lap = subset["LapTime"] == fastest_lap
            telemetry = (
                subset.loc[mask_team & mask_fastest_lap,].get_telemetry().add_distance()
            )

            # Find min speed in the distances
            mask_mindist = telemetry["Distance"] >= corner_distance[0]
            mask_maxdist = telemetry["Distance"] <= corner_distance[1]

            min_speed_teams.append(
                min(telemetry.loc[mask_mindist & mask_maxdist, "Speed"])
            )
            team_colors.append(ff1.plotting.get_team_color(team, session))
        d = {"Team": list_teams, "min_speeds": min_speed_teams, "colors": team_colors}
        df_final = pd.DataFrame(d)
        df_final.sort_values(by="min_speeds", ascending=False, inplace=True)

    elif type == "Driver":
        pass

    # Plot
    # 3) PLOT
    fig, ax = plt.subplots(1, figsize=(12, 8), dpi=700)

    barplot = ax.bar(df_final["Team"], df_final["min_speeds"], color=df_final["colors"])
    ax.bar_label(
        barplot, df_final["min_speeds"], padding=5, color="white", fontweight="bold"
    )

    ax.set_xlabel("Team")
    ax.set_ylabel("min_speeds")
    maximo = max(df_final["min_speeds"]) + 10
    minimo = min(df_final["min_speeds"]) - 50
    ax.set_ylim([minimo, maximo])
    ax.set_xlim([-0.5, len(df_final["Team"]) - 0.5])

    eventname = session.event.EventName.split(" ")[0]
    sesion_type = session.session_info["Type"]
    ax.set_title(
        f"Velocidad Minima {eventname} {sesion_type} ({type})",
        fontsize=20,
        fontweight="bold",
    )

    ## Añadir imagen del logo
    path = os.path.join(os.getcwd(), "src", "outputs", "logo_graficos.png")
    img = mpimg.imread(path)
    imagebox = OffsetImage(img, zoom=0.062, interpolation="bicubic", resample=True)
    ab = AnnotationBbox(
        imagebox,
        (1.0, 0.845),  # Coordenadas (x, y) para la esquina inferior izquierda
        xycoords="axes fraction",  # Coordenadas relativas al área de dibujo
        box_alignment=(1, 0),
        # Alinear la esquina inferior izquierda de la imagen con las coordenadas
        frameon=False,
    )  # Sin marco alrededor de la imagen
    ax.add_artist(ab)

    # Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        eventname = session.event.EventName.split(" ")[0]
        path = f"{session.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, session.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar figura
        sesion_type = session.session_info["Type"]
        plot_filename = path + f"/MinSpeeds_{eventname}_{sesion_type}_{type}.pdf"
        plt.savefig(plot_filename, format="pdf")
    else:
        plt.show()


def generate_laptime_column(df_results):
    laptimes = []
    for i in range(df_results.shape[0]):
        check_Q1 = pd.isna(df_results["Q1"].iloc[i])
        check_Q2 = pd.isna(df_results["Q2"].iloc[i])
        check_Q3 = pd.isna(df_results["Q3"].iloc[i])

        if not check_Q1:
            if not check_Q2:
                if not check_Q3:
                    laptimes.append(df_results["Q3"].iloc[i])
                else:
                    laptimes.append(df_results["Q2"].iloc[i])
            else:
                laptimes.append(df_results["Q1"].iloc[i])
        else:
            laptimes.append("NoTime")

    df_results["LapTime"] = laptimes
    return df_results


def join_with_speed_trap(df_results):
    DICT_TEAM_COLOR = {
        "Ferrari": "#dc0000",
        "Caterham": "#00ff00",
        "Force India": "#ff87bc",
        "HRT": "white",
        "Lotus F1": "#000000",
        "Marussia": "#9b0000",
        "McLaren": "#ff8700",
        "Mercedes": "#00d2be",
        "Red Bull": "#0600ef",
        "Renault": "#fff500",
        "Sauber": "#006eff",
        "Toro Rosso": "#469bff",
        "Williams": "#00d2be",
    }
    df_speed = pd.DataFrame(
        {
            "Abbreviation": [
                "VER",
                "PER",
                "KOB",
                "SEN",
                "MAL",
                "MAS",
                "DIR",
                "HUL",
                "MSC",
                "ROS",
                "ALO",
                "RIC",
                "GRO",
                "RAI",
                "KAR",
                "HAM",
                "BUT",
                "DLR",
                "GLO",
                "PIC",
                "KOV",
                "PET",
                "VET",
                "WEB",
            ],
            "Speed": [
                320.4,
                320.2,
                320.0,
                319.5,
                319.3,
                319.1,
                318.4,
                318.3,
                318.2,
                317.7,
                316.8,
                316.6,
                315.6,
                315.4,
                314.5,
                314.4,
                314.3,
                314.1,
                313.2,
                313.1,
                312.5,
                312.3,
                308.4,
                308.0,
            ],
        }
    )
    df_speed = df_speed.reset_index(drop=True)
    df_results = df_results.copy()

    df_new = pd.merge(
        df_results,
        df_speed,
        how="inner",
        left_on="Abbreviation",
        right_on="Abbreviation",
    )
    df_new["TeamColor"] = df_new["TeamName"].apply(lambda i: DICT_TEAM_COLOR[i])
    return df_new


def make_df_by_results(session):
    # Obtain teams list
    df_results = session.results
    list_teams = np.unique(df_results["TeamName"].to_list()).tolist()
    df_results = generate_laptime_column(df_results)
    df_results = join_with_speed_trap(df_results)

    # Loop into teams
    dict_results = {}
    for team in list_teams:
        ## Extract the best lap for the team
        mask_team = df_results["TeamName"] == team
        subset_team = df_results.loc[mask_team,]
        best_lap_team = subset_team.loc[
            subset_team["LapTime"] == min(subset_team["LapTime"]),
        ]

        ## Get telemetry of this lap
        if best_lap_team.shape[0] > 1:
            best_lap_team = best_lap_team.iloc[0]

        ## Calculate vmed and vmax of this lap
        circuit_length = DICT_LENGTH_CIRCUITS.get(
            session.event.EventName.split(" Grand ")[0]
        )

        vmed = (
            circuit_length / best_lap_team["LapTime"].dt.total_seconds().iloc[-1]
        ) * 3.6
        vmax = best_lap_team["Speed"].iloc[0]

        ## Extract team color
        team_color = best_lap_team["TeamColor"].iloc[0]

        ## Save
        dict_results[team] = {"vmed": vmed, "vmax": vmax, "teamcolor": team_color}

    return dict_results


def make_df_by_laps(session):
    # Obtain teams list
    list_teams = np.unique(session.results["TeamName"].to_list())
    # circuit = session.event.EventName.split(" ")[0]

    # Loop into teams
    dict_results = {}
    for team in list_teams:
        ## Extract the best lap for the team
        mask_team = session.laps["Team"] == team

        mask_personalBest = session.laps["IsPersonalBest"] == True
        subset_team = session.laps.loc[mask_team & mask_personalBest,]
        best_lap_team = subset_team.loc[
            subset_team["LapTime"] == min(subset_team["LapTime"]),
        ]
        # from tabulate import tabulate
        # print(tabulate(best_lap_team[['Driver', 'LapTime', 'Team']], headers='keys', showindex=False, tablefmt='grid'))
        # print("\n")

        ## Get telemetry of this lap
        if best_lap_team.shape[0] > 1:
            best_lap_team = best_lap_team[
                best_lap_team["Time"] == best_lap_team["Time"].max()
            ]

        telemetry_team = best_lap_team.get_telemetry().add_distance()

        ## Calculate vmed and vmax of this lap
        circuit_length = DICT_LENGTH_CIRCUITS.get(
            session.event.EventName.split(" Grand ")[0]
        )
        # print(f"circuit: {circuit}, circuit_length: {circuit_length}")
        vmed = (
            circuit_length / telemetry_team["Time"].dt.total_seconds().iloc[-1]
        ) * 3.6
        vmax = telemetry_team["Speed"].max()

        ## Extract team color
        team_color = ff1.plotting.get_team_color(team, session)

        ## Save
        dict_results[team] = {"vmed": vmed, "vmax": vmax, "teamcolor": team_color}
    return dict_results


def make_graph_aero_analisis(
    session, data_origin="results", save=True, save_stats_circuito=False
):
    if data_origin == "results":
        dict_results = make_df_by_results(session)
    elif data_origin == "laps":
        dict_results = make_df_by_laps(session)
    else:
        raise NotImplementedError(f"Data origin : {data_origin} Not implemented")

    print(dict_results)

    # ---------------------- Rankings --------------------------
    ## Usefull Parameters
    min_vmed = min(team["vmed"] for team in dict_results.values())
    min_vmax = min(team["vmax"] for team in dict_results.values())
    max_vmed = max(team["vmed"] for team in dict_results.values())
    max_vmax = max(team["vmax"] for team in dict_results.values())
    center_x = (min_vmed + max_vmed) / 2
    center_y = (min_vmax + max_vmax) / 2
    # print(min_vmed, min_vmax, max_vmed, max_vmax, center_x, center_y)

    ## Calcular downforce y eficiencia
    vmeds = []
    vmaxs = []
    for key, value in dict_results.items():
        downforce = ((value["vmed"] - value["vmax"]) + (center_y - center_x)) / np.sqrt(
            2
        )
        eficiencia = (
            (value["vmed"] + value["vmax"]) - (center_y + center_x)
        ) / np.sqrt(2)
        drag = value["vmax"] - center_y
        value["Downforce"] = downforce
        value["Eficiencia"] = eficiencia
        value["Bajo Drag"] = drag
        dict_results[key] = value

        vmeds.append(value["vmed"])
        vmaxs.append(value["vmax"])

    ## Mostrar el Ranking de valores
    eventname = session.event.EventName.split(" ")[0]

    show_rankings = False
    if show_rankings:
        print(f"\nRankings {eventname}")
        print("-" * 50)
        show_ranking_column(dict_results, "Downforce", "Equipos", session)
        show_ranking_column(dict_results, "Eficiencia", "Equipos", session)
        show_ranking_column(dict_results, "Bajo Drag", "Equipos", session)
        print(json.dumps(dict_results, indent=4))

    # ---------------------- PLOT --------------------------
    plt.close("all")
    fig, ax = plt.subplots(1, figsize=(12, 12), dpi=700)

    # Plot
    for key, value in dict_results.items():
        #  Set alpha
        if key == "Red Bull Racing":
            alpha = 0.8
        else:
            alpha = 1.0

        # Plot
        ax.plot(
            value["vmed"],
            value["vmax"],
            "o",
            color=value["teamcolor"],
            alpha=alpha,
            markersize=16,
        )

        # Etiqueta del equipo
        ax.text(
            value["vmed"] + 0.05,
            value["vmax"] + 0.05,
            DICT_TEAMS[key],
            color=value["teamcolor"],
            alpha=alpha,
            fontsize=20,
        )

    ## ARROWS
    arrow_length_x = (max_vmed - min_vmed) / 2
    arrow_length_y = (max_vmax - min_vmax) / 2

    # ax.arrow(x=max_vmed, y=center_y, dx=(max_vmed-min_vmed), dy=0, width=0.06, color="white")
    ax.arrow(
        center_x,
        center_y,
        arrow_length_x,
        0,
        head_width=0.2,
        head_length=0.2,
        fc="white",
        ec="white",
        linewidth=0.75,
    )
    ax.text(
        center_x + arrow_length_x, center_y + 0.25, "Rápido", color="white", fontsize=18
    )

    ax.arrow(
        center_x,
        center_y,
        -arrow_length_x,
        0,
        head_width=0.2,
        head_length=0.2,
        fc="white",
        ec="white",
        linewidth=0.75,
    )
    ax.text(
        center_x - arrow_length_x - 0.25,
        center_y + 0.25,
        "Lento",
        color="white",
        fontsize=18,
    )

    ax.arrow(
        center_x,
        center_y,
        0,
        arrow_length_y,
        head_width=0.1,
        head_length=0.2,
        fc="white",
        ec="white",
        linewidth=0.75,
    )
    ax.text(
        center_x - 0.2,
        center_y + arrow_length_y + 0.3,
        "Bajo Drag",
        color="white",
        fontsize=18,
    )
    ax.arrow(
        center_x,
        center_y,
        0,
        -arrow_length_y,
        head_width=0.1,
        head_length=0.2,
        fc="white",
        ec="white",
        linewidth=0.75,
    )
    ax.text(
        center_x - 0.2,
        center_y - arrow_length_y - 0.4,
        "Alto Drag",
        color="white",
        fontsize=18,
    )

    ax.arrow(
        center_x,
        center_y,
        max_vmed - center_x,
        max_vmax - center_y,
        head_width=0.2,
        head_length=0.2,
        fc="white",
        ec="white",
        linewidth=0.75,
        linestyle="dashed",
    )  # ↗️
    ax.text(
        max_vmed - 0.1, max_vmax + 0.3, "Alta Eficiencia", color="white", fontsize=18
    )

    ax.arrow(
        center_x,
        center_y,
        min_vmed - center_x,
        min_vmax - center_y,
        head_width=0.2,
        head_length=0.2,
        fc="white",
        ec="white",
        linewidth=0.75,
        linestyle="dashed",
    )  # ↙️
    ax.text(
        min_vmed - 0.2, min_vmax - 0.4, "Baja Eficiencia", color="white", fontsize=18
    )

    ax.arrow(
        center_x,
        center_y,
        max_vmed - center_x,
        min_vmax - center_y,
        head_width=0.1,
        head_length=0.2,
        fc="white",
        ec="white",
        linewidth=0.75,
        linestyle="dashed",
    )  # ↘️
    ax.text(
        max_vmed - 0.1, min_vmax - 0.44, "Alta Downforce", color="white", fontsize=18
    )

    ax.arrow(
        center_x,
        center_y,
        min_vmed - center_x,
        max_vmax - center_y,
        head_width=0.1,
        head_length=0.2,
        fc="white",
        ec="white",
        linewidth=0.75,
        linestyle="dashed",
    )
    ax.text(
        min_vmed - 0.2, max_vmax + 0.3, "Bajo Downforce", color="white", fontsize=18
    )

    ## Añadir logo
    x_logo = 0.36
    y_logo = 1.02
    zoom = 0.085
    add_team_logo(ax=ax, x=x_logo, y=y_logo, zoom=zoom)

    ## labels ans title
    ax.set_xlabel("Velocidad Media (km/h)", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_ylabel("Velocidad Máxima (km/h)", fontsize=18)

    try:
        sesion_type = session.session_info["Type"]
    except DataNotLoadedError:
        sesion_type = "Qualifying"

    ax.set_title(
        f"Aero Analisis\n{session.event.year} {eventname} {sesion_type}",
        fontsize=24,
        loc="right",
        fontweight="bold",
        pad=40,
    )
    ax.text(
        0.84,
        1.035,
        "*Mejor Vuelta por Equipo",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color="white",
        fontsize=17,
    )
    plt.subplots_adjust(bottom=0.07)

    # Axis limits
    ax.set_xlim(min_vmed - 0.5, max_vmed + 0.5)
    ax.set_ylim(min_vmax - 0.5, max_vmax + 0.5)

    # Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        path = f"{session.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, session.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar figura
        plot_filename = path + f"/AeroAnalisis_{eventname}_{sesion_type}.pdf"
        plt.draw()
        plt.savefig(plot_filename, format="pdf")

        # Save Avg Value
        if save_stats_circuito:
            file_path = os.path.join(
                os.getcwd(), "src", "outputs", "stats_circuitos.csv"
            )
            df_stats_circuitos = pd.read_csv(file_path, sep=",")
            corners = get_corners_df(session)
            lap_stats = get_lap_stats(session)

            ## New row
            new_row = {}
            race = session.event.EventName.split(" Grand ")[0]
            new_row["Race"] = race
            new_row["Circuito"] = get_circuito(race)
            new_row["Vmed"] = center_x
            new_row["Vmax"] = center_y
            new_row["Numero_Curvas"] = corners.shape[0]
            new_row["Numero_Curvas_Rapidas"] = sum(
                corners["Type_Corner_Speed"] == "Fast"
            )
            new_row["Numero_Curvas_Lentas"] = sum(
                corners["Type_Corner_Speed"] == "Slow"
            )
            new_row["Numero_Curvas_Medias"] = sum(
                corners["Type_Corner_Speed"] == "Medium"
            )
            new_row["Numero_Curvas_Flatout"] = sum(
                corners["Type_Corner"].str.startswith("Flatout")
            )
            new_row["Longitud"] = circuit_length

            for key, value in lap_stats.items():
                new_row[key] = value

            # Update
            mask_circuito = df_stats_circuitos["Race"] == race
            print(new_row)
            if sum(mask_circuito) == 0:
                # Add
                df_stats_circuitos.loc[len(df_stats_circuitos)] = new_row
            else:
                # Update
                df_stats_circuitos.loc[mask_circuito, list(new_row.keys())] = list(
                    new_row.values()
                )
            df_stats_circuitos.to_csv(file_path, sep=",", index=False)
    else:
        plt.show()


def plot_gap_teams(
    option,
    trend,
    teams_list,
    carreras_list,
    results,
    df_gap,
    session,
    nraces,
    save,
    use_Q1_data=False,
):
    team_gaps = {}
    team_gaps["Carrera"] = carreras_list
    team_gaps["Country"] = df_gap["Country"].to_list()
    for team in teams_list:
        drivers_in_the_team = results.loc[
            results["TeamName"] == team, "Abbreviation"
        ].to_list()
        subset_gap = df_gap[drivers_in_the_team]
        team_gaps[team] = subset_gap.min(axis=1).to_list()

    df_gap_team = pd.DataFrame(team_gaps)

    # PLOT
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    for team in teams_list:
        # Define color
        color = DICT_TEAM_COLOR[str(team)]

        # Plot Gap historic
        if trend:
            text = ""
            y_ajustada, coeficientes, text = generate_recta_ajuste(
                x=carreras_list, y=df_gap_team[team], text=text
            )
            ax.plot(
                carreras_list,
                df_gap_team[team],
                "--",
                color=color,
                alpha=0.5,
                linewidth=1,
            )
            # ax.plot(carreras_list, df_gap_team[team].rolling(window=2).mean(), color=color, linewidth=2, linestyle='solid', label=team)
            ax.plot(
                carreras_list,
                y_ajustada,
                color=color,
                linewidth=2,
                linestyle="solid",
                label=team,
            )
        else:
            ax.plot(
                carreras_list,
                df_gap_team[team],
                marker="o",
                color=color,
                linewidth=2,
                markersize=7,
            )
            ax.plot(
                carreras_list,
                df_gap_team[team],
                color=color,
                linewidth=1,
                linestyle="solid",
                label=team,
            )

    ## Plot Configure
    gap_max = df_gap_team.drop(columns=["Carrera", "Country"]).max().max()
    plot_title = "Histórico de Gap (%) Equipos"
    ax.set_title(plot_title, fontsize=20)
    ax.set_xlim([0.75, nraces + 0.25])
    ax.set_ylim([-0.1, gap_max + 0.1])
    ax.yaxis.set_inverted(True)  # Invertir eje y

    # Colocar banderas en los ticks del eje X
    for x, code in zip(carreras_list, df_gap_team["Country"]):
        bandera = get_flag_image(code)

        imagebox = OffsetImage(
            bandera,
            zoom=0.5,
        )

        ab = AnnotationBbox(
            imagebox,
            (x, gap_max + 0.1),
            frameon=False,
            box_alignment=(0.5, 1),
        )

        ax.add_artist(ab)

    # Limpiar el eje X (sin texto debajo)
    ax.set_xticks(carreras_list)
    ax.set_xticklabels([""] * len(carreras_list))
    ax.tick_params(axis="y", which="minor", left=False)
    ax.grid(axis="y", which="minor", visible=False)
    # ax.set_xlabel('Carrera')
    ax.set_ylabel("Gap (%)")

    ax.legend(bbox_to_anchor=(1.0, 1.02))

    # Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        eventname = session.event.EventName.split(" ")[0]
        path = f"{session.event.year}_{eventname}"
        path = os.path.join(
            os.getcwd(), "src", "outputs", str(session.event.year), path
        )

        # Crear el subdirectorio de GP
        if not os.path.exists(path):
            os.makedirs(path)

        # Crear el subdirectorio del nombre del evento
        path = os.path.join(path, session.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar figura
        if not use_Q1_data:
            nombre_png = f"Historico_gaps_{option}.png"
        else:
            nombre_png = f"Historico_gaps_Q1_{option}.png"

        plot_filename = os.path.join(path, nombre_png)
        plt.savefig(plot_filename, format="png", bbox_inches="tight", pad_inches=0)
    else:
        plt.show()


def plot_gap_teams_single(
    option, trend, teams_list, carreras_list, results, df_gap, session, nraces, save
):
    # Extract team gaps
    team_gaps = {}
    team_gaps["Carrera"] = carreras_list
    for team in teams_list:
        drivers_in_the_team = results.loc[
            results["TeamName"] == team, "Abbreviation"
        ].to_list()
        subset_gap = df_gap[drivers_in_the_team]
        team_gaps[team] = subset_gap.min(axis=1).to_list()

    df_gap_team = pd.DataFrame(team_gaps)
    df_gap_team.set_index("Carrera", inplace=True)
    mean_values = df_gap_team.mean().sort_values()
    teams_ordered_by_mean = mean_values.index.to_list()
    fastest_team = teams_ordered_by_mean[0]

    # Load and treat mejoras csv
    mejoras_path = os.path.join(os.getcwd(), "src", "outputs", "mejoras.csv")
    df_mejoras = pd.read_csv(mejoras_path, sep=",")

    for team in teams_list:
        df_mejoras[team] = df_mejoras[team].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) else []
        )
        df_mejoras[f"Count{team}"] = df_mejoras[team].apply(lambda x: len(x))

    # print(df_mejoras[['Carrera', 'CountMcLaren', 'CountMercedes', 'CountFerrari', 'CountRed Bull Racing', 'CountAston Martin']])

    # Plot
    for team in teams_ordered_by_mean[1:]:
        plt.close("all")
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

        # Define color
        color_fastest = ff1.plotting.get_team_color(fastest_team, session)
        color = ff1.plotting.get_team_color(team, session)

        # Plot Fastest team
        ax.plot(
            carreras_list,
            df_gap_team[fastest_team],
            marker="o",
            color=color_fastest,
            linewidth=2,
            markersize=7,
        )
        ax.plot(
            carreras_list,
            df_gap_team[fastest_team],
            color=color_fastest,
            linewidth=1,
            linestyle="solid",
            label=fastest_team,
        )

        # Plot Comparative team
        ax.plot(
            carreras_list,
            df_gap_team[team],
            marker="o",
            color=color,
            linewidth=2,
            markersize=7,
        )
        ax.plot(
            carreras_list,
            df_gap_team[team],
            color=color,
            linewidth=1,
            linestyle="solid",
            label=team,
        )

        ## Plot Mejoras
        gap_max = df_gap_team.max().max()
        for i, carrera in enumerate(carreras_list[1:]):
            gap_fastest = df_gap_team.loc[carrera, fastest_team]
            gap_team = df_gap_team.loc[carrera, team]

            # Check if team was faster than fastest overall team
            if gap_fastest > gap_team:
                aux = gap_fastest
                gap_fastest = gap_team
                gap_team = aux

                n_mejoras_fastest = df_mejoras.loc[
                    df_mejoras["Carrera"] == carrera, f"Count{team}"
                ].iloc[0]
                n_mejoras_team = df_mejoras.loc[
                    df_mejoras["Carrera"] == carrera, f"Count{fastest_team}"
                ].iloc[0]
                color_1 = color
                color_2 = color_fastest
            else:
                n_mejoras_fastest = df_mejoras.loc[
                    df_mejoras["Carrera"] == carrera, f"Count{fastest_team}"
                ].iloc[0]
                n_mejoras_team = df_mejoras.loc[
                    df_mejoras["Carrera"] == carrera, f"Count{team}"
                ].iloc[0]
                color_1 = color_fastest
                color_2 = color

            ### Only Fastest Team
            if (n_mejoras_fastest > 0) & (n_mejoras_team == 0):
                ax.bar(
                    carrera,
                    gap_max + 0.2,
                    bottom=gap_fastest,
                    width=0.3,
                    color=color_1,
                    alpha=0.6,
                )
                # ax.text(carrera, gap_fastest + 0.1, n_mejoras_fastest, ha='center', va='top', fontsize=15, color='white')

            ### Only Team
            if (n_mejoras_team > 0) & (n_mejoras_fastest == 0):
                ax.bar(
                    carrera,
                    gap_max + 0.2,
                    bottom=gap_team,
                    width=0.3,
                    color=color_2,
                    alpha=0.6,
                )
                # ax.text(carrera, gap_team + 0.1, n_mejoras_team, ha='center', va='top', fontsize=15, color='white')

            ## Both
            if (n_mejoras_team > 0) & (n_mejoras_fastest > 0):
                ax.bar(
                    carrera,
                    gap_max + 0.2,
                    bottom=gap_team,
                    width=0.3,
                    color=color_2,
                    alpha=0.6,
                )
                # ax.text(carrera, gap_team + 0.1, n_mejoras_team, ha='center', va='top', fontsize=15, color='white')

                ax.bar(
                    carrera,
                    gap_team,
                    bottom=gap_fastest,
                    width=0.3,
                    color=color_1,
                    alpha=0.6,
                )
                # ax.text(carrera, gap_fastest + 0.1, n_mejoras_fastest, ha='center', va='top', fontsize=15, color='white')

        ## Plot Configure
        plot_title = f"Histórico de Gap {team} vs {fastest_team}"
        ax.set_title(plot_title, fontsize=20)
        ax.set_xlim([0.75, nraces + 0.25])
        ax.set_ylim([-0.1, gap_max + 0.1])
        ax.yaxis.set_inverted(True)  # Invertir eje y

        # Colocar banderas en los ticks del eje X
        for i, nombre in enumerate(carreras_list):
            if nombre not in DICT_RACE_COUNTRYCODE:
                continue
            codigo = DICT_RACE_COUNTRYCODE[nombre]["countrycode"].lower()
            bandera = get_flag_image(codigo)
            imagebox = OffsetImage(bandera, zoom=0.5)
            ab = AnnotationBbox(
                imagebox, (nombre, gap_max + 0.1), frameon=False, box_alignment=(0.5, 1)
            )
            ax.add_artist(ab)

        # Limpiar el eje X (sin texto debajo)
        ax.set_xticks(carreras_list)
        ax.set_xticklabels([""] * len(carreras_list))
        ax.tick_params(axis="y", which="minor", left=False)
        ax.grid(axis="y", which="minor", visible=False)
        ax.set_xlabel("Carrera")
        ax.set_ylabel("Gap (%)")

        ax.legend(bbox_to_anchor=(1.0, 1.02))

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
            nombre_png = f"Historico_gaps_{option}_{team}.png"

            plot_filename = os.path.join(path, nombre_png)
            plt.savefig(plot_filename, format="png", bbox_inches="tight", pad_inches=0)
        else:
            plt.show()


def plot_gap_teams_single_difference(
    option, trend, teams_list, carreras_list, results, df_gap, session, nraces, save
):
    # Extract team gaps
    team_gaps = {}
    team_gaps["Carrera"] = carreras_list
    for team in teams_list:
        drivers_in_the_team = results.loc[
            results["TeamName"] == team, "Abbreviation"
        ].to_list()
        subset_gap = df_gap[drivers_in_the_team]
        team_gaps[team] = subset_gap.min(axis=1).to_list()

    df_gap_team = pd.DataFrame(team_gaps)
    df_gap_team.set_index("Carrera", inplace=True)
    mean_values = df_gap_team.mean().sort_values()
    teams_ordered_by_mean = mean_values.index.to_list()
    fastest_team = teams_ordered_by_mean[0]

    # Load and treat mejoras csv
    mejoras_path = os.path.join(os.getcwd(), "src", "outputs", "mejoras.csv")
    df_mejoras = pd.read_csv(mejoras_path, sep=",")

    for team in teams_list:
        df_mejoras[team] = df_mejoras[team].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) else []
        )
        df_mejoras[f"Count{team}"] = df_mejoras[team].apply(lambda x: len(x))

    # Plot
    for team in teams_ordered_by_mean[1:]:
        plt.close("all")
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

        # Define color
        color = ff1.plotting.get_team_color(team, session)
        color_fastest = ff1.plotting.get_team_color(fastest_team, session)

        # Calculate trend
        text = ""
        y_ajustada, coeficientes, text = generate_recta_ajuste(
            x=carreras_list, y=df_gap_team[team], text=text
        )
        print(f"team = {team}")
        print(
            f"Alcanza a {fastest_team} en la carrera {round(-coeficientes[1] / coeficientes[0], 0)}"
        )
        print(text)

        # Plot Comparative team
        df_gap_team[f"Gap{team}"] = df_gap_team[team] - df_gap_team[fastest_team]
        ax.plot(
            carreras_list,
            df_gap_team[f"Gap{team}"],
            color=color,
            alpha=0.7,
            linewidth=1,
            linestyle="solid",
            label=team,
        )
        ax.plot(
            carreras_list,
            y_ajustada,
            color=color,
            linewidth=2,
            linestyle="dashed",
            label=team,
        )
        ax.plot(
            carreras_list,
            [0.0] * len(carreras_list),
            color=color_fastest,
            linewidth=1,
            linestyle="solid",
            label=fastest_team,
        )

        ## Plot Configure
        gap_max = df_gap_team.max().max()
        gap_min = df_gap_team[f"Gap{team}"].min()
        plot_title = f"Histórico de Gap {team} vs {fastest_team}"
        ax.set_title(plot_title, fontsize=20)
        ax.set_xlim([0.75, nraces + 0.25])
        ax.set_ylim([gap_min - 0.1, gap_max + 0.1])

        # Colocar banderas en los ticks del eje X
        for i, nombre in enumerate(carreras_list):
            if nombre not in DICT_RACE_COUNTRYCODE:
                continue
            codigo = DICT_RACE_COUNTRYCODE[nombre]["countrycode"].lower()
            bandera = get_flag_image(codigo)
            imagebox = OffsetImage(bandera, zoom=0.5)
            ab = AnnotationBbox(
                imagebox, (nombre, gap_min - 0.1), frameon=False, box_alignment=(0.5, 1)
            )
            ax.add_artist(ab)

        # Limpiar el eje X (sin texto debajo)
        ax.set_xticks(carreras_list)
        ax.set_xticklabels([""] * len(carreras_list))
        ax.tick_params(axis="y", which="minor", left=False)
        ax.grid(axis="y", which="minor", visible=False)
        ax.set_xlabel("Carrera")
        ax.set_ylabel(f"Delta Gap {fastest_team}(%)")
        ax.legend(bbox_to_anchor=(1.0, 1.02))

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
            nombre_png = f"Historico_gaps_{option}_{team}.png"

            plot_filename = os.path.join(path, nombre_png)
            plt.savefig(plot_filename, format="png", bbox_inches="tight", pad_inches=0)
        else:
            plt.show()


def plot_gap_teams_by_cluster(teams_list, carreras_list, results, df_gap, save):
    # EXTRACT GAPS OF TEAMS
    team_gaps = {}
    team_gaps["Carrera"] = carreras_list
    for team in teams_list:
        drivers_in_the_team = results.loc[
            results["TeamName"] == team, "Abbreviation"
        ].to_list()
        subset_gap = df_gap[drivers_in_the_team]
        team_gaps[team] = subset_gap.min(axis=1).to_list()

    df_gap_team = pd.DataFrame(team_gaps)

    # DOWNLOAD STATS OF CIRCUIT
    stats_path = os.path.join(os.getcwd(), "src", "outputs", "stats_circuitos.csv")
    df_stats_circuitos = pd.read_csv(stats_path, sep=",")
    df_stats_circuitos["Carrera_int"] = [
        i + 1 for i in range(df_stats_circuitos.shape[0])
    ]
    df_stats_circuitos = df_stats_circuitos[["Carrera_int", "Cluster"]]
    df_stats_circuitos["Carrera"] = df_stats_circuitos["Carrera_int"]
    df_stats_circuitos.pop("Carrera_int")

    df_gap_cluster = df_gap_team.merge(df_stats_circuitos, on="Carrera", how="inner")
    print(df_gap_cluster)

    cmap = plt.cm.get_cmap("prism", df_gap_cluster["Cluster"].nunique())
    palette = [cmap(i) for i in range(df_gap_cluster["Cluster"].nunique())]

    # PLOT
    for team in teams_list:
        plt.close("all")
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

        ax = sns.barplot(
            data=df_gap_cluster,
            x="Cluster",
            y=team,
            hue="Cluster",
            errorbar=None,
            palette=palette,
        )
        ax.bar_label(ax.containers[0], fontsize=10)
        plot_title = f"Gap Medio (%) por Cluster - {team}"
        ax.set_title(plot_title, fontsize=20)

        # Save
        if save:
            # Crear subdirectorio con el nombre del pais_año para guardar los outputs
            path = os.path.join(os.getcwd(), "src", "outputs")

            # Guardar figura
            nombre_png = f"Gap_vs_Cluster_{team}.png"

            plot_filename = os.path.join(path, nombre_png)
            plt.savefig(plot_filename, format="png", bbox_inches="tight", pad_inches=0)
        else:
            plt.show()
    return df_gap_cluster


def plot_correlation_matrix(save):
    # Load data
    stats_path = os.path.join(os.getcwd(), "src", "outputs", "stats_circuitos.csv")
    df_stats_circuitos = pd.read_csv(stats_path, sep=",")

    # Clean - Only features
    df_stats_circuitos.drop(
        columns=["Race", "Circuito", "Cluster"], axis=1, inplace=True
    )

    ##################### PLOT ########################################################
    plt.close("all")
    fig, ax = plt.subplots(1, figsize=(15, 6), dpi=700)

    correlation_mat = df_stats_circuitos.corr()
    sns.heatmap(correlation_mat, annot=True, ax=ax)

    ax.set_title(
        "Mapa de Correlación Variables Circuitos", fontsize=24, fontweight="bold"
    )
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    ## Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        path = os.path.join(os.getcwd(), "src", "outputs")

        # Guardar figura
        nombre_png = "Correlation_Matrix_Stats_Circuitos.png"

        plot_filename = os.path.join(path, nombre_png)
        plt.savefig(plot_filename, format="png", bbox_inches="tight", pad_inches=0.5)
    else:
        plt.show()


def plot_next_race_ranking(session, df_gap_cluster, option, save):
    ##################### COMMON ########################################################
    race = session.session_info["Meeting"]["Name"].split(" Grand Prix")[
        0
    ]  # Obtener carrera

    ## Get df_horarios
    path_outputs = os.path.join(os.getcwd(), "src", "outputs")
    df_horarios = pd.read_csv(os.path.join(path_outputs, "horarios_calendario.csv"))
    df_horarios["FechaIni"] = pd.to_datetime(df_horarios["FechaIni"], format="%d/%m/%Y")
    df_horarios["FechaFin"] = pd.to_datetime(df_horarios["FechaFin"], format="%d/%m/%Y")

    ## Get next_race
    index_actual = df_horarios.loc[df_horarios["Race"] == race,].index
    next_race = df_horarios.loc[index_actual + 1, "Race"].iloc[0]

    ## Get last year pole time
    last_year_pole_time = get_last_year_pole_time(session, next_race)

    ## Get next cluster
    stats_path = os.path.join(os.getcwd(), "src", "outputs", "stats_circuitos.csv")
    df_stats_circuitos = pd.read_csv(stats_path, sep=",")
    print("\n")
    print(next_race)
    next_cluster = df_stats_circuitos.loc[
        df_stats_circuitos["Race"] == next_race, "Cluster"
    ].iloc[0]
    print(next_cluster)

    ## FILTER ROW
    mask_cluster = df_gap_cluster["Cluster"] == next_cluster
    if option == "last_race":
        # Get last value of Cluster
        row = df_gap_cluster[mask_cluster].iloc[-1]
        index = row.index.to_list()[1:-1]
        values = row.values.tolist()[1:-1]
        values_seconds = [
            round((value * last_year_pole_time.total_seconds()) / 100, 3)
            for value in values
        ]
        values = [round(value, 3) for value in values]

        ## Create color
        team_colors = []
        for team in index:
            team_colors.append(ff1.plotting.get_team_color(team, session))

        df_row = pd.DataFrame(
            {
                "Equipos": index,
                "Gaps(%)": values,
                "Gaps(s)": values_seconds,
                "Colors": team_colors,
            }
        )
        df_row.sort_values(by="Gaps(s)", ascending=True, inplace=True)
        print(df_row)
    elif option == "med":
        subset = df_gap_cluster[mask_cluster].copy()
        subset.drop(columns=["Carrera", "Cluster"], axis=1, inplace=True)
        medias = subset.mean(numeric_only=True)

        index = medias.index.to_list()
        values = medias.values.tolist()
        values_seconds = [
            round((value * last_year_pole_time.total_seconds()) / 100, 3)
            for value in values
        ]
        values = [round(value, 3) for value in values]

        ## Create color
        team_colors = []
        for team in index:
            team_colors.append(ff1.plotting.get_team_color(team, session))

        df_row = pd.DataFrame(
            {
                "Equipos": index,
                "Gaps(%)": values,
                "Gaps(s)": values_seconds,
                "Colors": team_colors,
            }
        )
        df_row.sort_values(by="Gaps(s)", ascending=True, inplace=True)
        print(df_row)

    ##################### PLOT ########################################################
    plt.close("all")
    fig, ax = plt.subplots(1, figsize=(12, 12), dpi=700)

    barplot = ax.barh(df_row["Equipos"], df_row["Gaps(s)"], color=df_row["Colors"])
    formatted_labels = df_row["Gaps(s)"]
    ax.bar_label(
        barplot,
        labels=formatted_labels,
        padding=5,
        color="white",
        fontweight="bold",
        fontsize=16,
    )

    ax.set_ylabel("")
    ax.set_xlabel("Gap (s)", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=18)
    maximo = max(df_row["Gaps(s)"]) + 0.1
    minimo = min(df_row["Gaps(s)"])
    ax.set_xlim([minimo, maximo])
    ax.set_ylim(-0.5, len(df_row["Equipos"]) - 0.5)
    ax.tick_params(axis="y", which="minor", left=False)
    ax.grid(axis="y", which="minor", visible=False)
    ax.invert_yaxis()

    ax.set_title(f"Ranking {next_race}", fontsize=26, fontweight="bold")

    ## Añadir imagen del logo
    path = os.path.join(os.getcwd(), "src", "outputs", "logo_graficos.png")
    img = mpimg.imread(path)
    imagebox = OffsetImage(img, zoom=0.12, interpolation="bicubic", resample=True)
    ab = AnnotationBbox(
        imagebox,
        (1.0, 0.8),  # Coordenadas (x, y) para la esquina inferior izquierda
        xycoords="axes fraction",  # Coordenadas relativas al área de dibujo
        box_alignment=(
            1,
            0,
        ),  # Alinear la esquina inferior izquierda de la imagen con las coordenadas
        frameon=False,  # Sin marco alrededor de la imagen
    )
    ax.add_artist(ab)

    # Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        path = os.path.join(os.getcwd(), "src", "outputs")

        # Guardar figura
        nombre_png = f"Ranking_Equipos_{next_race}_{option}.png"

        plot_filename = os.path.join(path, nombre_png)
        plt.savefig(plot_filename, format="png", bbox_inches="tight", pad_inches=0.5)
    else:
        plt.show()


def plot_gap_driver(
    option,
    drivers_list,
    carreras_list,
    nraces,
    df_gap,
    session,
    save,
    use_Q1_data=False,
):
    # PLOT
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    for driver in drivers_list:
        # Define color
        if driver == "DOO":
            color = ff1.plotting.get_driver_color("GAS", session)
        else:
            color = ff1.plotting.get_driver_color(driver, session)

        # Plot Gap historic
        ax.plot(
            carreras_list,
            df_gap[driver],
            marker="o",
            color=color,
            linewidth=2,
            markersize=7,
        )
        ax.plot(
            carreras_list,
            df_gap[driver],
            color=color,
            linewidth=1,
            linestyle="solid",
            label=driver,
        )

    ## Plot Configure
    gap_max = df_gap.drop(columns="Carrera").max().max()
    plot_title = "Histórico de Gap (%) Pilotos"
    ax.set_title(plot_title, fontsize=20)
    ax.set_xlim([0.75, nraces + 0.25])
    ax.set_ylim([-0.1, gap_max + 0.1])
    ax.yaxis.set_inverted(True)  # Invertir eje y

    # Colocar banderas en los ticks del eje X
    for i, nombre in enumerate(carreras_list):
        if nombre not in DICT_RACE_COUNTRYCODE:
            continue
        codigo = DICT_RACE_COUNTRYCODE[nombre]["countrycode"].lower()
        bandera = get_flag_image(codigo)
        imagebox = OffsetImage(bandera, zoom=0.5)
        ab = AnnotationBbox(
            imagebox, (nombre, gap_max + 0.1), frameon=False, box_alignment=(0.5, 1)
        )
        ax.add_artist(ab)

    # Limpiar el eje X (sin texto debajo)
    ax.set_xticks(carreras_list)
    ax.set_xticklabels([""] * len(carreras_list))
    ax.tick_params(axis="y", which="minor", left=False)
    ax.grid(axis="y", which="minor", visible=False)
    ax.set_xlabel("Carrera")
    ax.set_ylabel("Gap (%)")

    ax.legend(bbox_to_anchor=(1.0, 1.02))

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
        if not use_Q1_data:
            nombre_png = f"Historico_gaps_{option}.png"
        else:
            nombre_png = f"Historico_gaps_Q1_{option}.png"

        plot_filename = os.path.join(path, nombre_png)
        plt.savefig(plot_filename, format="png", bbox_inches="tight", pad_inches=0)
    else:
        plt.show()


def make_plot_gap(
    session=None,
    results=None,
    option="teams",
    trend=False,
    save=True,
    use_Q1_data=False,
):
    top_teams = ["McLaren", "Red Bull Racing", "Ferrari", "Mercedes", "Aston Martin"]

    # Load gap dfs
    if use_Q1_data:
        file_path = os.path.join(os.getcwd(), "src", "outputs", "gap_pilotos_Q1.csv")
    else:
        file_path = os.path.join(os.getcwd(), "src", "outputs", "gap_pilotos.csv")
    df_gap = pd.read_csv(file_path, sep=",")

    # Common
    drivers_list = df_gap.columns.to_list()
    drivers_list = [driver for driver in drivers_list if driver != "Carrera"]
    carreras_list = df_gap["Carrera"].to_list()
    nraces = df_gap.shape[0]

    # OPTIONS
    if option == "drivers":
        if session is not None:
            plot_gap_driver(
                option,
                drivers_list,
                carreras_list,
                nraces,
                df_gap,
                session,
                save,
                use_Q1_data,
            )

    elif option == "teams":
        # Extract gap of teams
        if results is not None:
            results["TeamName"] = results["equipo"]
            results["Abbreviation"] = results["Driver"]
        else:
            if session is not None:
                results = session.results
            else:
                raise ValueError("session or results not provided to make_plot_gap")

        teams_list = np.unique(results["TeamName"].to_list())
        # plot
        plot_gap_teams(
            option,
            trend,
            teams_list,
            carreras_list,
            results,
            df_gap,
            session,
            nraces,
            save,
            use_Q1_data,
        )

    elif option == "top_teams":
        # Extract gap of teams
        if session is not None:
            results = session.results
            plot_gap_teams(
                option,
                trend,
                top_teams,
                carreras_list,
                results,
                df_gap,
                session,
                nraces,
                save,
                use_Q1_data,
            )

    elif option == "top_teams_single":
        if session is not None:
            results = session.results
            plot_gap_teams_single(
                option,
                trend,
                top_teams,
                carreras_list,
                results,
                df_gap,
                session,
                nraces,
                save,
            )

    elif option == "top_teams_single_difference":
        if session is not None:
            results = session.results
            plot_gap_teams_single_difference(
                option,
                trend,
                top_teams,
                carreras_list,
                results,
                df_gap,
                session,
                nraces,
                save,
            )

    elif option == "teams_by_cluster":
        # Plot Team Gaps by cluster
        if results is not None:
            results["TeamName"] = results["equipo"]
            results["Abbreviation"] = results["Driver"]
        else:
            if session is not None:
                results = session.results
            else:
                raise ValueError("session or results not provided to make_plot_gap")

        teams_list = np.unique(results["TeamName"].to_list())
        df_gap_cluster = plot_gap_teams_by_cluster(
            teams_list, carreras_list, results, df_gap, save
        )

        # Plot correlation matrix of variables in cluster
        plot_correlation_matrix(save)

        # Plot Ranking Next race
        # option_ranking = 'last_race'
        option_ranking = "med"
        plot_next_race_ranking(session, df_gap_cluster, option_ranking, save)


def plot_comparativa_curvas(df_circuito, save=True):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=700)

    # Plot
    colors = ["yellow", "red"]
    df_circuito.T.plot(kind="bar", ax=ax, color=colors, width=0.4)

    # Personalización
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title("Comparativa de curvas", fontweight="bold", fontsize=25)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(fontsize=14, title="Circuito", title_fontsize=15)
    plt.xticks(rotation=0)

    # Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        path = os.path.join(os.getcwd(), "src", "outputs")
        nombre_png = f"Comparativa_curvas_Hungaroring_vs_Zandvoort.png"

        plot_filename = os.path.join(path, nombre_png)
        plt.savefig(plot_filename, format="png", bbox_inches="tight", pad_inches=0)
    else:
        plt.show()


def plot_stats_circuitos(df_stats, save=True):
    # Plot
    fig, ax = plt.subplots(figsize=(12, 12), dpi=700)

    for i in range(df_stats.shape[0]):
        ax.plot(
            df_stats["Vmed"].iloc[i],
            df_stats["Vmax"].iloc[i],
            "o",
            color=DICT_EVENT[df_stats["Race"].iloc[i]]["color"],
            markersize=14,
        )

        # Etiqueta del equipo
        ax.text(
            df_stats["Vmed"].iloc[i] + 1,
            df_stats["Vmax"].iloc[i] + 1,
            DICT_EVENT[df_stats["Race"].iloc[i]]["name"],
            color=DICT_EVENT[df_stats["Race"].iloc[i]]["color"],
            fontsize=16,
        )

    ax.set_ylabel("Velocidad Máxima (km/h)", fontsize=16)
    ax.set_xlabel("Velocidad Media (km/h)", fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_title("Comparativa Circuitos", fontweight="bold", fontsize=25, pad=20)

    # Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        path = os.path.join(os.getcwd(), "src", "outputs")
        nombre_png = f"Comparativa_circuitos.png"

        plot_filename = os.path.join(path, nombre_png)
        plt.savefig(plot_filename, format="png", bbox_inches="tight", pad_inches=0.3)
    else:
        plt.show()


def plot_stats_circuitos_pca(df_stats, save=True):
    # Calculate principal components df
    ## Normalize
    X = df_stats.drop(columns=["Race", "Circuito"])
    X_scaled = StandardScaler().fit_transform(X)

    ## Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["Race"] = df_stats["Race"]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12), dpi=700)

    for i in range(df_stats.shape[0]):
        ax.plot(
            df_pca["PC1"].iloc[i],
            df_pca["PC2"].iloc[i],
            "o",
            color=DICT_EVENT[df_pca["Race"].iloc[i]]["color"],
            markersize=14,
        )

        # Etiqueta del equipo
        ax.text(
            df_pca["PC1"].iloc[i] + 0.07,
            df_pca["PC2"].iloc[i] + 0.07,
            DICT_EVENT[df_pca["Race"].iloc[i]]["name"],
            color=DICT_EVENT[df_pca["Race"].iloc[i]]["color"],
            fontsize=16,
        )

    ax.set_ylabel("PC2", fontsize=16)
    ax.set_xlabel("PC1", fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_title("Comparativa Circuitos PCA", fontweight="bold", fontsize=25, pad=20)

    # Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        path = os.path.join(os.getcwd(), "src", "outputs")
        nombre_png = f"Comparativa_circuitos_pca.png"

        plot_filename = os.path.join(path, nombre_png)
        plt.savefig(plot_filename, format="png", bbox_inches="tight", pad_inches=0.3)
    else:
        plt.show()


def plot_stats_circuitos_pca_cluster(df_stats, save=True):
    # Calculate principal components df
    ## Normalize
    X = df_stats.drop(columns=["Race", "Circuito", "Cluster"])

    X_scaled = StandardScaler().fit_transform(X)
    ## Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["Race"] = df_stats["Race"]

    print("Explained Variance Ratio (%)")
    ratios = pca.explained_variance_ratio_
    print(ratios * 100)

    ## Calculate Clusters
    kmeans = KMeans(n_clusters=5, random_state=42)
    df_pca["Cluster"] = kmeans.fit_predict(X_pca)
    cmap = plt.cm.get_cmap("prism", df_pca["Cluster"].nunique())

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12), dpi=700)

    for cluster in df_pca["Cluster"].unique():
        cluster_points = df_pca[df_pca["Cluster"] == cluster][["PC1", "PC2"]].values

        # CLUSTERS
        if len(cluster_points) >= 3:
            hull = ConvexHull(cluster_points)
            # Rellenar todo el polígono del hull, para cluster de mas de uno
            ax.fill(
                cluster_points[hull.vertices, 0],
                cluster_points[hull.vertices, 1],
                alpha=0.2,
                color=cmap(cluster),
                label=f"Cluster {cluster}",
            )
            # Colorear el contorno
            for simplex in hull.simplices:
                ax.fill(
                    cluster_points[simplex, 0],
                    cluster_points[simplex, 1],
                    alpha=0.5,
                    color=cmap(cluster),
                )
        elif len(cluster_points) == 2:
            # Dibujar un polígono simple entre los dos puntos
            x = cluster_points[:, 0]
            y = cluster_points[:, 1]
            ax.fill(
                [x[0], x[1], x[1], x[0]],
                [y[0] + 0.05, y[1] + 0.05, y[1] - 0.05, y[0] - 0.05],
                alpha=0.5,
                color=cmap(cluster),
                label=f"Cluster {cluster}",
            )

        elif len(cluster_points) == 1:
            # Dibujar un círculo alrededor del punto
            ax.add_patch(
                plt.Circle(
                    (cluster_points[0, 0], cluster_points[0, 1]),
                    0.1,
                    color=cmap(cluster),
                    alpha=0.5,
                    label=f"Cluster {cluster}",
                )
            )

        # Dibujar puntos del cluster
        ax.plot(
            cluster_points[:, 0],
            cluster_points[:, 1],
            "o",
            color=cmap(cluster),
            markersize=14,
        )

        # Añadir nombres de circuitos
        for i, row in df_pca[df_pca["Cluster"] == cluster].iterrows():
            ax.text(
                row["PC1"] + 0.07,
                row["PC2"] + 0.07,
                DICT_EVENT[row["Race"]]["name"],
                color=cmap(row["Cluster"]),
                fontsize=16,
            )

    ax.set_ylabel("PC2", fontsize=16)
    ax.set_xlabel("PC1", fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_title("Comparativa Circuitos PCA", fontweight="bold", fontsize=25, pad=20)
    ax.legend(loc="best")

    # Save
    if save:
        # Crear subdirectorio con el nombre del pais_año para guardar los outputs
        path = os.path.join(os.getcwd(), "src", "outputs")
        nombre_png = f"Comparativa_circuitos_pca_cluster.png"

        plot_filename = os.path.join(path, nombre_png)
        plt.savefig(plot_filename, format="png", bbox_inches="tight", pad_inches=0.3)

        ## Save cluster num in stats_circuitos
        df_stats["Cluster"] = df_pca["Cluster"]
        file_path = os.path.join(os.getcwd(), "src", "outputs", "stats_circuitos.csv")
        df_stats.to_csv(file_path, sep=",", index=False)
    else:
        plt.show()
