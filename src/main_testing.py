# Library
import os
import logging
import numpy as np
import pandas as pd
import fastf1 as ff1
from fastf1 import plotting
from fastf1 import utils

from io import StringIO

from utils.functions import (calculate_deltas_twodrivers, show_laptime,
                             create_corner_tableclasif, get_final_sesion_table,
                             convert_img_to_ig, resize_and_pad_images, update_qualy_stats, get_lap_number,
                             convert_tiktok_image, convert_tiktok_image_by_rangos, convert_image_tiktok_margin)
from utils.graphs import (graph_telemetry, create_minisectors_graph, graph_long_acc, graph_lat_acc,
                          graph_comparaison_telemetry_twolaps, graph_comparaison_firststraight,
                          graph_comparaison_finalpart, rectas_imola, obten_subset_curvas,
                          plot_time_by_distance, graph_given_corner, graph_weather,
                          graph_given_distances, test_mapgraph, make_countplot)
from src.grapher.graph_by_type import graf_corners_onedriver, graph_analisis_rectasDRS
from src.logger.logging_utils import config_logger, parse_logger_to_df
from creators.creator import create_qualy_sesion

def main():
    # 1) Config de ff1 and in general
    ff1.plotting.setup_mpl()
    ff1.Cache.enable_cache("./cache")
    pd.options.mode.chained_assignment = None

    # Cargar las sesiones
    year1 = 2023
    year2 = 2024
    grand_prix = 'Bahrein'
    session = 'Q'

    quali1 = ff1.get_session(year1, grand_prix, session)
    quali2 = ff1.get_session(year2, grand_prix, session)

    quali1.load()
    quali2.load()

    # Cargar las curvas del circuito
    corners = quali1.get_circuit_info().corners

    # 2) Cargar las vueltas rapidas de ALO
    driver = 'ALO'

    # Cargamos todas sus vueltas
    laps_driver_2023 = quali1.laps.pick_drivers(driver)
    laps_driver_2024 = quali2.laps.pick_drivers(driver)

    # Cargamos la vuelta rapida de cada uno
    fastest_2023 = laps_driver_2023.pick_fastest()
    fastest_2024 = laps_driver_2024.pick_fastest()

    # calculamos su telemetria
    telemetry_2023 = fastest_2023.get_telemetry().add_distance()
    telemetry_2024 = fastest_2024.get_telemetry().add_distance()

    # 3) Ver el delta de ambas vueltas
    show_laptime(driver + '2023', telemetry_2023, fastest_2023)
    show_laptime(driver + '2024', telemetry_2024, fastest_2024)

    calculate_deltas_twodrivers(telemetry_2023, telemetry_2024,
                                fastest_2023, fastest_2024, driver + '2023', driver + '2024')

    # 4) Graph telemetry
    #graph_telemetry(fastest_driver_1=fastest_2023, fastest_driver_2=fastest_2024,
    #                driver_1=driver + '2023', driver_2=driver + '2024', corners=corners, quali=quali2, save=True)

    # 5) Graph track dominance
    create_minisectors_graph(telemetry_driver_1=telemetry_2023, telemetry_driver_2=telemetry_2024,
                             fastest_driver_1=fastest_2023, fastest_driver_2=fastest_2024,
                             driver_1=driver + '2023', driver_2=driver + '2024', quali=quali2,
                             n_minisectors=100, save=True)

    # 6) Clasificacion de las curvas del circuito por tipo
    corners = create_corner_tableclasif(corners, telemetry_2024)

    # 7) Graficar telemetria con su tipo
    # Parametros del grafico
    team_driver = fastest_2024['Team']
    color = ff1.plotting.team_color(team_driver)

    # Lanzamos la funcion
    graf_corners_onedriver(corners, telemetry_2024, color)

def main2(year, grand_prix, session, driver1, driver2, minisectors, acc, rectasDRS, update_sesion, update_stats):
    # 1) Config de ff1 and in general
    ff1.plotting.setup_mpl(color_scheme='fastf1')
    ff1.Cache.enable_cache("./cache")
    pd.options.mode.chained_assignment = None

    # logging
    log_capture = config_logger()

    # Cargar las sesiones
    quali = ff1.get_session(year, grand_prix, session)
    quali.load()

    # Cargar las curvas del circuito
    corners = quali.get_circuit_info().corners

    # Obtener el contenido capturado
    log_capture.seek(0)
    captured_logs = log_capture.read()
    df_logs = parse_logger_to_df(captured_logs)

    # Obtener la tabla resultado de la sesion
    df_results = get_final_sesion_table(quali)

    # ------------------------- CREATE RACE HTML  ----------------------------------------------------------------------
    if update_sesion:
        # Create qualy sesion
        create_qualy_sesion(quali, df_results)

    # ------------------------- CREATE RACE HTML  ----------------------------------------------------------------------
    if update_stats:
        # Update stats
        if quali.name != 'Sprint Qualifying':
            update_qualy_stats(df_results)

    # 2) Cargar las vueltas rapidas
    # Cargamos todas sus vueltas
    laps_driver_1 = quali.laps.pick_drivers(driver1)
    laps_driver_2 = quali.laps.pick_drivers(driver2)

    lap_number_driver_1 = get_lap_number(quali, laps_driver_1)
    lap_number_driver_2 = get_lap_number(quali, laps_driver_2)

    # Cargamos la vuelta rapida de cada uno
    #fastest_driver_1 = laps_driver_1.pick_fastest()
    fastest_driver_1 = laps_driver_1[laps_driver_1['LapNumber'] == lap_number_driver_1].iloc[0]

    #fastest_driver_2 = laps_driver_2.pick_fastest()
    #print(laps_driver_2[['Driver', 'LapNumber', 'LapTime', 'Deleted']])
    fastest_driver_2 = laps_driver_2[laps_driver_2['LapNumber'] == lap_number_driver_2].iloc[0]


    # calculamos su telemetria
    telemetry_driver_1 = fastest_driver_1.get_telemetry().add_distance()
    telemetry_driver_2 = fastest_driver_2.get_telemetry().add_distance()

    # 3) Ver el delta de ambas vueltas
    show_laptime(driver1, telemetry_driver_1, fastest_driver_1)
    show_laptime(driver2, telemetry_driver_2, fastest_driver_2)

    calculate_deltas_twodrivers(telemetry_driver_1, telemetry_driver_2,
                                fastest_driver_1, fastest_driver_2, driver1, driver2)

    # 4) Graph telemetry
    graph_telemetry(
        fastest_driver_1=fastest_driver_1, 
        fastest_driver_2=fastest_driver_2,
        driver_1=driver1, 
        driver_2=driver2, 
        corners=corners, 
        event=quali, 
        save=True
    )

    # Comparativa hasta la Frenada
    #graph_comparaison_firststraight(fastest_driver_1, fastest_driver_2, corners, save=True)
    #graph_comparaison_finalpart(fastest_driver_1, fastest_driver_2, corners, quali, save=True)
    #plot_time_by_distance(fastest_driver_1, fastest_driver_2, distance_ini=1800, distance_fin=2200,
    #                      driver1=driver1, driver2=driver2, quali, save=True)

    #rectas_imola(fastest_driver_1, fastest_driver_2, corners, driver1, driver2, quali, save=True)
    #obten_subset_curvas(fastest_driver_1, fastest_driver_2, corners, driver1, driver2, quali, save=True)

    # 5) Graph track dominance
    if minisectors:
        create_minisectors_graph(telemetry_driver_1=telemetry_driver_1, telemetry_driver_2=telemetry_driver_2,
                                 fastest_driver_1=fastest_driver_1, fastest_driver_2=fastest_driver_2,
                                 driver_1=driver1, driver_2=driver2, event=quali,
                                 n_minisectors=25, save=True)

    # 6) Clasificacion de las curvas del circuito por tipo
    corners = create_corner_tableclasif(corners, telemetry_driver_1)
    #print(corners[['Angle', 'Type_Corner', 'Type_Corner_Speed']])

    # 7) Graficar telemetria con su tipo
    # Parametros del grafico
    #team_driver = fastest_driver_1['Team']
    #color = ff1.plotting.team_color(team_driver)

    # Lanzamos la funcion
    #graf_corners_onedriver(corners, telemetry_driver_1, color)

    # 8) Acc. Lateral y Longitudinal
    if acc:
        graph_lat_acc(telemetry_driver_1, telemetry_driver_2,
                      fastest_driver_1, fastest_driver_2,
                      driver1, driver2, quali, save=False)
        graph_long_acc(telemetry_driver_1, telemetry_driver_2,
                       fastest_driver_1, fastest_driver_2,
                      driver1, driver2, quali, save=False)

    # 9) Analisis de rectas
    if rectasDRS:
        graph_analisis_rectasDRS(telemetry_driver_1, fastest_driver_1,
                                  telemetry_driver_2, fastest_driver_2)


def main_weather(year, grand_prix, session):
    # 1) Config de ff1 and in general
    ff1.plotting.setup_mpl()
    ff1.Cache.enable_cache("./cache")
    pd.options.mode.chained_assignment = None

    # Cargar las sesiones
    quali = ff1.get_session(year, grand_prix, session)
    quali.load()

    # extraer el df_weather
    df_weather = quali.weather_data

    # 2) Ploteo de los datos meteorologicos
    graph_weather(df_weather=df_weather, session=quali, save=True)


def main_testing(year, test_number, session_number):
    # 1) Config de ff1 and in general
    ff1.plotting.setup_mpl()
    ff1.Cache.enable_cache("./cache")
    pd.options.mode.chained_assignment = None

    # 2) Cargar sesion de test
    test = ff1.get_testing_session(year, test_number, session_number, backend="fastf1")
    test.load()

    print(test)

if __name__ == '__main__':
    # ------- INPUTS ------
    year = 2025
    test_number = 1
    session_number= 2

    main_testing(year, test_number, session_number)