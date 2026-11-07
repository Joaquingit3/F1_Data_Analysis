import os
import numpy as np
import pandas as pd
import fastf1 as ff1

from fastf1 import plotting
from fastf1 import utils

from src.grapher.graphs_race import (
    plot_positionchanges_race, 
    graph_ritmos_por_equipos, 
    compare_lift_straight, 
    make_plot_mundial, 
    make_plot_posiciones,
    plot_race_start, 
    plot_race_start_histogram
)
from src.grapher.graphs import (
    graph_weather,  
    graph_weather_in_rows, 
    graph_telemetry_custom, 
    graph_telemetry_custom_merged
)
from src.utils.functions import (
    analiza_stint,
    get_final_sesion_table, 
    update_mundial_pilotos, 
    update_mundial_pilotos_historic,
    update_posicion_pilotos_historic,
    update_mundial_equipos_historic,
    update_mundial_equipos,
    update_race_stats, 
    analisis_degradacion, 
    analiza_stint_compound, 
    calculate_deltas_twodrivers, 
    show_laptime
)
from src.creators.creator import (
    update_clasification, 
    create_race_sesion, 
    update_calendar, 
    create_race_document, 
    update_index
)
from src.grapher.graph_comparaison_race import plot_race_comparaison_two_drivers, graph_interval_two_drivers
from src.logger.logging_utils import config_logger, parse_logger_to_df

AUXILIAR = False

def race_comparaison(year, grand_prix, session, driver1, driver2, options: dict):
    # -------------------------- CONFIG --------------------------------------------------------------------------------
    # Config de ff1 and in general
    ff1.plotting.setup_mpl(color_scheme="fastf1")
    ff1.Cache.enable_cache("src/cache")
    pd.options.mode.chained_assignment = None

    log_capture = config_logger()

    # Load session
    race = ff1.get_session(year, grand_prix, session)
    race.load()

    # Obtener el contenido capturado
    log_capture.seek(0)
    captured_logs = log_capture.read()
    # df_logs = parse_logger_to_df(captured_logs)

    # Obtener la tabla resultado de la sesion (RACE)
    df_results = get_final_sesion_table(race)

    # ------------------------- CREATE RACE HTML  ----------------------------------------------------------------------
    if options["create_sesion"]:
        print("\nINFO     [0/2] Started Create Session process")
        create_race_sesion(race, df_results)
        print("INFO     [1/2] Finished Create Session process")

        print("\nINFO     [1/2] Started Race Document process")
        create_race_document(race)
        print("INFO     [2/2] Finished Create Session process")

    # ------------------------- UPDATE RACE STATS  ----------------------------------------------------------------------
    if options["opt_update_race_stats"]:
        print("\nINFO    [0/1] Started Update Race Stats process")
        update_race_stats(race, df_results)
        print("INFO    [1/1] Finished Update Race Stats process")

    # ------------------------- POSITION CHANGES PLOT ------------------------------------------------------------------
    if options["position_changes"]:
        print("\nINFO    [0/1] Started Position Changes graph")
        plot_positionchanges_race(race, df_logs, destacar=True, driver=driver1, save=True)
        print("INFO     [1/1] Finished Position Changes graph")

    # ------------------------- COMPARAISON TWO DRIVERS RACE -----------------------------------------------------------
    if options["race_start"]:
        print("\nINFO     Started Race Start graph")
        tiktok_format = False
        ## Plot Race Start two dirvers
        df_times = plot_race_start(race, driver1, driver2, tiktok_format, save=True)

        ## Plot tace start histogram
        plot_race_start_histogram(race, df_times, tiktok_format, save=True)

        print("INFO     Finished Race Start graph")

    # ------------------------- COMPARAISON TWO DRIVERS RACE -----------------------------------------------------------
    if options["comparaison"]:
        print("\nINFO     Started Comparaison Races process")
        corners = race.get_circuit_info().corners

        # Load laps
        laps_driver_1 = race.laps.pick_drivers(driver1)
        laps_driver_2 = race.laps.pick_drivers(driver2)

        # ------------------------- PLOT TIEMPO POR VUELTA -------------------------------------------------------------
        print("INFO     [0/2] Started Race Comparaison graph")
        copy_laps_d1 = laps_driver_1.copy()
        copy_laps_d2 = laps_driver_2.copy()
        plot_race_comparaison_two_drivers(
            laps_driver_1=copy_laps_d1,
            laps_driver_2=copy_laps_d2,
            driver_1=driver1,
            driver_2=driver2,
            event=race,
            save=True
        )
        print("INFO     [1/2] Finished Race Comparaison graph")

        # ------------------------- PLOT DELTA POR VUELTA --------------------------------------------------------------
        print("INFO     [2/2] Started Interval Race Comparaison graph")
        drivers = [driver1, driver2]
        graph_interval_two_drivers(copy_laps_d1, copy_laps_d2, drivers, race, save=True)
        print("INFO     [2/2] Finished Interval Race Comparaison graph")

        # -------------------------------- AUXILIAR --------------------------------------------------------------------
        if AUXILIAR:
            target_lap = 31.0
            compare_lift_straight(laps_driver_1, laps_driver_2, race, target_lap, drivers, save=True)

        print("INFO     Finished Comparaison Races process")
        
    # ------------------------- CHECK ONE LAP TELEMETRY ----------------------------------------------------------------
    if options["check_telemetry"]:
        # Params
        lap1 = 23.0
        lap2 = 24.0
        # lap1 = lap2 = 42.0
        columns = ['Speed', 'Delta', 'Throttle', 'Brake', 'RPM']
        # columns = ['Speed', 'Throttle', 'Brake', 'RPM']

        # distances = ["all"]
        # distances = [0, 1600] # Curva1
        # distances = [200, 500] # Curva1 Spa
        # distances = [3600, 4350] # Pouhon Spa
        # distances = [900, 1500] # Eau Rouge Spa
        distances = [3800, 4200] # Curva  Monza

        # Extract laps and corners
        laps = race.laps
        driver1_laps = laps.pick_drivers(driver1)
        driver2_laps = laps.pick_drivers(driver2)
        corners = race.get_circuit_info().corners

        # Extract telemetries of that laps
        fastest_lap1 = driver1_laps[driver1_laps['LapNumber']==lap1].iloc[0]
        fastest_lap2 = driver2_laps[driver2_laps['LapNumber']==lap2].iloc[0]
  
        # Graph telemetry
        # 3) Ver el delta de ambas vueltas
        #show_laptime(driver1, lap1_telemetry, fastest_lap1)
        #show_laptime(driver2, lap2_telemetry, fastest_lap2)

        #calculate_deltas_twodrivers(lap1_telemetry, lap2_telemetry,
        #                            fastest_lap1, fastest_lap2, driver1, driver2)

        # 4) Graph telemetry
        driver1 = driver1 + f"(lap {lap1})"
        driver2 = driver2 + f"(lap {lap2})"
        plot_title = f"{race.event.year} {race.event.EventName} Ascari"

        graph_telemetry_custom(
            fastest_driver_1=fastest_lap1, 
            fastest_driver_2=fastest_lap2,
            driver_1=driver1, 
            driver_2=driver2, 
            corners=corners,
            columns=columns,
            distances=distances, 
            event=race,
            plot_title=plot_title,
            save=True
        )

    # ------------------------- ZOOM TIME DISTANCE -----------------------------------------------------------------------------
    # if options["zoom_time_distance"]:
    #     lap1 = 35.0
    #     lap2 = 57.0

    #     # Calculate df of relative distance
    #     ## Load laps
    #     laps_driver_1 = race.laps.pick_drivers(driver1)
    #     laps_driver_2 = race.laps.pick_drivers(driver2)
        
        # 
    # ------------------------- PLOT DATOS METEOROLOGICOS --------------------------------------------------------------
    if options["weather"]:
        print("\nINFO     Started Weather Graphs process")
        # Extraer el df_weather
        df_weather = race.weather_data

        # Ploteo de los datos meteorologicos
        graph_weather(df_weather=df_weather, session=race, save=True)
        #graph_weather_in_rows(df_weather=df_weather, session=race, save=True)

        print("INFO     Finished Weather Graphs process")

    # ------------------------- ACTUALIZAR MUNDIAL ---------------------------------------------------------------------
    if options["update_mundial"]:
        print("\nINFO     Started Update Mundial process")

        # Load
        outputs_path = os.path.join(os.getcwd(), "src", "outputs")
        df_mundial_pilotos = pd.read_csv(os.path.join(outputs_path, 'mundial_pilotos.csv'), sep=",")
        df_mundial_equipos = pd.read_csv(os.path.join(outputs_path, 'mundial_equipos.csv'), sep=",")
        df_historic_mundial_pilotos = pd.read_csv(os.path.join(outputs_path, 'historic_mundial_pilotos.csv'), sep=",")
        df_historic_posiciones_pilotos = pd.read_csv(os.path.join(outputs_path, 'historic_posiciones_pilotos.csv'), sep=",")
        df_historic_mundial_constructores = pd.read_csv(os.path.join(outputs_path, 'historic_mundial_constructores.csv'), sep=",")

        # Mundial de pilotos
        print("INFO     [0/3] Started Update Mundial Pilotos")
        update_mundial_pilotos(df_mundial=df_mundial_pilotos)
        update_mundial_pilotos_historic(df_historic_mundial_pilotos)
        update_posicion_pilotos_historic(df_historic_posiciones_pilotos, df_results) 
        make_plot_mundial(df_historic_mundial_pilotos, race, "Pilotos", save=True)
        # make_plot_posiciones(df_historic_posiciones_pilotos, race, save=True)
        print("INFO     [1/3] Finished Update Mundial Pilotos")

        # Mundial de equipos
        print("INFO     [1/3] Started Update Mundial Constructores")
        update_mundial_equipos(session=race, df_mundial=df_mundial_equipos)
        update_mundial_equipos_historic(df_mundial_equipos, df_historic_mundial_constructores)
        make_plot_mundial(df_historic_mundial_constructores, race, "Equipos", save=True)
        print("INFO     [2/3] Finished Update Mundial Constructores")

        # Actualizar la clasificacion mundial
        print("INFO     [2/3] Started Update Clasificacion")
        update_clasification(
            df_mundial_pilotos=df_mundial_pilotos, 
            df_mundial_constructores=df_mundial_equipos,
            session=race
        )
        print("INFO     [3/3] Finished Update Clasificacion")

        print("INFO     Finished Update Mundial process")

    # ------------------------- ACTUALIZAR CALENDARIO ---------------------------------------------------------------------
    if options["actualiza_calendario"]:
        print("\nINFO    [0/1] Started Actualiza Calendario process")
        race_name = race.session_info['Meeting']['Name'].split(" Grand Prix")[0] # Obtener carrera
        update_calendar(race_name)
        print("INFO     [1/1] Finished Actualiza Calendario process")

    # ------------------------- UPDATE INDEX -------------------------------------------------------------------------------
    if options["update_index_file"]:
        print("\nINFO    [0/1] Started Update Index process")
        race_name = race.session_info['Meeting']['Name'].split(" Grand Prix")[0] # Obtener carrera
        update_index(race_name)
        print("INFO    [1/1] Finished Update Index process")

    # ------------------------- DEGRADACION -----------------------------------------------------------------------------
    if options["analizar_degradacion_ritmos"]:
        print("\nINFO     Started Analiza Degradacion y Ritmos process")
        # Condiciones
        compound = ''
        stint = 1.0

        if compound == "":
            data_global = analiza_stint(race, stint)
        else:
            data_global = analiza_stint_compound(race, compound, stint)

        # PLOTS
        graph_ritmos_por_equipos(race, data_global, stint)
        print("INFO     Finished Analiza Degradacion y Ritmos process")


if __name__ == '__main__':
    # Inputs
    year = 2025
    grand_prix = 'Qatar'
    session = 'R'
    # session = 'Sprint'

    driver1 = 'VER'
    driver2 = 'PIA'

    # Options
    position_changes = False
    comparaison = False
    weather = False 
    analizar_degradacion_ritmos = False
    check_telemetry = False
    # zoom_time_distance = False
    create_sesion = False   

    race_start = False

    opt_update_race_stats = False
    update_mundial = False
    actualiza_calendario = False
    update_index_file = False

    list_options = [
        "position_changes", 
        "comparaison", 
        "weather", 
        "update_mundial", 
        "create_sesion", 
        "opt_update_race_stats", 
        "analizar_degradacion_ritmos", 
        "actualiza_calendario",
        "check_telemetry",
        "race_start",
        "update_index_file"
        # "zoom_time_distance" 
    ]
    options = {}
    for option in list_options:
        options[option] = globals()[option]

    # Comparar dos pilotos
    race_comparaison(
        year,
        grand_prix,
        session,
        driver1,
        driver2,
        options
    )

