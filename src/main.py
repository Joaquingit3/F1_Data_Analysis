import argparse
import os
import sys
import traceback

import fastf1 as ff1
import pandas as pd
from dotenv import load_dotenv
from fastf1.exceptions import DataNotLoadedError

from src.creators.creator import create_qualy_sesion
from src.grapher.graph_by_type import graph_analisis_rectasDRS
from src.grapher.graphs import (
    create_minisectors_graph,
    graph_comparaison_telemetry_twolaps,
    graph_given_corner,
    graph_given_distances,
    graph_lat_acc,
    graph_long_acc,
    graph_telemetry,
    graph_weather,
    make_countplot,
    make_countplot_for_corners,
    make_graph_aero_analisis,
    make_plot_gap,
    plot_comparativa_curvas,
    plot_stats_circuitos,
    plot_stats_circuitos_pca,
    plot_stats_circuitos_pca_cluster,
)
from src.logger.logging_utils import config_logger
from src.utils.functions import (
    add_lap_info,
    calculate_deltas_twodrivers,
    convert_image_tiktok_margin,
    convert_tiktok_image,
    convert_tiktok_image_by_rangos,
    convert_tiktok_image_by_rangos_with_tiktok,
    get_final_sesion_table,
    get_lap_number,
    get_lap_number_of_session,
    resize_and_pad_images,
    show_laptime,
    update_qualy_stats,
)

from . import GLOBAL_OPTIONS, MODE_OPTIONS, MODES, SESSION_MODES


# FUNCTIONS
def get_session(year, grand_prix, session, **kwargs):
    try:
        session_data = ff1.get_session(year, int(grand_prix), session, **kwargs)
    except ValueError:
        session_data = ff1.get_session(year, grand_prix, session, **kwargs)
    return session_data


def main2(
    year,
    grand_prix,
    session,
    driver1,
    driver2,
    minisectors,
    acc,
    rectasDRS,
    create_session,
    update_stats,
    save_gap,
    plot_gap,
    use_Q1_data,
):
    # 1) Config de ff1 and in general
    ff1.plotting.setup_mpl(color_scheme="fastf1")
    ff1.Cache.enable_cache("src/cache")
    pd.options.mode.chained_assignment = None

    # logging
    log_capture = config_logger()

    # Cargar las sesiones
    quali = get_session(year, grand_prix, session)
    quali.load(laps=True, telemetry=True, weather=False, messages=False)

    # Obtener el contenido capturado
    log_capture.seek(0)
    captured_logs = log_capture.read()
    # df_logs = parse_logger_to_df(captured_logs)

    # Cargar las curvas del circuito
    try:
        corners = quali.get_circuit_info().corners
    except AttributeError:
        corners = []
        print("WARNING\tcorners NOT Exists!")
    except DataNotLoadedError:
        corners = []
        print("WARNING\tcorners NOT Exists!")
    print(corners)
    # Obtener la tabla resultado de la sesion
    try:
        df_results = get_final_sesion_table(quali, save_gap, use_Q1_data)
    except IndexError as e:
        print(f"[Error] Index Error: {e}\n")
    except DataNotLoadedError as e:
        print(f"[Error] Index Error: {e}\n")

    # ------------------------- PLOT QUALY GAP  ------------------------------------------------------------------------
    if plot_gap:
        # make_plot_gap(quali, option='top_teams', trend=False)
        make_plot_gap(quali, option="teams", trend=False, use_Q1_data=use_Q1_data)
        # teams_by_cluster
    # ------------------------- CREATE RACE HTML  ----------------------------------------------------------------------
    if create_session:
        # Create qualy sesion
        create_qualy_sesion(quali, df_results)

    # ------------------------- UPDATE STATS ----------------------------------------------------------------------------
    if update_stats and quali.name != "Sprint Qualifying":
        # Update stats
        update_qualy_stats(df_results)

    # 2) Cargar las vueltas rapidas
    # Cargamos todas sus vueltas
    laps_driver_1 = quali.laps.pick_drivers(driver1)
    laps_driver_2 = quali.laps.pick_drivers(driver2)

    # lap_number_driver_1 = get_lap_number(quali, laps_driver_1)
    lap_number_driver_1 = get_lap_number_of_session(quali, laps_driver_1, "Q3")
    # lap_number_driver_2 = get_lap_number(quali, laps_driver_2)
    lap_number_driver_2 = get_lap_number_of_session(quali, laps_driver_2, "Q3")
    # print(laps_driver_1[['LapNumber', 'LapTime', 'Driver']])
    # print(laps_driver_2[['LapNumber', 'LapTime', 'Driver']])
    # lap_number_driver_1 = 7.0
    # lap_number_driver_2 = 9.0

    # Cargamos la vuelta rapida de cada uno
    # fastest_driver_1 = laps_driver_1.pick_fastest()
    fastest_driver_1 = laps_driver_1[
        laps_driver_1["LapNumber"] == lap_number_driver_1
    ].iloc[0]
    print(fastest_driver_1)
    # fastest_driver_2 = laps_driver_2.pick_fastest()
    # print(laps_driver_2[['Driver', 'LapNumber', 'LapTime', 'Deleted']])
    fastest_driver_2 = laps_driver_2[
        laps_driver_2["LapNumber"] == lap_number_driver_2
    ].iloc[0]

    # calculamos su telemetria
    telemetry_driver_1 = fastest_driver_1.get_telemetry().add_distance()
    telemetry_driver_2 = fastest_driver_2.get_telemetry().add_distance()

    # 3) Ver el delta de ambas vueltas
    print(telemetry_driver_1)
    show_laptime(driver1, telemetry_driver_1, fastest_driver_1)
    show_laptime(driver2, telemetry_driver_2, fastest_driver_2)
    calculate_deltas_twodrivers(
        telemetry_driver_1,
        telemetry_driver_2,
        fastest_driver_1,
        fastest_driver_2,
        driver1,
        driver2,
    )

    # More info of the laps
    lap_driver1_info = add_lap_info(fastest_driver_1, quali)
    lap_driver2_info = add_lap_info(fastest_driver_2, quali)
    # print(tabulate(lap_driver1_info, headers='keys', showindex=False, tablefmt='grid'))
    # print(tabulate(lap_driver2_info, headers='keys', showindex=False, tablefmt='grid'))

    # Comparativa hasta la Frenada
    # graph_comparaison_firststraight(fastest_driver_1, fastest_driver_2, corners, save=True)
    # graph_comparaison_finalpart(fastest_driver_1, fastest_driver_2, corners, quali, save=True)
    # plot_time_by_distance(fastest_driver_1, fastest_driver_2, distance_ini=1800, distance_fin=2200,
    #                      driver1=driver1, driver2=driver2, quali, save=True)

    # rectas_imola(fastest_driver_1, fastest_driver_2, corners, driver1, driver2, quali, save=True)
    # obten_subset_curvas(fastest_driver_1, fastest_driver_2, corners, driver1, driver2, quali, save=True)

    if minisectors:
        # 5) Graph track dominance
        create_minisectors_graph(
            telemetry_driver_1=telemetry_driver_1,
            telemetry_driver_2=telemetry_driver_2,
            fastest_driver_1=fastest_driver_1,
            fastest_driver_2=fastest_driver_2,
            driver_1=driver1,
            driver_2=driver2,
            event=quali,
            n_minisectors=25,
            plot_wind=False,
            save=True,
        )
    else:
        # 4) Graph telemetry
        graph_telemetry(
            fastest_driver_1=fastest_driver_1,
            fastest_driver_2=fastest_driver_2,
            driver_1=driver1,
            driver_2=driver2,
            corners=corners,
            event=quali,
            save=True,
        )

    # 6) Clasificacion de las curvas del circuito por tipo
    # corners = create_corner_tableclasif(corners, telemetry_driver_1)
    # print(corners[['Angle', 'Type_Corner', 'Type_Corner_Speed']])

    # 7) Graficar telemetria con su tipo
    # Parametros del grafico
    # team_driver = fastest_driver_1['Team']
    # color = ff1.plotting.team_color(team_driver)

    # Lanzamos la funcion
    # graf_corners_onedriver(corners, telemetry_driver_1, color)

    # 8) Acc. Lateral y Longitudinal
    if acc:
        graph_lat_acc(
            telemetry_driver_1,
            telemetry_driver_2,
            fastest_driver_1,
            fastest_driver_2,
            driver1,
            driver2,
            quali,
            save=False,
        )
        graph_long_acc(
            telemetry_driver_1,
            telemetry_driver_2,
            fastest_driver_1,
            fastest_driver_2,
            driver1,
            driver2,
            quali,
            save=False,
        )

    # 9) Analisis de rectas
    if rectasDRS:
        graph_analisis_rectasDRS(
            telemetry_driver_1, fastest_driver_1, telemetry_driver_2, fastest_driver_2
        )


def main_carrera():
    # --------------------- SET-UP -------------------------------
    # Config de ff1 and in general
    ff1.plotting.setup_mpl()
    ff1.Cache.enable_cache("src/cache")
    pd.options.mode.chained_assignment = None

    # Session params
    year = 2024
    grand_prix = "Australia"
    session = "R"

    # Load session
    race = get_session(year, grand_prix, session)
    race.load()

    # -------------------- LOAD DATA ------------------------------
    corners = race.get_circuit_info().corners

    # Drivers params
    driver1 = "ALO"
    driver2 = "RUS"

    # Load laps
    laps_driver_1 = race.laps.pick_drivers(driver1)
    laps_driver_2 = race.laps.pick_drivers(driver2)

    # --------------------- SELECT DATA ----------------------------
    lap_starttime_driver1 = laps_driver_1["LapStartTime"]
    lap_starttime_driver2 = laps_driver_2["LapStartTime"]

    # Lap params
    lap_russ_accident = 57

    # Revisar este calculo
    index_ini_ALO = lap_starttime_driver1.index[0]
    index_ini_RUS = lap_starttime_driver2.index[0]
    time_prev_lap_ALO = lap_starttime_driver1[index_ini_ALO + lap_russ_accident - 2]
    time_prev_lap_RUS = lap_starttime_driver2[index_ini_RUS + lap_russ_accident - 2]
    time_inicio_lap_ALO = lap_starttime_driver1[index_ini_ALO + lap_russ_accident - 1]
    time_inicio_lap_RUS = lap_starttime_driver2[index_ini_RUS + lap_russ_accident - 1]
    time_fin_lap_ALO = lap_starttime_driver1[index_ini_ALO + lap_russ_accident]
    time_fin_lap_RUS = laps_driver_2.telemetry.loc[
        laps_driver_2.telemetry.index[-1], "SessionTime"
    ]

    # Telemetry data
    mask_lapaccident_1 = laps_driver_1.telemetry["SessionTime"] >= time_inicio_lap_ALO
    mask_lapaccident_2 = laps_driver_1.telemetry["SessionTime"] <= time_fin_lap_ALO
    telemetry_lapaccident_ALO = laps_driver_1.telemetry.loc[
        mask_lapaccident_1 & mask_lapaccident_2
    ]

    mask_lapaccident_1 = laps_driver_1.telemetry["SessionTime"] >= time_prev_lap_ALO
    mask_lapaccident_2 = laps_driver_1.telemetry["SessionTime"] <= time_inicio_lap_ALO
    telemetry_prevaccident_ALO = laps_driver_1.telemetry[
        mask_lapaccident_1 & mask_lapaccident_2
    ]

    mask_lapaccident_1 = laps_driver_1.telemetry["SessionTime"] >= time_inicio_lap_RUS
    mask_lapaccident_2 = laps_driver_1.telemetry["SessionTime"] <= time_fin_lap_RUS
    telemetry_lapaccident_RUS = laps_driver_2.telemetry[
        mask_lapaccident_1 & mask_lapaccident_2
    ]

    mask_lapaccident_1 = laps_driver_1.telemetry["SessionTime"] >= time_prev_lap_RUS
    mask_lapaccident_2 = laps_driver_1.telemetry["SessionTime"] <= time_inicio_lap_RUS
    telemetry_prevaccident_RUS = laps_driver_2.telemetry[
        mask_lapaccident_1 & mask_lapaccident_2
    ]

    # -----------------------------PLOT COMPARAISON ----------------------
    # Comparacion vueltas ALO - Prev Accident vs Accident Lap
    index_ini = telemetry_prevaccident_ALO.index[0]
    telemetry_prevaccident_ALO["DistanceLap"] = (
        telemetry_prevaccident_ALO["Distance"]
        - telemetry_prevaccident_ALO.loc[index_ini, "Distance"]
    )

    index_ini = telemetry_lapaccident_ALO.index[0]
    telemetry_lapaccident_ALO["DistanceLap"] = (
        telemetry_lapaccident_ALO["Distance"]
        - telemetry_lapaccident_ALO.loc[index_ini, "Distance"]
    )

    graph_comparaison_telemetry_twolaps(
        fastest_driver_1=laps_driver_1,
        fastest_driver_2=laps_driver_1,
        telemetry_driver_1=telemetry_prevaccident_ALO,
        telemetry_driver_2=telemetry_lapaccident_ALO,
        driver_1=driver1 + "Prev. Lap Accident",
        driver_2=driver1 + "Lap Accident",
        corners=corners,
        event=race,
        save=True,
    )

    # Comparacion vueltas RUS - Prev Accident vs Accident Lap
    index_ini = telemetry_prevaccident_RUS.index[0]
    telemetry_prevaccident_RUS["DistanceLap"] = (
        telemetry_prevaccident_RUS["Distance"]
        - telemetry_prevaccident_RUS.loc[index_ini, "Distance"]
    )

    index_ini = telemetry_lapaccident_RUS.index[0]
    telemetry_lapaccident_RUS["DistanceLap"] = (
        telemetry_lapaccident_RUS["Distance"]
        - telemetry_lapaccident_RUS.loc[index_ini, "Distance"]
    )

    graph_comparaison_telemetry_twolaps(
        fastest_driver_1=laps_driver_2,
        fastest_driver_2=laps_driver_2,
        telemetry_driver_1=telemetry_prevaccident_RUS,
        telemetry_driver_2=telemetry_lapaccident_RUS,
        driver_1=driver2 + "Prev. Lap Accident",
        driver_2=driver2 + "Lap Accident",
        corners=corners,
        event=race,
        save=True,
    )


def main_comparaison(
    year,
    grand_prix,
    session,
    list_drivers,
    distance,
    bycorners,
    list_comparaison,
    session_bars,
    name,
    aero_analisis,
    save_stats_circuito=False,
):
    # 1) Config de ff1 and in general
    ff1.plotting.setup_mpl(color_scheme="fastf1")
    ff1.Cache.enable_cache("src/cache")
    pd.options.mode.chained_assignment = None

    # Cargar las sesiones
    quali = get_session(year, grand_prix, session)

    quali.load(
        laps=True,
        telemetry=True,
        weather=False,
        messages=False,
    )
    # test_mapgraph(quali)

    # Analizar la sesion
    if session_bars:
        concepts = [
            "Sector1Time",
            "Sector2Time",
            "Sector3Time",
            "Sector1SessionTime",
            "Sector2SessionTime",
            "Sector3SessionTime",
            "SpeedI1",
            "SpeedI2",
            "SpeedFL",
            "SpeedST",
        ]

        # concepto = 'SpeedST'
        conceptos = ["Sectors", "SpeedST", "LapTime"]
        # concepto = 'LapTime'

        types = ["Team", "Team", "Driver"]
        # type='Driver'

        tiktok_format = False
        for concepto, type in zip(conceptos, types):
            make_countplot(
                concepto=concepto,
                session=quali,
                type=type,
                tiktok_format=tiktok_format,
                save=True,
            )

    # 2) Analisis curva dada
    if distance:
        graph_given_distances(
            event=quali,
            name=name,
            drivers=list_drivers,
            list_distances=list_comparaison,
            save=True,
        )
    elif bycorners:
        # Cargar las curvas del circuito
        try:
            corners = quali.get_circuit_info().corners
        except AttributeError:
            corners = []

        graph_given_corner(
            event=quali,
            corners=corners,
            drivers=list_drivers,
            list_corners=list_comparaison,
            save=True,
        )

    # 3) Aero Analisis
    if aero_analisis:
        make_graph_aero_analisis(
            quali,
            data_origin="laps",
            save=True,
            save_stats_circuito=save_stats_circuito,
        )


def main_comparaison_corners(year, grand_prix, session, list_distances, session_bars):
    # 1) Config de ff1 and in general
    ff1.plotting.setup_mpl()
    ff1.Cache.enable_cache("src/cache")
    pd.options.mode.chained_assignment = None

    # Cargar las sesiones
    quali = get_session(year, grand_prix, session)
    quali.load()

    # Analizar la sesion
    if session_bars:
        type = "Team"
        make_countplot_for_corners(list_distances, quali, type, save=True)


def main_comparaison_twosesion():
    # 1) Config de ff1 and in general
    ff1.plotting.setup_mpl()
    ff1.Cache.enable_cache("src/cache")
    pd.options.mode.chained_assignment = None

    # 2) QUALY SPRINT
    # 2.1) Cargar la qualy Sprint
    year = 2025
    grand_prix = "Australia"
    session = "Sprint Qualifying"
    driver = "RUS"

    squali = get_session(year, grand_prix, session)
    squali.load()

    # 2.2) Cargar la telemetria de Max
    sqlaps = squali.laps.pick_drivers(driver)
    lap_number_sqlap = get_lap_number(squali, sqlaps)
    fastest_sqlap = sqlaps[sqlaps["LapNumber"] == lap_number_sqlap].iloc[0]

    telemetry_sq = fastest_sqlap.get_telemetry().add_distance()

    # 3) QUALY
    # 3.1) Cargar la qualy Sprint
    session = "Q"

    quali = ff1.get_session(year, grand_prix, session)
    quali.load()

    corners = quali.get_circuit_info().corners

    # 3.2) Cargar la telemetria
    qlaps = quali.laps.pick_drivers(driver)
    lap_number_qlap = get_lap_number(quali, qlaps)
    fastest_qlap = qlaps[qlaps["LapNumber"] == lap_number_qlap].iloc[0]

    telemetry_q = fastest_qlap.get_telemetry().add_distance()

    # 4) Ver el delta de ambas vueltas
    show_laptime(f"{driver} SQ", telemetry_sq, fastest_sqlap)
    show_laptime(f"{driver} Q", telemetry_q, fastest_qlap)

    calculate_deltas_twodrivers(
        telemetry_sq,
        telemetry_q,
        fastest_sqlap,
        fastest_qlap,
        f"{driver}  SQ",
        f"{driver}  Q",
    )

    # 4) Graph telemetry
    graph_telemetry(
        fastest_driver_1=fastest_qlap,
        fastest_driver_2=fastest_sqlap,
        driver_1=f"{driver} Q",
        driver_2=f"{driver} SQ",
        corners=corners,
        event=quali,
        save=True,
    )


def main_weather(year, grand_prix, session):
    # 1) Config de ff1 and in general
    ff1.plotting.setup_mpl(color_scheme="fastf1")
    ff1.Cache.enable_cache("src/cache")
    pd.options.mode.chained_assignment = None

    # Cargar las sesiones
    quali = get_session(
        year,
        grand_prix,
        session,
        laps=False,
        telemetry=False,
        weather=True,
        messages=False,
    )
    quali.load()

    # extraer el df_weather
    df_weather = quali.weather_data

    # 2) Ploteo de los datos meteorologicos
    graph_weather(df_weather=df_weather, session=quali, save=True)


def main_image(img_path, target_size):
    # Reducir la imagen
    resize_and_pad_images(
        img_path=img_path, output_path=img_path, target_size=target_size
    )


def main_tiktokimage(img_path, image, json):
    # Extraer info
    tipo = json["tipo"]

    # Convertir a TikTok
    if tipo == "divisiones":
        divisiones = json["cantidad"]
        convert_tiktok_image(img_path=img_path, image=image, divisiones=divisiones)
    elif tipo == "rangos":
        rangos = json["cantidad"]
        convert_tiktok_image_by_rangos(img_path=img_path, image=image, rangos=rangos)
    elif tipo == "convert":
        convert_image_tiktok_margin(
            img_path=img_path, img_name=image, transparent=False
        )
    elif tipo in ("convert+rangos", "rangos+convert"):
        rangos = json["cantidad"]
        convert_tiktok_image_by_rangos_with_tiktok(
            img_path=img_path, image=image, rangos=rangos
        )
    else:
        print("ERROR    Tipo invalido")


def main_compare():
    ff1.plotting.setup_mpl(color_scheme="fastf1")
    # Generar dataframe circuito
    d = {"Lenta": [7, 6], "Media": [3, 3], "Rapida": [2, 4], "A Fondo": [2, 1]}
    df_circuito = pd.DataFrame(data=d, index=["Hungaroring", "Zandvoort"])

    # Plotear
    plot_comparativa_curvas(df_circuito, save=True)


def main_plot_circuits(plot_circuits_pca, cluster):
    ff1.plotting.setup_mpl(color_scheme="fastf1")

    # Load df_stats_circuits
    file_path = os.path.join(os.getcwd(), "src", "outputs", "stats_circuitos.csv")
    df_stats_circuits = pd.read_csv(file_path, sep=",")

    # Plot values
    if plot_circuits_pca:
        if cluster:
            plot_stats_circuitos_pca_cluster(df_stats_circuits, save=True)
        else:
            plot_stats_circuitos_pca(df_stats_circuits, save=True)
    else:
        plot_stats_circuitos(df_stats_circuits, save=True)


# ====================== ARGSPARSE =====================================================================================
def get_provided_options(argv):
    """
    Devuelve únicamente las opciones escritas explícitamente
    por el usuario.

    Ejemplo:
        ["--year", "2026", "--distance"]
    devuelve:
        {"--year", "--distance"}
    """
    provided_options = set()

    for argument in argv:
        if argument.startswith("--"):
            option = argument.split("=", maxsplit=1)[0]
            provided_options.add(option)

    return provided_options


def validate_args(parser, args, argv):
    selected_modes = set(args.modes)
    provided_options = get_provided_options(argv)

    # Opciones válidas como unión de las permitidas para todos
    # los modos seleccionados.
    allowed_options = set(GLOBAL_OPTIONS)

    for mode in selected_modes:
        allowed_options.update(MODE_OPTIONS[mode])

    invalid_options = provided_options - allowed_options

    if invalid_options:
        invalid_text = ", ".join(sorted(invalid_options))
        modes_text = ", ".join(args.modes)

        parser.error(
            f"Los argumentos {invalid_text} no son válidos "
            f"para los modos seleccionados: {modes_text}"
        )

    # Comprobar si alguno de los modos necesita una sesión.
    modes_requiring_session = selected_modes & SESSION_MODES

    if modes_requiring_session:
        missing_arguments = []

        if args.year is None:
            missing_arguments.append("--year")

        if args.grand_prix is None:
            missing_arguments.append("--grand-prix")

        if missing_arguments:
            parser.error(
                "Los siguientes argumentos son obligatorios para "
                f"{', '.join(sorted(modes_requiring_session))}: "
                f"{', '.join(missing_arguments)}"
            )

    # Evitar repetir un modo accidentalmente.
    if len(args.modes) != len(selected_modes):
        parser.error("No se puede indicar el mismo modo más de una vez")

    # Validaciones específicas de comparaison.
    if "comparaison" in selected_modes:
        if args.distance and len(args.list_distances) < 2:
            parser.error("--distance requiere al menos dos valores en --list-distances")

        if args.by_corners and not args.list_corners:
            parser.error("--by-corners requiere valores en --list-corners")

    # Cluster solo tiene sentido junto al PCA.
    if (
        "plot-circuits" in selected_modes
        and args.cluster
        and not args.plot_circuits_pca
    ):
        parser.error("--cluster requiere también --plot-circuits-pca")


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Generación de gráficas y análisis de sesiones de Fórmula 1")
    )

    # ------------------------------------------------------------------
    # Argumentos principales
    # ------------------------------------------------------------------
    principal_group = parser.add_argument_group("Argumentos principales")

    principal_group.add_argument(
        "--modes",
        choices=MODES,
        nargs="+",
        default=["comparaison"],
        help="Uno o varios tipos de análisis que se ejecutarán",
    )

    # No los hacemos required aquí porque hay modos como
    # compare-circuits que no los necesitan.
    principal_group.add_argument("--year", type=int)
    principal_group.add_argument("--grand-prix")
    principal_group.add_argument("--session", default="Q")

    # ------------------------------------------------------------------
    # Simple
    # ------------------------------------------------------------------
    simple_group = parser.add_argument_group("Argumentos del modo simple")

    simple_group.add_argument("--driver1", default="NOR")
    simple_group.add_argument("--driver2", default="HAM")
    simple_group.add_argument("--update-stats", action="store_true")
    simple_group.add_argument("--save-gap", action="store_true")
    simple_group.add_argument("--plot-gap", action="store_true")
    simple_group.add_argument("--use_Q1_data", action="store_true")
    simple_group.add_argument("--minisectors", action="store_true")
    simple_group.add_argument("--acc", action="store_true")
    simple_group.add_argument("--rectas-drs", action="store_true")
    simple_group.add_argument("--create-session", action="store_true")

    # ------------------------------------------------------------------
    # Comparaison
    # ------------------------------------------------------------------
    comparaison_group = parser.add_argument_group("Argumentos del modo comparaison")

    comparaison_group.add_argument(
        "--list-drivers",
        nargs="+",
        default=["PIA", "VER", "RUS", "HAM", "ALO"],
        help="Pilotos incluidos en las comparaciones",
    )

    comparison_type = comparaison_group.add_mutually_exclusive_group()

    comparison_type.add_argument(
        "--distance",
        action="store_true",
        help="Realiza la comparación usando distancias",
    )

    comparison_type.add_argument(
        "--by-corners",
        action="store_true",
        help="Realiza la comparación usando números de curva",
    )

    comparaison_group.add_argument(
        "--list-distances",
        nargs="+",
        type=float,
        default=[4200, 4500],
        metavar="DISTANCE",
        help="Distancias utilizadas en la comparación",
    )

    comparaison_group.add_argument(
        "--list-corners",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        metavar="CORNER",
        help="Números de curva utilizados en la comparación",
    )

    comparaison_group.add_argument(
        "--session-bars",
        action="store_true",
    )

    comparaison_group.add_argument(
        "--name",
        default="C14",
    )

    comparaison_group.add_argument(
        "--aero-analysis",
        action="store_true",
    )

    # ------------------------------------------------------------------
    # Plot circuits
    # ------------------------------------------------------------------
    circuits_group = parser.add_argument_group("Argumentos del modo plot-circuits")

    circuits_group.add_argument(
        "--plot-circuits-pca",
        action="store_true",
    )

    circuits_group.add_argument(
        "--cluster",
        action="store_true",
    )

    args = parser.parse_args()

    validate_args(parser, args, sys.argv[1:])

    return args


def run_mode(mode, args):
    print()
    print("=" * 70)
    print(f"Ejecutando modo: {mode}")
    print("=" * 70)

    if mode == "simple":
        main2(
            year=args.year,
            grand_prix=args.grand_prix,
            session=args.session,
            driver1=args.driver1,
            driver2=args.driver2,
            minisectors=args.minisectors,
            acc=args.acc,
            rectasDRS=args.rectas_drs,
            create_session=args.create_session,
            update_stats=args.update_stats,
            save_gap=args.save_gap,
            plot_gap=args.plot_gap,
            use_Q1_data=args.use_Q1_data,
        )

    elif mode == "weather":
        main_weather(
            year=args.year,
            grand_prix=args.grand_prix,
            session=args.session,
        )

    elif mode == "comparaison":
        if args.distance:
            list_comparaison = args.list_distances
        elif args.by_corners:
            list_comparaison = args.list_corners
        else:
            list_comparaison = []

        main_comparaison(
            year=args.year,
            grand_prix=args.grand_prix,
            session=args.session,
            list_drivers=args.list_drivers,
            distance=args.distance,
            bycorners=args.by_corners,
            list_comparaison=list_comparaison,
            session_bars=args.session_bars,
            name=args.name,
            aero_analisis=args.aero_analysis,
        )

    elif mode == "comparaison-corners":
        main_comparaison_corners(
            year=args.year,
            grand_prix=args.grand_prix,
            session=args.session,
            list_distances=args.list_distances,
            session_bars=args.session_bars,
        )

    elif mode == "comparaison-sessions":
        main_comparaison_twosesion()

    elif mode == "compare-circuits":
        main_compare()

    elif mode == "plot-circuits":
        main_plot_circuits(
            plot_circuits_pca=args.plot_circuits_pca,
            cluster=args.cluster,
        )


def run(args):
    failed_modes = []

    for mode in args.modes:
        try:
            run_mode(mode, args)
        except Exception as exc:
            failed_modes.append(mode)
            print(
                f"Error ejecutando el modo {mode}: {exc}, traceback: {traceback.format_exc()}",
                file=sys.stderr,
            )

    if failed_modes:
        raise RuntimeError("Fallaron los modos: " + ", ".join(failed_modes))


if __name__ == "__main__":
    # Automatical part
    args = parse_args()
    run(args)

    # Aditional part
    carpeta = "Converter"
    image_path = os.path.join("src", "outputs", carpeta)
    target_size = (1772, 997)
    red_img = False  # Main

    divisions = 2
    rangos = [(100, 1720), (1720, 2660), (2660, 3600)]
    json = {
        "tipo": "rangos",
        "cantidad": rangos,
    }
    image = "Delta_NOR_PIA.png"
    convert_tiktok = False  # Main

    # Reducir imagenes
    if red_img:
        main_image(image_path, target_size)

    # Extrapolar al formateo TikTok
    if convert_tiktok:
        main_tiktokimage(image_path, image, json)
