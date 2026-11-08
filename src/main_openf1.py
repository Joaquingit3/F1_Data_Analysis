import sys
import time

from rich.console import Console
from tabulate import tabulate
from src.creators.creator import create_qualy_sesion
from src.utils.functions import save_time_gap, show_laptime, update_qualy_stats
from src.grapher.graphs import make_plot_gap
from src.driver.openf1_driver import OpenF1Driver, OpenF1AuthenticationError
from src.grapher.graphs import (
    graph_telemetry,
    create_minisectors_graph,
    graph_long_acc,
    graph_lat_acc,
)
from src.utils.functions import (
    calculate_deltas_twodrivers,
    get_lap_number_of_session,
    add_lap_info
)
from src.grapher.graph_by_type import graph_analisis_rectasDRS

console = Console()


def custom_wait(t):
    with console.status(f"[bold cyan]Waiting {t}s to recall OpenF1 ...", spinner="dots"):
        time.sleep(t)
    console.print("[green]✓ Done![/green]")


def main(
    year,
    grand_prix,
    session,
    driver1,
    driver2,
    minisectors=False,
    acc=False,
    rectasDRS=False,
    create_session=False,
    update_stats=False,
    save_gap=False,
    plot_gap=False,
    use_Q1_data=False,
):
    # 1) Config de ff1 and in general
    openf1_client = OpenF1Driver()

    # Cargar las sesiones
    if session in ("Q", "Qualifying"):
        session_name = "Qualifying"
    elif session in ("SQ", "Qualifying"):
        session_name = "Sprint Qualifying"
    elif session in ("R", "Race"):
        session_name = "Race"
    elif session in ("P", "Practice"):
        session_name = "Practice"
    else:
        msg = f"Incorrect session_name: {session_name}"
        raise ValueError(msg)

    session_data = openf1_client.get_sesion_info(year, grand_prix, session_name)
    custom_wait(3)

    # Cargar las curvas del circuito
    corners = openf1_client.get_circuit_corners(year, grand_prix)
    custom_wait(3)

    # TODO: Obtener la tabla resultado de la sesion
    # df_results = get_final_sesion_table(session_data, openf1_client, year, grand_prix, save_gap, use_Q1_data)
    df_results = openf1_client.get_qualifying_results(year, grand_prix)
    print(tabulate(df_results, headers="keys", showindex=False, tablefmt="grid"))
    custom_wait(3)

    # ------------------------- PLOT QUALY GAP  ------------------------------------------------------------------------
    if plot_gap:
        make_plot_gap(
            session=session_data,
            results=df_results,
            option="teams",
            trend=False,
            use_Q1_data=use_Q1_data,
        )

    # ------------------------- CREATE RACE HTML  ----------------------------------------------------------------------
    if create_session:
        # Create qualy sesion
        create_qualy_sesion(session_data, df_results)

    # ------------------------- UPDATE STATS ----------------------------------------------------------------------------
    if update_stats and session_data.name != "Sprint Qualifying":
        # Update stats
        update_qualy_stats(df_results)

    # Save GAP
    if save_gap:
        save_time_gap(
            df_results,
            session_data.session_info.Meeting.Number,
            session_data.session_info.Meeting.Country.Code,
            use_Q1_data=use_Q1_data,
        )

    # 2) Cargar las vueltas rapidas
    laps_driver_1 = openf1_client.get_laps_from_driver(df_results, session_data.session_key, driver1)
    laps_driver_2 = openf1_client.get_laps_from_driver(df_results, session_data.session_key, driver2)
    custom_wait(3)
    lap_number_driver_1 = get_lap_number_of_session(session_data, laps_driver_1, "Q3")
    lap_number_driver_2 = get_lap_number_of_session(session_data, laps_driver_2, "Q3")
    
    fastest_driver_1 = laps_driver_1[
        laps_driver_1["LapNumber"] == lap_number_driver_1
    ].iloc[0]
    fastest_driver_2 = laps_driver_2[
        laps_driver_2["LapNumber"] == lap_number_driver_2
    ].iloc[0]

    telemetry_driver_1 = openf1_client.get_telemetry_from_driver(df_results, fastest_driver_1, session_data.session_key, driver1)
    telemetry_driver_2 = openf1_client.get_telemetry_from_driver(df_results, fastest_driver_2, session_data.session_key, driver2)
    
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

    if minisectors:
        # 5) Graph track dominance
        create_minisectors_graph(
            telemetry_driver_1=telemetry_driver_1,
            telemetry_driver_2=telemetry_driver_2,
            fastest_driver_1=fastest_driver_1,
            fastest_driver_2=fastest_driver_2,
            driver_1=driver1,
            driver_2=driver2,
            event=session_data,
            n_minisectors=25,
            plot_wind=False,
            save=True,
        )
    else:
        # 4) Graph telemetry
        graph_telemetry(
            fastest_driver_1=fastest_driver_1,
            fastest_driver_2=fastest_driver_2,
            telemetry_driver_1=telemetry_driver_1,
            telemetry_driver_2=telemetry_driver_2,
            driver_1=driver1,
            driver_2=driver2,
            corners=corners,
            event=session_data,
            save=True,
        )

    # # 8) Acc. Lateral y Longitudinal
    # if acc:
    #     graph_lat_acc(
    #         telemetry_driver_1,
    #         telemetry_driver_2,
    #         fastest_driver_1,
    #         fastest_driver_2,
    #         driver1,
    #         driver2,
    #         session_data,
    #         save=False,
    #     )
    #     graph_long_acc(
    #         telemetry_driver_1,
    #         telemetry_driver_2,
    #         fastest_driver_1,
    #         fastest_driver_2,
    #         driver1,
    #         driver2,
    #         session_data,
    #         save=False,
    #     )

    # # 9) Analisis de rectas
    # if rectasDRS:
    #     graph_analisis_rectasDRS(
    #         telemetry_driver_1,
    #         fastest_driver_1,
    #         telemetry_driver_2,
    #         fastest_driver_2
    #     )


if __name__ == "__main__":
    year = 2026
    grand_prix = "São Paulo"
    session = "Qualifying"
    driver1 = "NOR"
    driver2 = "ALO"

    try:
        main(
            year=year,
            grand_prix=grand_prix,
            session=session,
            driver1=driver1,
            driver2=driver2,
            plot_gap=False,
            save_gap=False,
            use_Q1_data=False
        )
    except OpenF1AuthenticationError as e:
        print(f"[OpenF1] {e}", file=sys.stderr)
        sys.exit(1)