import json
import os
import time
import httpx
import asyncio
import requests
import numpy as np
import pandas as pd
import fastf1 as ff1
import seaborn as sns

from datetime import datetime, timedelta
from matplotlib import pyplot as plt

from concurrent.futures import ThreadPoolExecutor

from tabulate import tabulate

from src import DICT_TEAM_COLOR

PARALELL = False
DROPNA = True


def process_to_dataframe_victories(races, format="df"):
    if format == "df":
        years = []
        rounds = []
        for race in races:
            years.append(int(race["season"]))
            rounds.append(int(race["round"]))
        return pd.DataFrame({"year": years, "round": rounds})
    elif format == "list":
        wins_session_list = [
            (int(race["season"]), int(race["round"]), "Q") for race in races
        ]
        return wins_session_list


def get_driver_wins_from_path(path):
    with open(path, "r") as f:
        data = json.load(f)
    races = data["MRData"]["RaceTable"]["Races"]
    return process_to_dataframe_victories(races, format="list")


async def get_driver_wins(driver):
    base_url = f"https://api.jolpi.ca/ergast/f1/drivers/{driver}/results/1/"

    limit = 100
    offset = 0

    all_races = []

    async with httpx.AsyncClient() as client:
        while True:
            response = await client.get(
                base_url,
                params={
                    "limit": limit,
                    "offset": offset,
                },
                headers={"User-Agent": "MyF1Project/1.0"},
            )

            if response.status_code != httpx.codes.OK:
                print(
                    f"[ERROR] Obteniendo Driver wins "
                    f"Status code: {response.status_code} "
                    f"msg: {response.text}"
                )
                return None

            data = response.json()

            races = data["MRData"]["RaceTable"]["Races"]
            all_races.extend(races)

            # Total number of results
            total = int(data["MRData"]["total"])

            # Have we retrieved everything?
            if len(all_races) >= total:
                break

            offset += limit

    # Save all results
    data["MRData"]["RaceTable"]["Races"] = all_races
    path = os.path.join(
        os.getcwd(), "src", "outputs", "data_wins", driver, f"wins_{driver}.json"
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    return process_to_dataframe_victories(all_races, format="list")


async def get_drivers_wins(drivers):
    dict_driver_wins = {}
    for driver in drivers:
        path = os.path.join(
            os.getcwd(), "src", "outputs", "data_wins", driver, f"wins_{driver}.json"
        )

        if not os.path.exists(path):
            list_session_wins = await get_driver_wins(driver)
        else:
            list_session_wins = get_driver_wins_from_path(path)

        dict_driver_wins[driver] = list_session_wins
    return dict_driver_wins


def load_session(year, gp, identifier):
    session = ff1.get_session(year, gp, identifier)
    session.load(
        laps=False,
        telemetry=False,
        weather=False,
        messages=False,
    )
    return session


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


def calculate_gap_percentage(fastest, time_lap):
    if time_lap == "NoTime" or pd.isna(time_lap):
        return "NA"

    # Convert string to Timedelta if necessary
    if isinstance(time_lap, str):
        time_lap = pd.to_timedelta("00:" + time_lap)

    if isinstance(fastest, str):
        fastest = pd.to_timedelta("00:" + fastest)

    # Convert to seconds
    fastest_seconds = fastest.total_seconds()
    time_seconds = time_lap.total_seconds()

    # Calculate percentage gap
    delta = ((time_seconds - fastest_seconds) / fastest_seconds) * 100

    return delta


def get_df_gaps(sessions, driver):
    gaps = []
    teams = []
    years = []
    races = []
    rounds = []
    for session in sessions:
        results = session.results
        if not results.empty:
            gap, team = get_qualy_gap(session.results, driver)
        else:
            try:
                print(
                    f"[DEBUG] {session.event.year} {session.event.EventName} Not Available"
                )
                gap = pd.NA
                team = pd.NA
                print("\n")
            except ValueError:
                continue

        gaps.append(gap)
        teams.append(team)
        years.append(session.event.year)
        races.append(session.event.EventName)
        rounds.append(session.event.RoundNumber)

    return pd.DataFrame(
        {"Year": years, "Round": rounds, "Race": races, "Gap": gaps, "Team": teams}
    )


def get_df_gaps_from_path(path):
    return pd.read_csv(path, sep=",")


def get_qualy_gap(results, driver):
    # Get Laptime column
    df_results = generate_laptime_column(results)

    # Get Pole Laptime
    pole_laptime = df_results["LapTime"].iloc[0]

    # Get 'driver' Laptime
    driver_laptime = df_results.loc[df_results["DriverId"] == driver, "LapTime"].iloc[0]
    driver_team = df_results.loc[df_results["DriverId"] == driver, "TeamName"].iloc[0]

    # Get gap
    pole_gap = calculate_gap_percentage(pole_laptime, driver_laptime)

    if pole_gap == 0.0:
        # Pole Siter -> Get gap from second
        second_laptime = df_results["LapTime"].iloc[1]
        second_team = df_results["TeamName"].iloc[1]

        if driver_team == second_team:
            rival_laptime = df_results["LapTime"].iloc[2]
            return calculate_gap_percentage(rival_laptime, driver_laptime), driver_team
        else:
            return calculate_gap_percentage(second_laptime, driver_laptime), driver_team

    return pole_gap, driver_team

def format_driver_name(driver):
    driver_title = ""
    driver_splited = driver.split("_")

    for split in driver_splited:
        driver_title += f"{split[0].upper()}{split[1:]} "
    return driver_title


def plot_gap_by_team(df, driver):
    teams = df["Team"].unique()
    n_teams = len(teams)

    fig, axes = plt.subplots(
        nrows=n_teams + 1,
        ncols=1,
        figsize=(11, 4 + n_teams * 2.5),
        sharex=True,
        dpi=200,
    )

    # Mismos bins para TODOS los histogramas
    max_gap = np.ceil(df["Gap"].abs().max() * 4) / 4

    bins = np.arange(-max_gap, max_gap + 0.25, 0.25)

    # ---------------------------
    # GENERAL
    # ---------------------------
    ax = axes[0]

    ax.hist(df["Gap"], bins=bins, edgecolor="white", linewidth=1.2, alpha=0.9)

    ax.axvline(0, color="black", linewidth=2)

    median_gap = df["Gap"].median()
    median_gap_advantage = df.loc[df["Gap"] <= 0.0, "Gap"].median()
    median_gap_disadvantage = df.loc[df["Gap"] > 0.0, "Gap"].median()

    # Median
    ax.axvline(
        median_gap,
        linestyle="--",
        color="red",
        linewidth=1.8,
        label=f"Mediana: {median_gap:+.2f}%",
    )

    ax.axvline(
        median_gap_advantage,
        linestyle=":",
        color="lime",
        linewidth=1.5,
        label=f"Mediana con Ventaja: {median_gap_advantage:+.2f}%",
    )

    ax.axvline(
        median_gap_disadvantage,
        linestyle=":",
        color="red",
        linewidth=1.5,
        label=f"Mediana con Desventaja: {median_gap_disadvantage:+.2f}%",
    )

    driver_title = f"{driver[0].upper()}{driver[1:]}"
    ax.set_title(
        f"{driver_title} · {df['Gap'].shape[0]} victorias · Mediana {median_gap:+.2f}% · con Ventaja {median_gap_advantage:-.2f}% · con Desventaja {median_gap_disadvantage:+.2f}%",
        fontsize=18,
        fontweight="bold",
        loc="left",
    )
    ax.set_ylabel("Victorias")
    ax.legend(frameon=False)

    # ---------------------------
    # POR EQUIPO
    # ---------------------------
    for ax, team in zip(axes[1:], teams):
        subset_team = df.loc[df["Team"] == team, ].copy()

        ax.hist(
            subset_team["Gap"].to_list(),
            bins=bins,
            edgecolor="white",
            facecolor=DICT_TEAM_COLOR[team],
            linewidth=1.2,
            alpha=0.9,
        )

        ax.axvline(0, color="black", linewidth=1.5)

        median_gap = subset_team["Gap"].median()
        median_gap_advantage = subset_team.loc[subset_team["Gap"] <= 0.0,"Gap"].median()
        median_gap_disadvantage = subset_team.loc[subset_team["Gap"] > 0.0,"Gap"].median()

        ax.axvline(
            median_gap,
            linestyle="--",
            color="black",
            linewidth=1.5,
            label=f"Mediana: {median_gap:+.2f}%",
        )
        ax.axvline(
            median_gap_advantage,
            linestyle=":",
            color="lime",
            linewidth=1.5,
            label=f"Mediana con Ventaja: {median_gap_advantage:+.2f}%",
        )

        ax.axvline(
            median_gap_disadvantage,
            linestyle=":",
            color="red",
            linewidth=1.5,
            label=f"Mediana con Desventaja: {median_gap_disadvantage:+.2f}%",
        )

        ax.set_title(
            f"{team} · {subset_team.shape[0]} victorias · Mediana {median_gap:+.2f}% · con Ventaja {median_gap_advantage:-.2f}% · con Desventaja {median_gap_disadvantage:+.2f}%",
            fontsize=15,
            loc="left",
        )

        ax.set_ylabel("Victorias")

    # Solo necesitamos xlabel abajo
    axes[-1].set_xlabel("Ventaja ←   Gap de clasificación (%)   → Desventaja")

    # Estilo común
    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.2)

        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    driver_title = f"{driver[0].upper()}{driver[1:]}"
    output_path = os.path.join(
        os.getcwd(),
        "src",
        "outputs",
        "data_wins",
        driver,
        f"Gap_per_win_{driver_title}_by_teams.png",
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_hist_qualy_gap_from_victory(gaps, driver):

    gaps = np.asarray([gap for gap in gaps if gap != "NA"], dtype=float)

    if len(gaps) == 0:
        return

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=200)

    # Symmetrical bins around zero
    max_gap = np.ceil(max(abs(gaps.min()), abs(gaps.max())) * 4) / 4

    bins = np.arange(-max_gap, max_gap + 0.25, 0.25)

    ax.hist(gaps, bins=bins, edgecolor="white", linewidth=1.2, alpha=0.9)

    median_gap = np.median(gaps)
    median_gap_disavantage = np.median(gaps[gaps>0.0])
    median_gap_avantage = np.median(gaps[gaps<=0.0])

    # Neutral point
    ax.axvline(0, linewidth=2, color="black", label="Rendimiento igualado")

    # Median
    ax.axvline(
        median_gap,
        linestyle="--",
        color="black",
        linewidth=1.8,
        label=f"Mediana: {median_gap:+.2f}%",
    )
    ax.axvline(
        median_gap_avantage,
        linestyle=":",
        color="lime",
        linewidth=1.8,
        label=f"Mediana con Ventaja: {median_gap_avantage:+.2f}%",
    )
    ax.axvline(
        median_gap_disavantage,
        linestyle=":",
        color="red",
        linewidth=1.8,
        label=f"Mediana con Desventaja: {median_gap_disavantage:+.2f}%",
    )

    driver_title = f"{driver[0].upper()}{driver[1:]}"
    ax.set_title(
        f"Ventaja de clasificación en las victorias de {driver_title}",
        fontsize=20,
        fontweight="bold",
        loc="left",
        pad=18,
    )

    ax.text(
        0,
        1.01,
        f"Gap respecto al rival más rápido · {len(gaps)} victorias",
        transform=ax.transAxes,
        fontsize=14,
        alpha=0.65,
    )

    ax.set_xlabel(
        "Ventaja del coche ←   Gap de clasificación (%)   → Desventaja del coche",
        fontsize=14,
    )

    ax.set_ylabel("Número de victorias", fontsize=14)

    ax.grid(axis="y", linestyle="--", alpha=0.25)

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, fontsize=10)

    ax.margins(x=0.01)

    plt.tight_layout()

    output_path = os.path.join(
        os.getcwd(),
        "src",
        "outputs",
        "data_wins",
        driver,
        f"Gap_per_win_{driver_title}.png",
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_hist_qualy_gaps_from_victory(df_gaps, drivers):
    fig, ax = plt.subplots(len(drivers), figsize=(11, 15), dpi=200)

    for i, driver in enumerate(drivers):
        subset_driver = df_gaps[df_gaps["Driver"] == driver].copy()
        check_na = sum(subset_driver["Gap"].isna())
        gaps = np.asarray(subset_driver["Gap"].dropna().to_list())

        if len(gaps) == 0:
            return

        # Symmetrical bins around zero
        max_gap = (
            np.ceil(max(abs(df_gaps["Gap"].min()), abs(df_gaps["Gap"].max())) * 4) / 4
        )

        bins = np.arange(-max_gap, max_gap + 0.25, 0.25)

        ax[i].hist(
            gaps, bins=bins, edgecolor="white", density=True, linewidth=1.2, alpha=0.9
        )

        median_gap_avantage = subset_driver.loc[
            subset_driver["Gap"] <= 0.0, "Gap"
        ].median()
        median_gap_disavantage = subset_driver.loc[
            subset_driver["Gap"] > 0.0, "Gap"
        ].median()
        median_gap = np.median(gaps)

        # Neutral point
        ax[i].axvline(0, linewidth=2, color="black")

        # Median Gaps
        ax[i].axvline(
            median_gap_avantage,
            linestyle=":",
            color="lime",
            linewidth=1.8,
            label=f"Mediana con ventaja: {median_gap_avantage:+.2f}%",
        )
        ax[i].axvline(
            median_gap_disavantage,
            linestyle=":",
            color="red",
            linewidth=1.8,
            label=f"Mediana con desventaja: {median_gap_disavantage:+.2f}%",
        )
        ax[i].axvline(
            median_gap,
            linestyle="--",
            color="black",
            linewidth=1.8,
            label=f"Mediana: {median_gap:+.2f}%",
        )

        ax[i].set_title(
            f"Ventaja en Qualy en las victorias de {format_driver_name(driver)}",
            fontsize=20,
            fontweight="bold",
            loc="left",
            pad=18,
        )
        if check_na:
            msg = f"Gap respecto al rival más rápido · {len(gaps)}* victorias"
        else:
            msg = f"Gap respecto al rival más rápido · {len(gaps)} victorias"

        ax[i].text(
            0,
            1.01,
            msg,
            transform=ax[i].transAxes,
            fontsize=14,
            alpha=0.65,
        )

        if i == len(drivers) - 1:
            ax[i].set_xlabel(
                "Ventaja del coche ←   Gap de clasificación (%)   → Desventaja del coche",
                fontsize=14,
            )

        ax[i].set_ylabel("Número de victorias", fontsize=14)

        ax[i].grid(axis="y", linestyle="--", alpha=0.25)

        ax[i].set_axisbelow(True)

        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)

        ax[i].legend(frameon=False, fontsize=10)

        ax[i].margins(x=0.01)

    # Save
    plt.tight_layout()
    output_path = os.path.join(
        os.getcwd(), "src", "outputs", "data_wins", "Gap_per_win_drivers.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

def plot_boxplot_driver_wins(df_gaps, drivers):
    df_gaps["Con Ventaja"] = df_gaps["Gap"] <= 0.0

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    fig.suptitle(
        "Distribución del gap de clasificación en las victorias",
        fontsize=16,
        fontweight="bold",
        ha="right"
    )

    # -------- Con Ventaja
    df_ventaja = df_gaps[df_gaps["Con Ventaja"]==True]
    order_adv = (
        df_ventaja.groupby("Driver")["Gap"]
        .median()
        .sort_values()
        .index
    )

    sns.boxplot(
        data=df_ventaja.reset_index(drop=True),
        x="Driver",
        y="Gap",
        order=order_adv,
        showfliers=True,
        ax=ax[0],
    )

    ax[0].axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )

    ax[0].set_title(
        "Con Ventaja",
        fontsize=12,
        # fontweight="bold",
        loc="left"
    )

    ax[0].set_xlabel("Piloto")
    ax[0].set_ylabel("Gap de clasificación (%)")
    ax[0].tick_params(axis="x", labelrotation=30)

    ax[0].grid(axis="y", alpha=0.2)
    ax[0].set_axisbelow(True)

    # -------- Con DesVentaja
    df_desventaja = df_gaps[df_gaps["Con Ventaja"]==False]
    order_dis = (
        df_desventaja.groupby("Driver")["Gap"]
        .median()
        .sort_values()
        .index
    )
    sns.boxplot(
        data=df_desventaja.reset_index(drop=True),
        x="Driver",
        y="Gap",
        order=order_dis,
        color="orange",
        showfliers=True,
        ax=ax[1],
    )

    ax[1].axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )

    ax[1].set_title(
        "Con Desventaja",
        fontsize=12,
        # fontweight="bold",
        loc="left"
    )

    ax[1].set_xlabel("Piloto")
    ax[1].set_ylabel("Gap de clasificación (%)")
    ax[1].tick_params(axis="x", labelrotation=30)

    ax[1].grid(axis="y", alpha=0.2)
    ax[1].set_axisbelow(True)

    # Save
    plt.tight_layout()
    output_path = os.path.join(
        os.getcwd(), "src", "outputs", "data_wins", "Boxplot_Gap_per_win_drivers.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # plt.show()


def shown_resume_times(t_driver_wins, t_sessions, t_gaps, t_plot, t_stats):
    t_total = t_driver_wins + t_sessions + t_gaps + t_plot + t_stats
    print("\t")
    print("=" * 50)
    print("Times")
    print("=" * 50)
    print(f"t_driver_wins: {t_driver_wins:.2f}s {t_driver_wins / t_total * 100:.2f}%")
    print(f"t_sessions: {t_sessions:.2f}s {t_sessions / t_total * 100:.2f}%")
    print(f"t_gaps: {t_gaps:.2f}s {t_gaps / t_total * 100:.2f}%")
    print(f"t_plot: {t_plot:.2f}s {t_plot / t_total * 100:.2f}%")
    print(f"t_stats: {t_stats:.2f}s {t_stats / t_total * 100:.2f}%")
    print("=" * 50)


def make_resume_stats_single(driver, df_gaps):
    # Stats
    n_victories = df_gaps.shape[0]
    n_nans = sum(df_gaps["Gap"].isna())

    gap_med = df_gaps["Gap"].mean()
    gap_median = df_gaps["Gap"].median()
    advantage_ratio = sum(df_gaps["Gap"] <= 0.0) / n_victories
    disadvantage_ratio = 1 - advantage_ratio
    print("\n")
    print("Piloto\tVictorias\tNaN\tGap medio\tMediana\t% con ventaja% con desventaja")
    print(
        f"{driver}\t{n_victories}\t{n_nans}\t{gap_med:.2f}%\t{gap_median:.2f}%\t{advantage_ratio * 100:.2f}%\t{disadvantage_ratio * 100:.2f}%"
    )


def make_resume_stats_multi(df_gaps):
    df_grouped = df_gaps.groupby(
        ["Driver"],
        observed=True,
        sort=False,
    ).agg(
        Victorias=("Gap", "size"),
        NaN=("Gap", lambda x: x.isna().sum()),
        GapMed=("Gap", "mean"),
        GapMedian=("Gap", "median"),
        AdvantageRatio=("Gap", lambda x: (x <= 0.0).sum() / x.count()),
        GapMedianAdvantage=("Gap", lambda x: x[x <= 0.0].median()),
        GapMedianDisadvantage=("Gap", lambda x: x[x > 0.0].median()),
    )
    df_grouped["% con Ventaja"] = df_grouped["AdvantageRatio"] * 100
    df_grouped["% con Desventaja"] = (1 - df_grouped["AdvantageRatio"]) * 100

    df_grouped = df_grouped.drop(["AdvantageRatio"], axis=1)
    df_grouped = df_grouped.sort_values(by="GapMedian", ascending=False)
    df_grouped = df_grouped[
        [
            "Victorias",
            "NaN",
            "GapMed",
            "GapMedian",
            "% con Ventaja",
            "GapMedianAdvantage",
            "% con Desventaja",
            "GapMedianDisadvantage",
        ]
    ]
    print(
        tabulate(df_grouped.round(2), headers="keys", showindex=True, tablefmt="grid")
    )


# ====== MAIN ==========================================================================================================
async def main_single_driver(driver):
    path = os.path.join(os.getcwd(), "src", "outputs", "data_wins", driver)
    os.makedirs(path, exist_ok=True)

    # Get df driver_wins
    t0 = time.time()
    path = os.path.join(
        os.getcwd(), "src", "outputs", "data_wins", driver, f"wins_{driver}.json"
    )
    if not os.path.exists(path):
        list_session_wins = await get_driver_wins(driver)
    else:
        list_session_wins = get_driver_wins_from_path(path)
    t_driver_wins = time.time() - t0

    # Load all sesions
    path = os.path.join(
        os.getcwd(), "src", "outputs", "data_wins", driver, f"df_{driver}.csv"
    )
    if not os.path.exists(path):
        t0_sessions = time.time()

        if PARALELL:
            with ThreadPoolExecutor(max_workers=3) as executor:
                sessions = executor.map(
                    lambda args: load_session(*args), list_session_wins
                )
        else:
            sessions = []
            for year, gp, identifier in list_session_wins:
                session = load_session(year, gp, identifier)
                sessions.append(session)
                # time.sleep(1)
        t_sessions = time.time() - t0_sessions

        # Process sesions results to get gap
        t0_gaps = time.time()
        df_gaps = get_df_gaps(sessions, driver)
        df_gaps.to_csv(path, index=False, sep=",")
        t_gaps = time.time() - t0_gaps
    else:
        df_gaps = get_df_gaps_from_path(path)
        t_sessions = 0.0
        n_sessions_nan = sum(df_gaps["Gap"].isna())
        if n_sessions_nan > 0:
            print(f"Faltan por cargar {n_sessions_nan}")
            t0_sessions = time.time()
            if PARALELL:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    sessions = executor.map(
                        lambda args: load_session(*args), list_session_wins
                    )
            else:
                sessions = []
                subset = df_gaps.loc[df_gaps["Gap"].isna(),]
                list_session_wins = [
                    (year, gp, "Q") for year, gp in zip(subset["Year"], subset["Round"])
                ]
                print(len(list_session_wins))
                for year, gp, identifier in list_session_wins:
                    session = load_session(year, gp, identifier)

                    results = session.results
                    if not results.empty:
                        gap, team = get_qualy_gap(session.results, driver)
                        mask_year = df_gaps["Year"] == year
                        mask_race = df_gaps["Round"] == gp
                        # mask_race = df_gaps["Race"] == gp

                        df_gaps.loc[mask_year & mask_race, "Gap"] = gap
                        df_gaps.loc[mask_year & mask_race, "Team"] = team
                    # time.sleep(1)

                # Process sesions results to get gap
                t0_gaps = time.time()
                df_gaps.to_csv(path, index=False, sep=",")
                t_gaps = time.time() - t0_gaps

            t_sessions = time.time() - t0_sessions

        t_gaps = 0.0

    # Final stats
    t0_stats = time.time()
    make_resume_stats_single(driver, df_gaps)
    t_stats = time.time() - t0_stats

    # Plot
    if DROPNA:
        df_gaps = df_gaps[~df_gaps["Gap"].isna()]

    t0_plot = time.time()
    plot_gap_by_team(df_gaps, driver)
    plot_hist_qualy_gap_from_victory(df_gaps["Gap"].to_list(), driver)
    t_plot = time.time() - t0_plot

    # shown_resume_times(t_driver_wins, t_sessions, t_gaps, t_plot, t_stats)


async def main_multi_driver(drivers):
    # Get df driver_wins
    dict_drivers_session_wins = await get_drivers_wins(drivers)

    # Get df_gaps
    df_gaps = pd.DataFrame()
    for driver in drivers:
        # Get df from driver
        path = os.path.join(
            os.getcwd(), "src", "outputs", "data_wins", driver, f"df_{driver}.csv"
        )
        if not os.path.exists(path):
            if PARALELL:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    sessions = executor.map(
                        lambda args: load_session(*args),
                        dict_drivers_session_wins["unique_sessions"],
                    )
            else:
                sessions = []
                for year, gp, identifier in dict_drivers_session_wins[driver]:
                    session = load_session(year, gp, identifier)
                    sessions.append(session)

            # Process sesions results to get gap
            df_gaps_driver = get_df_gaps(sessions, driver)
            df_gaps_driver.to_csv(path, index=False, sep=",")
        else:
            df_gaps_driver = get_df_gaps_from_path(path)

        # Concat
        df_gaps_driver["Driver"] = [driver] * df_gaps_driver.shape[0]
        df_gaps = pd.concat([df_gaps, df_gaps_driver])
        df_gaps = df_gaps.reset_index(drop=True)

    print(f"Final df shape {df_gaps.shape}")

    # Final stats
    make_resume_stats_multi(df_gaps)

    # Plots
    plot_hist_qualy_gaps_from_victory(df_gaps, drivers)
    plot_boxplot_driver_wins(df_gaps, drivers)


# ======================================================================================================================
if __name__ == "__main__":
    # Cache
    ff1.Cache.enable_cache("src/cache")
    pd.options.mode.chained_assignment = None
    path = os.path.join(os.getcwd(), "src", "outputs", "data_wins")
    os.makedirs(path, exist_ok=True)

    # driver = ["alonso"]
    driver = ["alonso", "michael_schumacher", "max_verstappen", "vettel", "hamilton"]

    if len(driver) == 1:
        asyncio.run(main_single_driver(driver[0]))
    else:
        asyncio.run(main_multi_driver(driver))
