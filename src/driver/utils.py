import numpy as np
import pandas as pd
from datetime import datetime

from tabulate import tabulate
from . import DICT_ABR_TO_NAME, DICT_ABR_TO_NACIONALITY


def format_laptime(seconds):
    if pd.isna(seconds):
        return "NoTime"
    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes}:{remaining:06.3f}"


def format_laptime_lap_df(seconds):
    if pd.isna(seconds):
        return seconds
    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes}:{remaining:06.3f}"


def get_phase(values, index):
    if isinstance(values, list) and len(values) > index:
        return values[index]
    return None


def get_best_time(subset):
    laptimes = []
    for i in range(subset.shape[0]):
        if subset["Q3"].iloc[i] == "NoTime":
            if subset["Q2"].iloc[i] == "NoTime":
                laptimes.append(subset["Q1"].iloc[i])
            else:
                laptimes.append(subset["Q2"].iloc[i])
        else:
            laptimes.append(subset["Q3"].iloc[i])
    return laptimes


def calculate_delta(fastest, time):
    # Adapt time if it is possible
    if time != "NoTime":
        time = datetime.strptime(time, "%M:%S.%f")
        # Calculate delta
        delta = time - fastest

        # Extract seconds and milliseconds
        seconds = delta.seconds
        milliseconds = delta.microseconds // 1000

        if milliseconds < 100:
            delta_string = "+" + str(seconds) + ".0" + str(milliseconds)
        else:
            delta_string = "+" + str(seconds) + "." + str(milliseconds)

        return delta_string

    else:
        return "NA"


def calculate_gap_percentage(fastest, time):
    # Adapt time if it is possible
    if time != "NoTime":
        time = datetime.strptime(time, "%M:%S.%f")

        # Convertimos to timedelta
        reference = datetime(1900, 1, 1)
        fastest_td = fastest - reference
        time_td = time - reference

        # Calculate delta
        delta = (
            (time_td - fastest_td).total_seconds() / fastest_td.total_seconds()
        ) * 100
        return round(delta, 3)
    else:
        return "NA"


def add_gap(df_timelist):
    # fastest time
    fastest_time = df_timelist.loc[0, "LapTime"]
    if fastest_time == "NoTime" or pd.isna(fastest_time):
        raise ValueError("Fastest time of the table is NoTime")
    else:
        fastest_time = datetime.strptime(fastest_time, "%M:%S.%f")

        # GAP Time
        # Create Gap column
        df_timelist["Gap"] = df_timelist["LapTime"].apply(
            lambda i: calculate_delta(fastest_time, i)
        )

        # GAP %
        df_timelist["Gap (%)"] = df_timelist["LapTime"].apply(
            lambda i: calculate_gap_percentage(fastest_time, i)
        )

        # Update the firstone to none
        df_timelist.loc[0, "Gap"] = ""
        return df_timelist


def format_openf1_df_to_fastf1(df):
    df = df.rename(
        columns={
            "name_acronym": "Driver",
            "team_name": "equipo",
            "position": "posicion",
            "driver_number": "DriverNum",
        }
    )

    df["piloto"] = df["Driver"].apply(lambda i: DICT_ABR_TO_NAME[i])
    df["nacionalidad"] = df["Driver"].apply(lambda i: DICT_ABR_TO_NACIONALITY[i])
    df["LapTime"] = get_best_time(df)
    df = add_gap(df)

    df = df[
        [
            "posicion",
            "nacionalidad",
            "Driver",
            "DriverNum",
            "equipo",
            "LapTime",
            "Gap",
            "Gap (%)",
        ]
    ]

    df.sort_values(
        "posicion", key=lambda x: pd.to_numeric(x, errors="coerce"), na_position="last"
    ).reset_index(drop=True)
    # df["posicion"] = pd.to_numeric(
    #     df["posicion"],
    #     errors="coerce"
    # ).astype("Int64")

    # df = df.sort_values(
    #     "posicion",
    #     na_position="last"
    # ).reset_index(drop=True)
    return df


def format_laps(laps: pd.DataFrame):
    laps = laps.rename(
        columns={
            "date_start": "Time",
            "driver_number": "DriverNum",
            "duration_sector_1": "Sector1Time",
            "duration_sector_2": "Sector2Time",
            "duration_sector_3": "Sector3Time",
            "i1_speed": "I1Speed",
            "i2_speed": "I2Speed",
            "st_speed": "STSpeed",
            "lap_number": "LapNumber",
        }
    )
    laps["LapTime"] = laps["lap_duration"].apply(lambda i: format_laptime_lap_df(i))
    laps["Sector1Time"] = pd.to_timedelta(laps["Sector1Time"], unit="s")
    laps["Sector2Time"] = pd.to_timedelta(laps["Sector2Time"], unit="s")
    laps["Sector3Time"] = pd.to_timedelta(laps["Sector3Time"], unit="s")
    return laps[
        [
            "Time",
            "LapNumber",
            "LapTime",
            "lap_duration",
            "Sector1Time",
            "Sector2Time",
            "Sector3Time",
            "I1Speed",
            "I2Speed",
            "STSpeed",
            "Team"
        ]
    ]


def format_race_control(race_control: pd.DataFrame):
    # Cols:
    # - Status: Finished or Started
    # - Time
    race_control = race_control.rename(columns={"date": "Time"})
    mask_flag = race_control["category"] == "Flag"
    race_control = race_control[mask_flag]
    race_control["Status"] = race_control["flag"].map(
        {"GREEN": "Started", "CHEQUERED": "Finished", "CHECKERED": "Finished"}
    )
    # print(tabulate(race_control, headers="keys", showindex=False, tablefmt="grid"))

    return race_control


def format_telemetry(telemetry: pd.DataFrame) -> pd.DataFrame:
    telemetry = telemetry.rename(
        columns={
            "date": "SessionTime",
            "brake": "Brake",
            "speed": "Speed",
            "throttle": "Throttle",
            "n_gear": "nGear",
            "rpm": "RPM",
            "drs": "DRS",
        }
    )
    telemetry["SessionTime"] = pd.to_datetime(telemetry["SessionTime"], utc=True)
    telemetry["Time"] = telemetry["SessionTime"].diff().fillna(pd.Timedelta(0)).cumsum()

    telemetry = telemetry.sort_values("SessionTime").reset_index(drop=True)
    dt = telemetry["SessionTime"].diff().dt.total_seconds().fillna(0)

    telemetry["Distance"] = (telemetry["Speed"] / 3.6 * dt).cumsum()
    return telemetry[
        [
            "SessionTime",
            "Time",
            "Speed",
            "Throttle",
            "Brake",
            "nGear",
            "RPM",
            "DRS",
            "Distance",
        ]
    ]

def delta_time(reference_lap: pd.DataFrame, compare_lap: pd.DataFrame):
    ref = reference_lap.copy()
    comp = compare_lap.copy()

    def mini_pro(stream):
        dstream_start = stream[1] - stream[0]
        dstream_end = stream[-1] - stream[-2]

        return np.concatenate(
            [[stream[0] - dstream_start], stream, [stream[-1] + dstream_end]]
        )

    # Convertir Time a segundos
    ref_time = ref["Time"].dt.total_seconds().to_numpy()
    comp_time = comp["Time"].dt.total_seconds().to_numpy()

    # Distancias acumuladas
    ref_distance = ref["Distance"].to_numpy()
    comp_distance = comp["Distance"].to_numpy()

    # Normalizar la distancia de ambas vueltas
    multiplier = ref_distance[-1] / comp_distance[-1]

    # Extrapolar extremos
    ltime = mini_pro(comp_time)
    ldistance = mini_pro(comp_distance) * multiplier

    # Interpolar tiempo de la vuelta comparada
    lap_time = np.interp(ref_distance, ldistance, ltime)

    # Diferencia de tiempo
    delta = lap_time - ref_time

    return delta, ref, comp
