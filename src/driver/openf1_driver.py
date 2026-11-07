import numpy as np
import pandas as pd
import requests

from dataclasses import dataclass

from tabulate import tabulate
from .plotting_utils import setup_mpl
from .utils import (
    format_laptime,
    format_openf1_df_to_fastf1,
    get_phase,
    format_laps,
    format_race_control,
    format_telemetry,
)
from . import BASE_OPENF1_URL


@dataclass
class Event:
    EventName: str
    year: int


@dataclass
class CountryClass:
    Code: str


@dataclass
class MeetingClass:
    Number: int
    Country: CountryClass


@dataclass
class SessionInfo:
    Meeting: MeetingClass
    Type: str


@dataclass
class Session:
    event: Event
    name: str
    session_info: SessionInfo
    session_key: str
    session_status: pd.DataFrame

class OpenF1AuthenticationError(Exception):
    pass

class OpenF1Driver:
    def __init__(self):
        self.base_url = BASE_OPENF1_URL
        self.timeout = 60
        self.date_format = "%Y-%m-%dT%H:%M:%S.%f"
        self._config_plots()
        pd.options.mode.chained_assignment = None

    def _config_plots(self):
        setup_mpl(color_scheme="fastf1")

    def get_openf1_data(self, endpoint, **kwargs):
        response = requests.get(
            f"{self.base_url}/{endpoint}", params=kwargs, timeout=self.timeout
        )
        if response.ok:
            print(f"[OpenF1] Finished calling {endpoint}")
            return pd.DataFrame(response.json())

        if response.status_code == 401:
            raise OpenF1AuthenticationError(
                f"🔒 OpenF1 requiere autenticación.\n{response.json()['detail']}"
            )

        raise requests.HTTPError(
            f"[OpenF1] Error {response.status_code}: {response.text}",
            response=response
        )

    def get_telemetry_from_driver(self, df_results, lap, race_key, driver):
        driver_number = df_results.loc[
            df_results["Driver"] == driver, "DriverNum"
        ].iloc[0]
        start = pd.to_datetime(lap["Time"], utc=True)
        end = start + pd.to_timedelta(lap["lap_duration"], unit="s")
        data = self.get_openf1_data(
            "car_data", session_key=int(race_key), driver_number=int(driver_number)
        )
        mask_start = data["date"] >= start.strftime(self.date_format)[:-3]
        mask_end = data["date"] <= end.strftime(self.date_format)[:-3]
        data = data[mask_start & mask_end]
        return format_telemetry(data)

    def get_laps_from_driver(self, df_results, race_key, driver):
        driver_number = df_results.loc[
            df_results["Driver"] == driver, "DriverNum"
        ].iloc[0]
        driver_team = df_results.loc[df_results["Driver"] == driver, "equipo"].iloc[0]
        laps = self.get_openf1_data(
            "laps", session_key=int(race_key), driver_number=driver_number
        )
        laps["Team"] = [driver_team] * laps.shape[0]
        return format_laps(laps)

    def get_results(self, race_key):
        return self.get_openf1_data("session_result", session_key=int(race_key))

    def get_sesion_info(self, year, location, session_name):
        meetings = self.get_openf1_data(
            "meetings",
            year=year,
        )
        meetings = meetings[
            meetings["meeting_name"].str.contains("Grand Prix", case=False, na=False)
            & ~meetings["is_cancelled"].fillna(False)
        ]

        meetings = meetings.sort_values("date_start").reset_index(drop=True)
        meetings["race_num"] = range(1, len(meetings) + 1)

        meeting = meetings[meetings["location"] == location].iloc[0]

        session_data = self.get_openf1_data(
            "sessions",
            meeting_key=int(meeting["meeting_key"]),
            session_name=session_name,
        ).iloc[0]

        # Get sesion_status
        race_control = self.get_openf1_data(
            "race_control",
            date=str(pd.Timestamp(session_data["date_start"]).date()),
            session_key=int(session_data["session_key"]),
        )
        race_control = format_race_control(race_control)

        session = Session(
            event=Event(EventName=meeting["meeting_name"], year=int(meeting["year"])),
            name=session_data["session_name"],
            session_info=SessionInfo(
                Meeting=MeetingClass(
                    Number=int(meeting["race_num"]),
                    Country=CountryClass(Code=meeting["country_code"]),
                ),
                Type=session_name,
            ),
            session_key=session_data["session_key"],
            session_status=race_control,
        )
        return session


    def get_qualifying_results(self, year, location):
        # 1. Obtener la sesión de clasificación
        sessions = self.get_openf1_data(
            "sessions", year=year, location=location, session_name="Qualifying"
        )

        if sessions.empty:
            raise ValueError("No se encontró la sesión")

        session_key = int(sessions.iloc[0]["session_key"])

        # 2. Obtener resultados oficiales
        results = self.get_openf1_data("session_result", session_key=session_key)

        # 3. Obtener pilotos y equipos
        drivers = self.get_openf1_data("drivers", session_key=session_key)

        # 4. Unir ambos DataFrames
        df = results.merge(
            drivers[["driver_number", "name_acronym", "team_name"]],
            on="driver_number",
            how="left",
        )

        # 5. Extraer Q1, Q2 y Q3
        for i, phase in enumerate(["Q1", "Q2", "Q3"]):
            df[phase] = df["duration"].apply(lambda x: get_phase(x, i))

        # 6. Convertir segundos a M:SS.mmm
        for phase in ["Q1", "Q2", "Q3"]:
            df[phase] = df[phase].apply(format_laptime)

        # 7. Seleccionar columnas y ordenar
        return format_openf1_df_to_fastf1(df)

    def get_circuit_corners(self, year, location):
        meeting = self.get_openf1_data("meetings", year=year, location=location).iloc[0]

        circuit_url = meeting["circuit_info_url"]

        response = requests.get(
            circuit_url, headers={"User-Agent": "FastF1/"}, timeout=30
        )
        response.raise_for_status()

        circuit_info = response.json()

        corners = pd.DataFrame(circuit_info["corners"])

        corners = corners.rename(
            columns={
                "number": "Number",
                "letter": "Letter",
                "angle": "Angle",
                "length": "Distance",
            }
        )

        # Las coordenadas pueden venir anidadas
        if "trackPosition" in corners.columns:
            positions = pd.json_normalize(corners["trackPosition"])
            corners["Y"] = positions["y"].to_numpy()
            corners["X"] = positions["x"].to_numpy()

        corners["Distance"] = corners["Distance"] / 10

        if "Letter" in corners.columns:
            return corners[["X", "Y", "Number", "Letter", "Angle", "Distance"]]
        else:
            return corners[["X", "Y", "Number", "Angle", "Distance"]]

