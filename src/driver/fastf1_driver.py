import fastf1 as ff1
import pandas as pd
from fastf1 import plotting, utils
from fastf1.exceptions import DataNotLoadedError

from src.utils.functions import format_time
from . import WEATHER_COLUMNS


class FastF1Driver:
    def __init__(self, year, grand_prix, session, **kwargs):
        self._config()
        self.session = self.get_session(year, grand_prix, session, **kwargs)
        self.session = self.session.load()

    def _config(self):
        ff1.plotting.setup_mpl()
        ff1.Cache.enable_cache("src/cache")
        pd.options.mode.chained_assignment = None

    def get_session(self, year, grand_prix, session, **kwargs):
        try:
            session_data = ff1.get_session(year, int(grand_prix), session, **kwargs)
        except ValueError:
            session_data = ff1.get_session(year, grand_prix, session, **kwargs)
        return session_data

    def get_circuit_info(self):
        try:
            corners = self.session.get_circuit_info().corners
        except AttributeError:
            corners = []
            print("WARNING\tcorners NOT Exists!")
        except DataNotLoadedError:
            corners = []
            print("WARNING\tcorners NOT Exists!")
        return corners

    def add_weather_data(self, lap_more_data, time_of_lap, laptime):
        try:
            df_weather = self.session.weather_data
            weather_lap = df_weather[(df_weather['Time'] >= time_of_lap) & (df_weather['Time'] <= time_of_lap+laptime)]
            weather_lap['WeatherTime'] = weather_lap['Time']
            weather_lap.pop('Time')

            # Save weather columns
            for col in weather_lap.columns.to_list():
                lap_more_data[col] = weather_lap[col].iloc[0]
        except DataNotLoadedError:
            # Save weather columns
            for col in WEATHER_COLUMNS:
                lap_more_data[col] = "NaN"

        return lap_more_data


    def get_laps_from_driver(self, driver):
        return self.session.laps.pick_drivers(driver)

    def _get_best_time_of_session(self, df, session):
        # Extract the subset of not NA times in the desire session
        mask_NA = df["LapTime"].isna()
        subset = df.loc[~mask_NA,]
        mask_session = subset["SessionTime"] == session
        subset_session = subset.loc[mask_session,]

        if subset_session.shape[0] == 0:
            official_time = "NoTime"
        else:
            official_time = min(subset_session["LapTime"])

        return official_time

    def _get_init_and_finish_sesion(self):
        # Calcular el listado de objetos para delimitar las sesiones
        df_session_status = self.session.session_status
        finished_list = df_session_status.loc[
            df_session_status["Status"] == "Finished", "Time"
        ].to_list()
        mask_status_started = df_session_status["Status"] == "Started"
        started_list = []
        for i in range(len(finished_list)):
            if i == 0:
                mask_init = (
                    df_session_status["Time"] >= df_session_status["Time"].iloc[0]
                )
                mask_finish = df_session_status["Time"] <= finished_list[i]
            else:
                mask_init = df_session_status["Time"] >= finished_list[i - 1]
                mask_finish = df_session_status["Time"] <= finished_list[i]

            value = df_session_status.loc[
                mask_init & mask_finish & mask_status_started, "Time"
            ].to_list()[0]
            started_list.append(value)
        return started_list, finished_list

    def _generate_sessionTime(self, time, started_sesions, finished_sesions):
        if (time >= started_sesions[0]) and (time <= finished_sesions[0]):
            # Estamos en Q1
            return "Q1"
        else:
            if (time >= started_sesions[1]) and (time <= finished_sesions[1]):
                # Estamos en Q2
                return "Q2"
            else:
                if (time >= started_sesions[2]) and (time <= finished_sesions[2]):
                    # Estamos en Q3
                    return "Q3"
                else:
                    # Se hizo despues de acabar el cronometro
                    if (time >= finished_sesions[0]) and (time <= started_sesions[1]):
                        return "Q1"
                    elif (time >= finished_sesions[1]) and (time <= started_sesions[2]):
                        return "Q2"
                    elif time >= finished_sesions[2]:
                        return "Q3"

    def get_lap_number_of_session(self, laps_driver, desire_session):
        # Añadir Session of the lap (Q1, Q2, Q3)
        started_sesions, finished_sesions = self._get_init_and_finish_sesion()
        laps_driver["SessionTime"] = laps_driver["Time"].apply(
            lambda time: self._generate_sessionTime(
                time, started_sesions, finished_sesions
            )
        )
        # CHHECK THE LAPSTIMES OF THE DRIVER
        best_lap = self._get_best_time_of_session(laps_driver, desire_session)

        # Extract lapnumber
        lap_number = laps_driver.loc[
            laps_driver["LapTime"] == best_lap, "LapNumber"
        ].iloc[0]
        return lap_number

    def get_specific_lap_driver(
        self, laps_driver, best_session: str | None, lap_number: int | None
    ):
        if lap_number is not None:
            if best_session is not None:
                lap_number_driver = self.get_lap_number_of_session(
                    laps_driver, best_session
                )
            else:
                raise ValueError(
                    "get_specific_lap_driver method needs or best_session or lap_number param"
                )
        else:
            lap_number_driver = lap_number

        return laps_driver[laps_driver["LapNumber"] == lap_number_driver].iloc[0]

    def get_telemetry_from_lap(self, fastest_lap):
        return fastest_lap.get_telemetry().add_distance()

    def show_laptime(self, driver, telemetry, fastest_driver):
        # Calculate the rows of the telemetry
        nrow = telemetry.shape[0]

        print(f"Tiempo {driver}: "
            f"{format_time(telemetry.Time[nrow + 1])}, "
            f"S1: {format_time(fastest_driver.Sector1Time)}, "
            f"S2: {format_time(fastest_driver.Sector2Time)}, "
            f"S3: {format_time(fastest_driver.Sector3Time)}")

    def add_lap_info(self, lap):
        # Extract info of the lap 
        driver = lap['Driver']
        laptime = lap['LapTime']
        team = lap['Team']

        driver_laps = self.session.laps.pick_drivers(driver)
        time_of_lap = driver_laps.loc[driver_laps['LapTime']==laptime, 'LapStartTime'].iloc[0]
        compound = driver_laps.loc[driver_laps['LapTime']==laptime, 'Compound'].iloc[0]

        # Save new columns
        lap_more_data = {}
        lap_more_data['Compound'] = compound
        lap_more_data['Time'] = time_of_lap
        lap_more_data['LapTime'] = laptime
        lap_more_data['Driver'] = driver
        lap_more_data['TeamName'] = team
        
        # Weather extraction
        lap_more_data = self.add_weather_data(lap_more_data, time_of_lap, laptime)

        # Select only this columns
        df_lap = pd.DataFrame(lap_more_data, index=[0])
        df_lap = df_lap[['LapTime', 'Driver', 'Time', 'TeamName'] + WEATHER_COLUMNS]
        return df_lap