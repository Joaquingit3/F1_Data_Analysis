import numpy as np
import pandas as pd

def create_sesion_csv(sesion):
    # Extract laps 
    laps = sesion.laps

    # Extract unique driver list
    drivers = np.unique(laps['Driver'])

    # Loop every driver
    drivers_data = []
    for driver in drivers:
        ## Extract data from their laps 
        driver_laps = laps.loc[laps['Driver'] == driver,]
        team = driver_laps['Team'].iloc[0]
        best_lap = "" # TODO

        ## Save in a dict
        drivers_data.append({"Driver": driver, "Team": team, "BestLapTime": best_lap})

        del driver_laps


    # Save as df of the session and return