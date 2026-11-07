# Library
import os
import logging
import numpy as np
import pandas as pd
import fastf1 as ff1

from fastf1 import plotting
from fastf1 import utils

def main(year, test_number, session_number):
    # 1) Config de ff1 and in general
    ff1.plotting.setup_mpl()
    ff1.Cache.enable_cache("./cache")
    pd.options.mode.chained_assignment = None

    # 2) Cargar sesion de test
    event = ff1.get_session(year, grand_prix, session)
    event.load()

    print(event)

if __name__ == '__main__':
    # ------- INPUTS ------
    year = 2024
    grand_prix = 'AbuDhabi'
    session = 'FP1'

    main(year, grand_prix, session)