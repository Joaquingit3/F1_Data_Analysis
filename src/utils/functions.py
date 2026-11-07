# Library
import os
import requests
import numpy as np
import pandas as pd
import fastf1 as ff1

from io import BytesIO
from PIL import Image, ImageOps, ImageEnhance
from pdf2image import convert_from_path

from datetime import timedelta
from datetime import datetime
from tabulate import tabulate
from fastf1.exceptions import DataNotLoadedError
from src.grapher.graphs_basic import plot_results_table
from src.common.const import *


def format_time(time):
    """
    Función que permite convertir una variable Timedelta
    en formato 0 days 00:01:25.858000 a tiempo por
    vuelta como 1:25.858 con minutos, segundos y microsegundos
    o a tiempo delta 1.428s si no tiene minutos
    """

    minutos = time.seconds // 60
    segundos = time.seconds % 60
    microsegundos = time.microseconds // 1000

    if minutos > 0:
        # Es un tiempo por vuelta
        formato = "{:d}:{:02d}.{:03d}".format(minutos, segundos, microsegundos)
        return formato
    else:
        # Es un tiempo delta
        formato = "{:d}.{:03d}".format(segundos, microsegundos)
        return formato


def show_laptime(driver, telemetry, fastest_driver):
    # Calculate the rows of the telemetry
    print(
        f"Tiempo {driver}: "
        f"{format_time(telemetry.Time.iloc[-1])}, "
        f"S1: {format_time(fastest_driver.Sector1Time)}, "
        f"S2: {format_time(fastest_driver.Sector2Time)}, "
        f"S3: {format_time(fastest_driver.Sector3Time)}"
    )


def calculate_deltas_twodrivers(
    telemetry_driver_1,
    telemetry_driver_2,
    fastest_driver_1,
    fastest_driver_2,
    driver_1,
    driver_2,
):
    # Calculamos el tamaño de los vectores
    if format_time(telemetry_driver_2.Time.iloc[-1]) > format_time(
        telemetry_driver_1.Time.iloc[-1]
    ):
        # El piloto 2 esta por detras del 1
        delta = telemetry_driver_2.Time.iloc[-1] - telemetry_driver_1.Time.iloc[-1]

        # Mirar el Primer Sector
        if fastest_driver_2.Sector1Time > fastest_driver_1.Sector1Time:
            # El Piloto 2 ha hecho peor sector 1 que el piloto 1
            delta_s1 = fastest_driver_2.Sector1Time - fastest_driver_1.Sector1Time

            # Mirar el Segundo Sector
            if fastest_driver_2.Sector2Time > fastest_driver_1.Sector2Time:
                # El Piloto 2 ha hecho peor sector 2 que el piloto 1
                delta_s2 = fastest_driver_2.Sector2Time - fastest_driver_1.Sector2Time

                # Mirar el Tercer sector
                if fastest_driver_2.Sector3Time > fastest_driver_1.Sector3Time:
                    # El Piloto 2 ha hecho peor sector 3 que el piloto 1
                    delta_s3 = (
                        fastest_driver_2.Sector3Time - fastest_driver_1.Sector3Time
                    )
                    print(
                        f"Delta {driver_2}-{driver_1}: {format_time(delta)}s, S1: {format_time(delta_s1)}s, S2: {format_time(delta_s2)}s, S3: {format_time(delta_s3)}s\n"
                    )

                else:
                    # El Piloto 2 ha hecho peor sector 3 que el piloto 1
                    delta_s3 = (
                        fastest_driver_1.Sector3Time - fastest_driver_2.Sector3Time
                    )
                    print(
                        f"Delta {driver_2}-{driver_1}: {format_time(delta)}s, S1: {format_time(delta_s1)}s, S2: {format_time(delta_s2)}s, S3: -{format_time(delta_s3)}s\n"
                    )

            else:
                # El Piloto 2 ha hecho mejor sector 2 que el piloto 1
                delta_s2 = fastest_driver_1.Sector2Time - fastest_driver_2.Sector2Time

                # Mirar el Tercer sector
                if fastest_driver_2.Sector3Time > fastest_driver_1.Sector3Time:
                    # El Piloto 2 ha hecho peor sector 3 que el piloto 1
                    delta_s3 = (
                        fastest_driver_2.Sector3Time - fastest_driver_1.Sector3Time
                    )
                    print(
                        f"Delta {driver_2}-{driver_1}: {format_time(delta)}, S1: {format_time(delta_s1)}, S2: {format_time(delta_s2)}, S3: {format_time(delta_s3)}\n"
                    )

                else:
                    # El Piloto 2 ha hecho peor sector 3 que el piloto 1
                    delta_s3 = (
                        fastest_driver_1.Sector3Time - fastest_driver_2.Sector3Time
                    )
                    print(
                        f"Delta {driver_2}-{driver_1}: {format_time(delta)}, S1: {format_time(delta_s1)}, S2: {format_time(delta_s2)}, S3: -{format_time(delta_s3)}\n"
                    )

        else:
            # El Piloto 2 ha hecho mejor sector 1 que el piloto 1
            delta_s1 = fastest_driver_1.Sector1Time - fastest_driver_2.Sector1Time

            # Mirar el Segundo Sector
            if fastest_driver_2.Sector2Time > fastest_driver_1.Sector2Time:
                # El Piloto 2 ha hecho peor sector 2 que el piloto 1
                delta_s2 = fastest_driver_2.Sector2Time - fastest_driver_1.Sector2Time

                # Mirar el Tercer sector
                if fastest_driver_2.Sector3Time > fastest_driver_1.Sector3Time:
                    # El Piloto 2 ha hecho peor sector 3 que el piloto 1
                    delta_s3 = (
                        fastest_driver_2.Sector3Time - fastest_driver_1.Sector3Time
                    )
                    print(
                        f"Delta {driver_2}-{driver_1}: {format_time(delta)}, S1: -{format_time(delta_s1)}, S2: {format_time(delta_s2)}, S3: {format_time(delta_s3)}\n"
                    )

                else:
                    # El Piloto 2 ha hecho mejor sector 3 que el piloto 1
                    delta_s3 = (
                        fastest_driver_1.Sector3Time - fastest_driver_2.Sector3Time
                    )
                    print(
                        f"Delta {driver_2}-{driver_1}: {format_time(delta)}, S1: -{format_time(delta_s1)}, S2: {format_time(delta_s2)}, S3: -{format_time(delta_s3)}\n"
                    )

            else:
                # El Piloto 2 ha hecho mejor sector 2 que el piloto 1
                delta_s2 = fastest_driver_1.Sector2Time - fastest_driver_2.Sector2Time

                # Mirar el Tercer sector
                if fastest_driver_2.Sector3Time > fastest_driver_1.Sector3Time:
                    # El Piloto 2 ha hecho peor sector 3 que el piloto 1
                    delta_s3 = (
                        fastest_driver_2.Sector3Time - fastest_driver_1.Sector3Time
                    )
                    print(
                        f"Delta {driver_2}-{driver_1}: {format_time(delta)}, S1: -{format_time(delta_s1)}, S2: -{format_time(delta_s2)}, S3: {format_time(delta_s3)}\n"
                    )

                else:
                    # El Piloto 2 ha hecho mejor sector 3 que el piloto 1
                    delta_s3 = (
                        fastest_driver_1.Sector3Time - fastest_driver_2.Sector3Time
                    )
                    print(
                        f"Delta {driver_2}-{driver_1}: {format_time(delta)}, S1: -{format_time(delta_s1)}, S2: -{format_time(delta_s2)}, S3: -{format_time(delta_s3)}\n"
                    )

    else:
        # El piloto 1 esta por detras del 2
        delta = telemetry_driver_1.Time.iloc[-1] - telemetry_driver_2.Time.iloc[-1]

        # Mirar el Primer Sector
        if fastest_driver_2.Sector1Time > fastest_driver_1.Sector1Time:
            # El Piloto 2 ha hecho peor sector 1 que el piloto 1
            delta_s1 = fastest_driver_2.Sector1Time - fastest_driver_1.Sector1Time

            # Mirar el Segundo Sector
            if fastest_driver_2.Sector2Time > fastest_driver_1.Sector2Time:
                # El Piloto 2 ha hecho peor sector 2 que el piloto 1
                delta_s2 = fastest_driver_2.Sector2Time - fastest_driver_1.Sector2Time

                # Mirar el Tercer sector
                if fastest_driver_2.Sector3Time > fastest_driver_1.Sector3Time:
                    # El Piloto 2 ha hecho peor sector 3 que el piloto 1
                    delta_s3 = (
                        fastest_driver_2.Sector3Time - fastest_driver_1.Sector3Time
                    )
                    print(
                        f"Delta {driver_1}-{driver_2}: {format_time(delta)}, S1: -{format_time(delta_s1)}, S2: -{format_time(delta_s2)}, S3: -{format_time(delta_s3)}"
                    )

                else:
                    # El Piloto 2 ha hecho peor sector 3 que el piloto 1
                    delta_s3 = (
                        fastest_driver_1.Sector3Time - fastest_driver_2.Sector3Time
                    )
                    print(
                        f"Delta {driver_1}-{driver_2}: {format_time(delta)}, S1: -{format_time(delta_s1)}, S2: -{format_time(delta_s2)}, S3: {format_time(delta_s3)}"
                    )

            else:
                # El Piloto 2 ha hecho mejor sector 2 que el piloto 1
                delta_s2 = fastest_driver_1.Sector2Time - fastest_driver_2.Sector2Time

                # Mirar el Tercer sector
                if fastest_driver_2.Sector3Time > fastest_driver_1.Sector3Time:
                    # El Piloto 2 ha hecho peor sector 3 que el piloto 1
                    delta_s3 = (
                        fastest_driver_2.Sector3Time - fastest_driver_1.Sector3Time
                    )
                    print(
                        f"Delta {driver_1}-{driver_2}: {format_time(delta)}, S1: {format_time(delta_s1)}, S2: {format_time(delta_s2)}, S3: -{format_time(delta_s3)}"
                    )

                else:
                    # El Piloto 2 ha hecho peor sector 3 que el piloto 1
                    delta_s3 = (
                        fastest_driver_1.Sector3Time - fastest_driver_2.Sector3Time
                    )
                    print(
                        f"Delta {driver_1}-{driver_2}: {format_time(delta)}, S1: {format_time(delta_s1)}, S2: {format_time(delta_s2)}, S3: {format_time(delta_s3)}"
                    )

        else:
            # El Piloto 2 ha hecho mejor sector 1 que el piloto 1
            delta_s1 = fastest_driver_1.Sector1Time - fastest_driver_2.Sector1Time

            # Mirar el Segundo Sector
            if fastest_driver_2.Sector2Time > fastest_driver_1.Sector2Time:
                # El Piloto 2 ha hecho peor sector 2 que el piloto 1
                delta_s2 = fastest_driver_2.Sector2Time - fastest_driver_1.Sector2Time

                # Mirar el Tercer sector
                if fastest_driver_2.Sector3Time > fastest_driver_1.Sector3Time:
                    # El Piloto 2 ha hecho peor sector 3 que el piloto 1
                    delta_s3 = (
                        fastest_driver_2.Sector3Time - fastest_driver_1.Sector3Time
                    )
                    print(
                        f"Delta {driver_1}-{driver_2}: {format_time(delta)}, S1: {format_time(delta_s1)}, S2: -{format_time(delta_s2)}, S3: -{format_time(delta_s3)}"
                    )

                else:
                    # El Piloto 2 ha hecho peor sector 3 que el piloto 1
                    delta_s3 = (
                        fastest_driver_1.Sector3Time - fastest_driver_2.Sector3Time
                    )
                    print(
                        f"Delta {driver_1}-{driver_2}: {format_time(delta)}, S1: {format_time(delta_s1)}, S2: -{format_time(delta_s2)}, S3: {format_time(delta_s3)}"
                    )

            else:
                # El Piloto 2 ha hecho mejor sector 2 que el piloto 1
                delta_s2 = fastest_driver_1.Sector2Time - fastest_driver_2.Sector2Time

                # Mirar el Tercer sector
                if fastest_driver_2.Sector3Time > fastest_driver_1.Sector3Time:
                    # El Piloto 2 ha hecho peor sector 3 que el piloto 1
                    delta_s3 = (
                        fastest_driver_2.Sector3Time - fastest_driver_1.Sector3Time
                    )
                    print(
                        f"Delta {driver_1}-{driver_2}: {format_time(delta)}, S1: {format_time(delta_s1)}, S2: {format_time(delta_s2)}, S3: -{format_time(delta_s3)}"
                    )

                else:
                    # El Piloto 2 ha hecho peor sector 3 que el piloto 1
                    delta_s3 = (
                        fastest_driver_1.Sector3Time - fastest_driver_2.Sector3Time
                    )
                    print(
                        f"Delta {driver_1}-{driver_2}: {format_time(delta)}, S1: {format_time(delta_s1)}, S2: {format_time(delta_s2)}, S3: {format_time(delta_s3)}"
                    )


# Funcion para calcular la aceleracion longitudinal
def calcula_longacc(telemetry):
    # Ordena el DataFrame por tiempo de Session(esto es importante para calcular la derivada correctamente)
    telemetry = telemetry.sort_values(by=["SessionTime"])

    # Calcula la diferencia de tiempo entre las filas adyacentes (diferencia de tiempo) y expresarlo en segundos
    telemetry["TimeDifference"] = telemetry["SessionTime"].diff()
    telemetry["TimeDifference"] = telemetry["TimeDifference"].dt.total_seconds()

    # Calcula la aceleración en X (longitudinal) como la derivada de la velocidad
    # La derivada se calcula dividiendo el cambio en velocidad por el cambio en tiempo
    telemetry["Long Acc"] = telemetry["Speed"].diff() / (
        telemetry["TimeDifference"] * 3.6 * 9.81
    )

    # Nota: El 3.6 es para convertir los Km/h a m/s, ya que Speed está en km/h.
    #       El 9.81 es para convertir a g, ya que 1g = 9.81m/s^2

    # Elimina la columna 'TimeDifference'
    telemetry = telemetry.drop(columns=["TimeDifference"])

    # Devolvemos el dataset
    return telemetry


def calcula_lateralacc(telemetry):
    # Ordena el DataFrame por tiempo de Session(esto es importante para calcular la derivada correctamente)
    telemetry = telemetry.sort_values(by=["SessionTime"])

    # Calcula la diferencia de tiempo entre las filas adyacentes (diferencia de tiempo) y expresarlo en segundos
    telemetry["TimeDifference"] = telemetry["SessionTime"].diff()
    telemetry["TimeDifference"] = telemetry["TimeDifference"].dt.total_seconds()

    # Calculamos la derivada primera en la posicion X y en Y, teniendo en cuenta que estan en dm,
    # lo expresamos a metros dividiendo entre 10.
    telemetry["vx"] = ((telemetry["X"] / 10.0).diff()) / telemetry["TimeDifference"]
    telemetry["vy"] = ((telemetry["Y"] / 10.0).diff()) / telemetry["TimeDifference"]

    # Calculamos la derivada segunda
    telemetry["ax"] = (telemetry["vx"].diff()) / telemetry["TimeDifference"]
    telemetry["ay"] = (telemetry["vy"].diff()) / telemetry["TimeDifference"]

    # Calculamos la curvatura en cada punto
    # telemetry['curvatura'] = (telemetry['vx']*telemetry['ay'] - telemetry['ax']*telemetry['vy'])/(telemetry['vx']*telemetry['vx'] + telemetry['vy']*telemetry['vy'])**1.5

    # Calculamos la aceleracion lateral en g's
    # telemetry['Lat Acc'] = ((telemetry['Speed']/3.6)*(telemetry['Speed']/3.6)*telemetry['curvatura'])/9.81

    telemetry["Lat Acc"] = (
        (telemetry["vx"] * telemetry["ay"] - telemetry["ax"] * telemetry["vy"])
        / (telemetry["vx"] * telemetry["vx"] + telemetry["vy"] * telemetry["vy"]) ** 0.5
    ) / 9.81

    # Elimina la columnas auxiliares
    # telemetry = telemetry.drop(columns=['TimeDifference', 'vx', 'vy', 'ax', 'ay', 'curvatura'])
    telemetry = telemetry.drop(columns=["TimeDifference", "vx", "vy", "ax", "ay"])

    # Devolver el valor
    return telemetry


def clasif_corner_by_type(corner_speed, telemetry):
    # Creamos un df auxiliar reseteando los indices
    aux_df = telemetry.reset_index()

    # Creamos una columna llamada distancia a la curva en valor absoluto
    aux_df["Distancia_curva"] = abs(aux_df["Distance"] - corner_speed)

    # Obtenemos el indice de aux más cercano, correspondiente a la curva
    indice_min = aux_df["Distancia_curva"].idxmin()

    # Buscamos el indice a 50m antes del indice_min
    distance_corner = aux_df.loc[indice_min, "Distance"]
    distance_brake = distance_corner - 50

    # Obtenemos el indice de ese punto de frenada creando una columna
    # de Distance_frenada
    aux_df["Distancia_frenada"] = abs(aux_df["Distance"] - distance_brake)
    indice_min_brake = aux_df["Distancia_frenada"].idxmin()

    # Obtenemos el valor de Brake en ese indice, por delante y por detras.
    brake_point = aux_df.loc[indice_min_brake, "Brake"]
    brake_point_less1 = aux_df.loc[indice_min_brake - 1, "Brake"]
    brake_point_plus1 = aux_df.loc[indice_min_brake + 1, "Brake"]

    # Comprobamos que todos sean True, en base a la condición identificamos
    # el tipo de curva
    if brake_point & brake_point_less1 & brake_point_plus1:
        return "HardBraking"
    else:
        if brake_point | brake_point_less1 | brake_point_plus1:
            return "LittleBraking"
        else:
            return "Not_Braking"


def clasif_corner_by_speed(corner_speed, telemetry):
    # Creamos un df auxiliar reseteando los indices
    aux_df = telemetry.reset_index()

    # Creamos una columna llamada distancia a la curva en valor absoluto
    aux_df["Distancia_curva"] = abs(aux_df["Distance"] - corner_speed)

    # Obtenemos el indice de aux más cercano, correspondiente a la curva
    indice_min = aux_df["Distancia_curva"].idxmin()

    # Guardamos en una lista las velocidades en las posiciones indice_min - 2: indice_min + 2
    # La velocidad que marca la curva será el minimo de esas velocidades
    aux_speed = aux_df.loc[indice_min - 2 : indice_min + 2, "Speed"]
    v_min = min(aux_speed)

    # Una condicional para clasificar las curvas
    if v_min <= 150.0:
        # Curva lenta
        return "Slow"
    elif v_min > 240.0:
        # Curva rapida
        return "Fast"
    elif v_min > 150.0 and v_min <= 240.0:
        # Curva media
        return "Medium"
    else:
        print("Error, la velocidad de la curva es erronea")
        return None


def signo(n):
    if n > 0:
        # Es positivo -> 0
        return 0
    else:
        # Es negativo -> 1
        return 1


def find_chicanes(corners):
    # 1) Busco los indices de las curvas Hard Braking
    mask_hard_braking = corners["Type_Corner"] == "HardBraking"
    index_hard_braking = corners.loc[mask_hard_braking].index

    # Creo una nueva columna 'Is_Chicane' y por defecto todos NO, ahora al analizar
    # los indices, veremos si es Chicane o No
    corners["Is_Chicane"] = "No"

    # Recorremos las curvas de HardBraking a ver si son chicanes
    cont_chicanes = 1
    for index in index_hard_braking:
        # Miramos primero que no corresponda con el indice de la ultima curva
        # en ese caso seguimos mirando
        if index != (corners.shape[0] - 1):
            # 2) Miramos si esa curva y la siguiente son de angulos de signo distinto
            corner_angle = corners.loc[index, "Angle"]
            corner_angle_next = corners.loc[index + 1, "Angle"]

            if signo(corner_angle) != signo(corner_angle_next):
                # Tienen distinto angulo
                # 3) Miramos que estan cercanas
                dist_corners = (
                    corners.loc[index + 1, "Distance"] - corners.loc[index, "Distance"]
                )

                if dist_corners <= 105:
                    # Son cercanas, de distinto angulo y la primera es una frenada fuerte => Chicane
                    text = "Yes_" + str(cont_chicanes)
                    corners.loc[index, "Is_Chicane"] = text
                    corners.loc[index + 1, "Is_Chicane"] = text
                    cont_chicanes += 1


def find_Scurves(corners):
    # Añado una nueva columna que sea la Diff_Distance, diferencia entre curvas
    corners["Diff_Distance"] = 0.0  # La primera sera puesta como 0.0
    corners.loc[1:, "Diff_Distance"] = np.diff(corners["Distance"])

    # Seleccionamos aquellas curvas que no son Chicanes y que no son HardBraking
    # Lo guardamos en un df adicional para resetear los indices
    mask_nochicane = corners["Is_Chicane"] == "No"
    mask_nohardbraking = corners["Type_Corner"] != "HardBraking"
    aux_corners = corners.loc[mask_nochicane & mask_nohardbraking].reset_index()

    # 1) Buscamos los grupo de curvas consecutivas dentro de aux_corners
    groups = []
    aux = []
    for i in range(aux_corners.shape[0] - 1):
        if i == aux_corners.shape[0] - 2:
            # Es la ultima que podemos comparar
            aux.append(aux_corners.loc[i, "Number"])

            if aux_corners.loc[i + 1, "Number"] == aux_corners.loc[i, "Number"] + 1:
                # Son consecuitivos, guardamos al ser la última
                aux.append(aux_corners.loc[i + 1, "Number"])
                groups.append(np.unique(aux).tolist())

            else:
                # No son consecutivas, guardamos igualmente
                groups.append(np.unique(aux).tolist())

        else:
            aux.append(aux_corners.loc[i, "Number"])

            if aux_corners.loc[i + 1, "Number"] == aux_corners.loc[i, "Number"] + 1:
                # Son consecuitivos
                aux.append(aux_corners.loc[i + 1, "Number"])
            else:
                # Guardamos
                groups.append(np.unique(aux).tolist())
                aux = []

    # 2) Filtramos dentro de la lista los grupos de 3 o más curvas
    three_groups = []
    for i in range(len(groups)):
        if len(groups[i]) >= 3:
            three_groups.append(groups[i])

    # 3) Ver si son trios o mas curvas son cercanas, y quedarnos con las mismas
    three_groups_consecutive = []

    for i in range(len(three_groups)):
        consecutive = []
        for j in range(1, len(three_groups[i])):
            mask = aux_corners["Number"] == three_groups[i][j]
            series = aux_corners.loc[mask, "Diff_Distance"]
            if int(series.iloc[0]) <= 250:
                # Son seguidas
                consecutive.append(True)
            else:
                # No son seguidas
                consecutive.append(False)

        # Revisamos si hay 3 o mas curvas cercanas, como se comienza desde el j =1 => sum(consecutive) >=2
        if sum(consecutive) >= 2:
            # Tenemos un grupo de curvas consecutivas cercanas
            three_groups_consecutive.append(three_groups[i])

    # 4) Vamos a crear una nueva columna en corners, 'Is_Scurve'
    corners["Is_Scurve"] = "No"

    ## Recorremos la lista de listas three_groups_consecutive
    cont_Scurve = 0
    for i in range(len(three_groups_consecutive)):
        ### Recorremos las curvas dentro del grupo
        cont_Scurve += 1
        for j in range(len(three_groups_consecutive[i])):
            ### Creamos una mascara para la fila de este numero
            mask = corners["Number"] == three_groups_consecutive[i][j]

            ### Guardamos en esa fila que es una chicane
            text = "Yes_" + str(cont_Scurve)
            corners.loc[mask, "Is_Scurve"] = text

    # Eliminamos la columna Diff_Distance
    corners.pop("Diff_Distance")


def redefine_type_corner(corners):
    # 1) Cambiamos la etiqueta para las curvas Chicane
    mask_chicane = corners["Is_Chicane"] != "No"
    index_chicane = corners.loc[mask_chicane].index

    ## Iteramos los indices de chicane
    for index in index_chicane:
        # Sacamos el nº de chicane
        n_chicane = corners.loc[index, "Is_Chicane"].split("_")[1]

        # Guardamos el Type_Corner como Chicane_numero
        corners.loc[index, "Type_Corner"] = "Chicane_" + n_chicane

    # 2) Cambiamos la etiqueta para las curvas S
    mask_scurve = corners["Is_Scurve"] != "No"
    index_scurve = corners.loc[mask_scurve].index

    ## Iteramos los indices de scurve
    for index in index_scurve:
        # Sacamos el nº de scurve
        n_scurve = corners.loc[index, "Is_Scurve"].split("_")[1]

        # Guardamos el Type_Corner como Scorner_numero
        corners.loc[index, "Type_Corner"] = "Scorner_" + n_scurve

    # 3) Cambiamos la etiqueta a las curvas Not_Braking por Flat_out
    mask_flat = corners["Type_Corner"] == "Not_Braking"
    index_flat = corners.loc[mask_flat].index

    cont = 1
    for index in index_flat:
        text = "Flatout_" + str(cont)
        corners.loc[index, "Type_Corner"] = text
        cont += 1

    # 4) Añadimos el contador de curvas HardBraking
    mask_hard = corners["Type_Corner"] == "HardBraking"
    index_hard = corners.loc[mask_hard].index

    cont = 1
    for index in index_hard:
        text = "HardBraking_" + str(cont)
        corners.loc[index, "Type_Corner"] = text
        cont += 1

    # 5) Añadimos el contador de curvas LittleBraking
    mask_little = corners["Type_Corner"] == "LittleBraking"
    index_little = corners.loc[mask_little].index

    cont = 1
    for index in index_little:
        text = "LittleBraking_" + str(cont)
        corners.loc[index, "Type_Corner"] = text
        cont += 1

    # 6) Quitamos las columnas 'Is_Chicane' y 'Is_Scurve'
    corners.pop("Is_Chicane")
    corners.pop("Is_Scurve")


def create_corner_tableclasif(corners, telemetry):
    # Funcion para clasificar las curvas
    corners["Type_Corner"] = corners["Distance"].apply(
        lambda x: clasif_corner_by_type(x, telemetry)
    )

    # Funcion para clasificar las curvas por su velocidad
    corners["Type_Corner_Speed"] = corners["Distance"].apply(
        lambda x: clasif_corner_by_speed(x, telemetry)
    )

    # Busco que curvas son chicanes
    find_chicanes(corners)

    # Busco las curvas S (3 o más curvas seguidas) y cercanas
    find_Scurves(corners)

    # Ahora de las curvas que nos quedan, tenemos las de HardBraking y las Flatout,
    # Para ello cambiaremos Type_Corner por el tipo, (Las Not_braking que no sean S, por Flatout)
    # Las que sean S por S_curve_numero. Las Chicanes por Chicane_numero (numero se refiere a chicane numero x del circuito)
    redefine_type_corner(corners)

    # Print
    return corners


def paint_maxspeed(
    subset_DRS_i,
    telemetry_driver_1,
    telemetry_driver_2,
    ax,
    color_driver1,
    color_driver2,
):
    # Calcular la distancia de cierre del DRS en base al piloto 1
    distance = subset_DRS_i.loc[subset_DRS_i.index[-1]]
    distance_driver_1 = telemetry_driver_1["Distance"] == distance
    index = telemetry_driver_1.loc[distance_driver_1, "Speed"].index[0]

    # Crear el subset para ambos pilotos un subset con index +- 3
    subset_driver1 = telemetry_driver_1.loc[(index - 3) : (index + 3)]
    subset_driver2 = telemetry_driver_2.loc[(index - 3) : (index + 3)]
    print(
        f"Shapes| subdriver1 = {subset_driver1.shape[0]}, subdriver2 = {subset_driver2.shape[0]}\n"
    )
    print(subset_driver1[["Speed", "DRS", "Throttle"]])
    print(subset_driver2[["Speed", "DRS", "Throttle"]])

    # Mask driver1, con acelarador al tope y drs abierto
    mask_DRSdriver1 = subset_driver1["DRS"] == 12
    mask_Throttledriver1 = subset_driver1["Throttle"] >= 99
    mask_driver1 = mask_DRSdriver1 & mask_Throttledriver1

    # Mask driver2
    mask_DRSdriver2 = subset_driver2["DRS"] == 12
    mask_Throttledriver2 = subset_driver2["Throttle"] >= 99
    mask_driver2 = mask_DRSdriver2 & mask_Throttledriver2

    # Cogemos el ultimo valor de ambos caso
    v_final_driver_1 = subset_driver1.loc[mask_driver1, "Speed"].values[-1]
    print(subset_driver2.loc[mask_driver2, "Speed"])

    v_final_driver_2 = subset_driver2.loc[mask_driver2, "Speed"].values[-1]

    # Añadir el texto de la velocidad, ordenando por el piloto mas rapido
    if v_final_driver_1 > v_final_driver_2:
        ax[0].text(
            distance, v_final_driver_1, str(v_final_driver_1), color=color_driver1
        )
        ax[0].text(
            distance, v_final_driver_2 - 15, str(v_final_driver_2), color=color_driver2
        )
    else:
        ax[0].text(
            distance, v_final_driver_2, str(v_final_driver_2), color=color_driver2
        )
        ax[0].text(
            distance, v_final_driver_1 - 15, str(v_final_driver_1), color=color_driver1
        )


def find_maxspeed_one_driver(subset_DRS_i):
    # Calcular la distancia de cierre del DRS
    distance = subset_DRS_i.loc[subset_DRS_i.index[-1], "Distance"]

    # Cogemos el ultimo valor de ambos caso
    # print(tabulate(subset_DRS_i, headers='keys', showindex=False, tablefmt='grid'))
    mask_throttle = subset_DRS_i["Throttle"] >= 98
    # v_final = subset_DRS_i.loc[subset_DRS_i.index[-1], 'Speed']
    v_final = round(subset_DRS_i.loc[mask_throttle, "Speed"].to_list()[-1], 1)

    return v_final, distance


def find_maxspeed_one_driver_parametrize(
    subset_DRS_i, dist_column, speed_column, throttle_column
):
    # Calcular la distancia de cierre del DRS
    distance = subset_DRS_i.loc[subset_DRS_i.index[-1], dist_column]

    # Cogemos el ultimo valor de ambos caso
    # print(tabulate(subset_DRS_i, headers='keys', showindex=False, tablefmt='grid'))
    mask_throttle = subset_DRS_i[throttle_column] >= 98
    # v_final = subset_DRS_i.loc[subset_DRS_i.index[-1], 'Speed']
    v_final = round(subset_DRS_i.loc[mask_throttle, speed_column].to_list()[-1], 1)

    return v_final, distance


def convert_to_ms(speed):
    return speed / 3.6


def convert_to_kmh(speed):
    return speed * 3.6


def convert_to_timedelta(t):
    return pd.to_timedelta(t, unit="s")


def estimate_unknown_speed_and_time(telemetry, x):
    # 1) DEFINIR EL SUBSET
    # Mascaras antes y despues de esa distancia
    mask_distancia_anterior = telemetry["Distance"] <= x
    mask_distancia_posterior = telemetry["Distance"] >= x

    # Tomar los indices ultimo de distancia_anterior y primero de disntancia_posterior
    index_anterior = telemetry.loc[mask_distancia_anterior,].index.to_list()[-1]
    index_posterior = telemetry.loc[mask_distancia_posterior,].index.to_list()[0]

    # Definir el subset de trabajo
    subset = telemetry.loc[index_anterior:index_posterior,].copy()
    subset = subset.reset_index(drop=True)

    # 2) CALCULOS DE PARAMETROS INICIALES
    # Calcular v0 v1 en m/s
    v0 = convert_to_ms(subset["Speed"].values[0])
    v1 = convert_to_ms(subset["Speed"].values[1])

    # Definir t0 y t1
    t0 = int(subset["Time"].values[0]) / 10**9
    t1 = int(subset["Time"].values[1]) / 10**9

    # Definir x0 y x1
    x0 = subset["Distance"].values[0]
    x1 = subset["Distance"].values[1]

    # Calcular a
    a = (v1 - v0) / (t1 - t0)

    # 3) CALCULOS DE T Y V
    if a != 0.0:
        # Calcular ts
        t_mas = round((-v0 + (v0 * v0 + 2 * a * (x - x0)) ** 0.5) / a + t0, 3)
        t_menos = round(t0 - (v0 + (v0 * v0 + 2 * a * (x - x0)) ** 0.5) / a, 3)

        # TODO Verificar el valor con t0 y t1
        # print(f"t_menos={t_menos}, t0={t0}, t1={t1}")
        # print(f"t_mas={t_mas}, t0={t0}, t1={t1}\n")
        if (t_menos >= t0) & (t_menos <= t1):
            t = t_menos
        elif (t_mas >= t0) & (t_mas <= t1):
            t = t_mas
        else:
            t = t1

        # Calcular v
        v = v0 + a * (t - t0)

    else:
        # Velocidad
        v = v0

        # Tiempo
        t = (x - x0 + v0 * t0) / v0

    # 4) CONVERTIR A LAS UNIDADES
    v = convert_to_kmh(v)
    t = convert_to_timedelta(t)

    return v, t


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
        return delta
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


def get_init_and_finish_sesion(session):
    # Calcular el listado de objetos para delimitar las sesiones
    df_session_status = session.session_status
    finished_list = df_session_status.loc[
        df_session_status["Status"] == "Finished", "Time"
    ].to_list()
    mask_status_started = df_session_status["Status"] == "Started"
    started_list = []
    for i in range(len(finished_list)):
        if i == 0:
            mask_init = df_session_status["Time"] >= df_session_status["Time"].iloc[0]
            mask_finish = df_session_status["Time"] <= finished_list[i]
        else:
            mask_init = df_session_status["Time"] >= finished_list[i - 1]
            mask_finish = df_session_status["Time"] <= finished_list[i]

        value = df_session_status.loc[
            mask_init & mask_finish & mask_status_started, "Time"
        ].to_list()[0]
        started_list.append(value)
    return started_list, finished_list


def generate_sessionTime(time, started_sesions, finished_sesions):
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


def get_best_time(subset):
    # Find his last sesion
    mask_Q3 = subset["SessionTime"] == "Q3"
    mask_Q2 = subset["SessionTime"] == "Q2"
    mask_Q1 = subset["SessionTime"] == "Q1"
    mask_NA = subset["LapTime"].isna()

    if sum(mask_Q3) == 0:
        # El tiempo oficial no es en Q3
        if sum(mask_Q2) == 0:
            # El tiempo oficial no es en Q2 es en Q1
            subset = subset.loc[~mask_NA,]
            mask_Q1 = subset["SessionTime"] == "Q1"
            subset_q1 = subset.loc[mask_Q1,]
            if subset_q1.shape[0] == 0:
                official_time = "NoTime"
            else:
                official_time = min(subset_q1["LapTime"])

            return official_time
        else:
            subset = subset.loc[~mask_NA,]
            mask_Q2 = subset["SessionTime"] == "Q2"
            subset_q2 = subset.loc[mask_Q2,]

            if subset_q2.shape[0] == 0:
                official_time = "NoTime"
            else:
                official_time = min(subset_q2["LapTime"])

            return official_time
    else:
        subset = subset.loc[~mask_NA,]
        mask_Q3 = subset["SessionTime"] == "Q3"
        subset_q3 = subset.loc[mask_Q3,]

        if subset_q3.shape[0] == 0:
            official_time = "NoTime"
        else:
            official_time = min(subset_q3["LapTime"])

        return official_time


# def get_best_time(subset):
#     # Find his last sesion
#     mask_Q3 = subset['SessionTime'] == 'Q3'
#     mask_Q2 = subset['SessionTime'] == 'Q2'
#     mask_Q1 = subset['SessionTime'] == 'Q1'
#     mask_NA = subset['LapTime'].isna()

#     if sum(mask_Q3) == 0:
#         # El tiempo oficial no es en Q3
#         if sum(mask_Q2) == 0:
#             # El tiempo oficial no es en Q2 es en Q1
#             subset = subset.loc[~mask_NA,]
#             mask_Q1 = subset['SessionTime'] == 'Q1'
#             subset_q1 = subset.loc[mask_Q1,]
#             if subset_q1.shape[0] == 0:
#                 official_time = 'NoTime'
#             else:
#                 official_time = min(subset_q1['LapTime'])

#             return official_time
#         else:
#             subset = subset.loc[~mask_NA,]
#             mask_Q2 = subset['SessionTime'] == 'Q2'
#             subset_q2 = subset.loc[mask_Q2,]

#             if subset_q2.shape[0] == 0:
#                 official_time = 'NoTime'
#             else:
#                 official_time = min(subset_q2['LapTime'])

#             return official_time
#     else:
#         subset = subset.loc[~mask_NA,]
#         mask_Q3 = subset['SessionTime'] == 'Q3'
#         subset_q3 = subset.loc[mask_Q3,]

#         if subset_q3.shape[0] == 0:
#             official_time = 'NoTime'
#         else:
#             official_time = min(subset_q3['LapTime'])

#         return official_time


def get_best_time_practice(subset):
    # Find his last sesion
    mask_NA = subset["LapTime"].isna()

    subset = subset.loc[~mask_NA,]
    if subset.shape[0] == 0:
        official_time = "NoTime"
    else:
        official_time = min(subset["LapTime"])

    return official_time


def get_best_time_of_session(df, session):
    # Extract the subset of not NA times in the desire session
    mask_NA = df["LapTime"].isna()
    subset = df.loc[~mask_NA,]
    mask_session = subset["SessionTime"] == session

    subset_session = subset.loc[mask_session,]

    if subset_session.shape[0] == 0:
        mask_session = subset["SessionTime"] == max(subset["SessionTime"])
        subset_session = subset.loc[mask_session,]
        official_time = min(subset_session["LapTime"])
    else:
        official_time = min(subset_session["LapTime"])

    return official_time


def get_lap_number_of_session(session, subset, desire_session):
    # Añadir Session of the lap (Q1, Q2, Q3)
    started_sesions, finished_sesions = get_init_and_finish_sesion(session)
    subset["SessionTime"] = subset["Time"].apply(
        lambda time: generate_sessionTime(time, started_sesions, finished_sesions)
    )
    # CHECK THE LAPTIMES OF THE DRIVER
    best_lap = get_best_time_of_session(subset, desire_session)

    # Extract lapnumber
    lap_number = subset.loc[subset["LapTime"] == best_lap, "LapNumber"].iloc[0]

    return lap_number


def get_lap_number(session, subset):
    if session.session_info["Type"] == "Qualifying":
        # Añadir Session of the lap (Q1, Q2, Q3)
        started_sesions, finished_sesions = get_init_and_finish_sesion(session)
        subset["SessionTime"] = subset["Time"].apply(
            lambda time: generate_sessionTime(time, started_sesions, finished_sesions)
        )

        # CHHECK THE LAPSTIMES OF THE DRIVER
        best_lap = get_best_time(subset)

        # Extract lapnumber
        lap_number = subset.loc[subset["LapTime"] == best_lap, "LapNumber"].iloc[0]

        return lap_number
    elif session.session_info["Type"] == "Practice":
        # CHHECK THE LAPSTIMES OF THE DRIVER
        best_lap = get_best_time_practice(subset)

        # Extract lapnumber
        lap_number = subset.loc[subset["LapTime"] == best_lap, "LapNumber"].iloc[0]

        return lap_number


def pass_to_finaltime(finaltime):
    # Obtain final_time round
    final_diff_round = round(finaltime, 3)

    # Pass to time
    if final_diff_round >= 60.0:
        minutes = int(final_diff_round // 60.0)
        seconds = final_diff_round - 60.0
        seconds = round(seconds, 3)
        if seconds < 10.0:
            return f"+{minutes}:0{seconds}"
        else:
            return f"+{minutes}:{seconds}"
    else:
        return f"+{final_diff_round}"


def pass_totaltime(finaltime):
    # Obtain final_time round
    final_diff_round = round(finaltime, 3)
    print(final_diff_round)


def generate_correct_laptimes(laps_driver, session):
    # Checkear que no haya vueltas nat
    list_corrected_laps = []
    if sum(pd.isna(laps_driver["LapTime"])) > 0:
        # Hay vueltas Na
        for i in range(laps_driver.shape[0]):
            lap = laps_driver["LapTime"].iloc[i]
            if pd.isna(lap) and i != laps_driver.shape[0] - 1:
                # Vuelta NaT, corregir como la diferencia entre los starttime consecutivos i+1-i
                start_time_i = laps_driver["LapStartTime"].iloc[i]
                start_time_next = laps_driver["LapStartTime"].iloc[i + 1]
                real_time_lap = start_time_next - start_time_i
                list_corrected_laps.append(real_time_lap)
            elif pd.isna(lap) and i == laps_driver.shape[0] - 1:
                df_session_status = session.session_status
                start_time_next = df_session_status.loc[
                    df_session_status["Status"] == "Finished", "Time"
                ].to_list()[-1]

                # Vuelta NaT, corregir como la diferencia entre los starttime consecutivos i+1-i
                start_time_i = laps_driver["LapStartTime"].iloc[i]
                real_time_lap = start_time_next - start_time_i
                list_corrected_laps.append(real_time_lap)
            else:
                list_corrected_laps.append(lap)
    else:
        list_corrected_laps = laps_driver["LapTime"].to_list()
    return list_corrected_laps


def get_df_penalties(event):
    # --------------------- VARIABLES --------------------------------------------
    INIT_PHRASE = "FIA STEWARDS: "
    SERVED_PHRASE = "PENALTY SERVED"

    # --------------------- DF PENALTIES -----------------------------------------
    ## Df penalty creation
    df_race_control = event.race_control_messages
    mask_penalities = (
        df_race_control["Message"]
        .str.split("-", n=1)
        .str[0]
        .str.contains("PENALTY", na=False)
    )  # En la primera division con -
    df_penalties = df_race_control.loc[mask_penalities, ["Time", "Lap", "Message"]]

    ## Generate auxiliar columns
    df_penalties["BaseMessage"] = df_penalties["Message"].apply(
        lambda i: i[len(INIT_PHRASE) :] if i.startswith(INIT_PHRASE) else i
    )
    df_penalties["SplittedMessage"] = df_penalties["BaseMessage"].apply(
        lambda i: i.split("-")
    )
    df_penalties["SplittedMessage"] = df_penalties["SplittedMessage"].apply(
        lambda lista: [s.lstrip() for s in lista]
    )
    df_penalties["PenaltyReassons"] = df_penalties["SplittedMessage"].apply(
        lambda i: i[2:] if i[0].startswith(SERVED_PHRASE) else i[1:]
    )
    df_penalties["ImportantMessageSplitted"] = df_penalties["SplittedMessage"].apply(
        lambda i: (
            i[0].split(" ") + i[1].split(" ")
            if i[0].startswith(SERVED_PHRASE)
            else i[0].split(" ")
        )
    )

    ## Structure data
    penalty_types = []
    drivers = []
    drivers_numbers = []
    quantity_penalties = []
    units_penalties = []

    for i in range(df_penalties.shape[0]):
        aux_ImportantMessageSplitted = df_penalties["ImportantMessageSplitted"].iloc[i]
        aux_SplittedMessage = df_penalties["SplittedMessage"].iloc[i]
        # PenaltyType, DriverNumber
        if " ".join(aux_ImportantMessageSplitted[0:2]) == SERVED_PHRASE:
            penalty_types.append(SERVED_PHRASE)
            drivers_numbers.append(
                aux_ImportantMessageSplitted[
                    aux_ImportantMessageSplitted.index("CAR") + 1
                ]
            )
        else:
            penalty_types.append(
                " ".join(
                    aux_ImportantMessageSplitted[
                        (aux_ImportantMessageSplitted.index("PENALTY") - 1) : (
                            aux_ImportantMessageSplitted.index("PENALTY") + 1
                        )
                    ]
                )
            )
            drivers_numbers.append(
                aux_ImportantMessageSplitted[
                    aux_ImportantMessageSplitted.index("CAR") + 1
                ]
            )

        # Driver, QuantityPenalty, UnitPenalty
        if aux_SplittedMessage[0].startswith(SERVED_PHRASE):
            message = aux_SplittedMessage[1].split(" ")
            drivers.append(message[message.index("CAR") + 2][1:-1])
            quantity_penalties.append(message[0])
            units_penalties.append(message[1])
        else:
            message = aux_SplittedMessage[0].split(" ")
            drivers.append(message[message.index("CAR") + 2][1:-1])
            quantity_penalties.append(message[0])
            units_penalties.append(message[1])

    ## Crear columnas
    df_penalties["PenaltyType"] = penalty_types
    df_penalties["QuantityPenalty"] = quantity_penalties
    df_penalties["UnitPenalty"] = units_penalties
    df_penalties["Driver"] = drivers
    df_penalties["DriverNumber"] = drivers_numbers

    ## Convert types
    df_penalties["DriverNumber"] = df_penalties["DriverNumber"].astype(int)
    df_penalties["QuantityPenalty"] = df_penalties["QuantityPenalty"].apply(
        lambda i: int(i) if i != "DRIVE" else i
    )

    ## Select the important Columns
    df_penalties = df_penalties[
        [
            "Lap",
            "PenaltyType",
            "QuantityPenalty",
            "UnitPenalty",
            "Driver",
            "DriverNumber",
            "PenaltyReassons",
        ]
    ]
    ## Drivers case of PENALTY and PENALTY SERVED but PENALTY SERVED without PENALTY REASSONS
    unique_drivers_list = np.unique(df_penalties["Driver"])
    for driver in unique_drivers_list:
        mask_driver = df_penalties["Driver"] == driver
        if sum(mask_driver) == 2:
            subset = df_penalties[mask_driver]
            mask_penalty_served = subset["PenaltyType"] == "PENALTY SERVED"
            if sum(mask_penalty_served) == 1:
                var = subset.loc[mask_penalty_served, "PenaltyReassons"].iloc[0]
                if isinstance(var, list) and not var:
                    # Fill with the last penalty reassons
                    penalty_reasssons = []
                    penalty_reasssons.append(
                        subset.loc[~mask_penalty_served, "PenaltyReassons"].iloc[0]
                    )
                    print(f"driver={driver} penalty_reasssons={penalty_reasssons}")
                    df_penalties.loc[
                        mask_driver & mask_penalty_served, "PenaltyReassons"
                    ] = penalty_reasssons

    return df_penalties


def get_df_pending_penalties(df_penalties):
    # Aseguramos que 'PenaltyReassons' sea string para ambas tablas
    df_penalties["PenaltyReassonsStr"] = df_penalties["PenaltyReassons"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else str(x)
    )

    # Filtramos cada tipo de penalización
    time_penalties = df_penalties[df_penalties["PenaltyType"] == "TIME PENALTY"].copy()
    served_penalties = df_penalties[
        df_penalties["PenaltyType"] == "PENALTY SERVED"
    ].copy()

    # Usamos la nueva columna string como clave de unión
    merged = pd.merge(
        time_penalties,
        served_penalties,
        on=["DriverNumber", "QuantityPenalty", "UnitPenalty", "PenaltyReassonsStr"],
        # on=['DriverNumber', 'QuantityPenalty', 'UnitPenalty'],
        how="left",
        indicator=True,
        suffixes=("", "_served"),
    )

    # Filter not served and filter columns
    pending_penalties = merged[merged["_merge"] == "left_only"]
    pending_penalties = pending_penalties[time_penalties.columns]
    pending_penalties.pop("PenaltyReassonsStr")
    df_penalties.pop("PenaltyReassonsStr")

    # Check
    return pending_penalties


def get_final_global_table_qualyfying(session):
    try:
        # 1) Get Grid Table and change position to int
        results = session.results[
            ["Position", "Abbreviation", "FullName", "TeamName", "CountryCode"]
        ]
        list_abbreviation = results["Abbreviation"].to_list()
        list_nacionalidad = [
            DICT_PILOTOS_NACIONALIDAD[list_abbreviation[i]]
            if nacionalidad == ""
            else nacionalidad
            for i, nacionalidad in enumerate(results["CountryCode"].to_list())
        ]
        list_teams = results["TeamName"].to_list()
        list_pilotos = results["FullName"].to_list()

        list_position = [i + 1 for i in range(len(list_abbreviation))]
        # 2) Get Laps
        laps = session.laps
        # mask_personalBest = laps['IsPersonalBest'] == True
        mask_notdeleted = laps["Deleted"] == False
        new_laps = laps.loc[mask_notdeleted,]
        if new_laps.empty:
            mask_notdeleted = laps["Deleted"] != None
            new_laps = laps.loc[mask_notdeleted,]

        # 3) Add Session of the lap (Q1, Q2, Q3)
        started_sesions, finished_sesions = get_init_and_finish_sesion(session)
        new_laps["SessionTime"] = new_laps["Time"].apply(
            lambda time: generate_sessionTime(time, started_sesions, finished_sesions)
        )
        # 4) Get the laps
        list_laps = []
        for i, driver in enumerate(list_abbreviation):
            mask_driver = new_laps["Driver"] == driver
            subset_driver = new_laps.loc[mask_driver,]

            if sum(subset_driver["LapTime"].isna()) == len(subset_driver["LapTime"]):
                best_lap = "NoTime"
            else:
                best_lap = get_best_time(subset_driver)

                if best_lap != "NoTime":
                    best_lap = format_time(best_lap)

            list_laps.append(best_lap)

        # 5) Create df
        # 5.1 Traducir equipos a nombre normal
        list_teams = [DICT_TEAMS[team] for team in list_teams]

        # 5.2 Cambiar los apellidos a mayusculas
        list_final_pilotos = []
        for piloto in list_pilotos:
            nombre = piloto.split(" ")[0]
            apellido = piloto.split(" ")[1].upper()
            list_final_pilotos.append(nombre + " " + apellido)

        d = {
            "posicion": list_position,
            "nacionalidad": list_nacionalidad,
            "Driver": list_abbreviation,
            "piloto": list_final_pilotos,
            "equipo": list_teams,
            "LapTime": list_laps,
        }
        df_results = pd.DataFrame(d)

        # 6) Create deltas
        df_results = add_gap(df_results)
        return df_results
    except Exception as e:
        raise e


def get_final_global_table_practice(session):
    try:
        # 1) Get Grid Table and change position to int
        results = session.results[
            ["Position", "Abbreviation", "FullName", "TeamName", "CountryCode"]
        ]
        list_abbreviation = results["Abbreviation"].to_list()
        list_nacionalidad = [
            DICT_PILOTOS_NACIONALIDAD[list_abbreviation[i]]
            if nacionalidad == ""
            else nacionalidad
            for i, nacionalidad in enumerate(results["CountryCode"].to_list())
        ]
        list_teams = results["TeamName"].to_list()
        list_pilotos = results["FullName"].to_list()
        list_position = [i + 1 for i in range(len(list_abbreviation))]

        # 2) Get Laps
        laps = session.laps
        # mask_personalBest = laps['IsPersonalBest'] == True
        mask_notdeleted = laps["Deleted"] == False
        new_laps = laps.loc[mask_notdeleted,]

        # 3) Get the laps
        list_laps = []
        for i, driver in enumerate(list_abbreviation):
            mask_driver = new_laps["Driver"] == driver
            subset_driver = new_laps.loc[mask_driver,]

            if sum(subset_driver["LapTime"].isna()) == len(subset_driver["LapTime"]):
                best_lap = "NoTime"
            else:
                best_lap = get_best_time_practice(subset_driver)

                if best_lap != "NoTime":
                    best_lap = format_time(best_lap)

            list_laps.append(best_lap)

        # 4) Create df
        # 4.1 Traducir equipos a nombre normal
        list_teams = [DICT_TEAMS[team] for team in list_teams]

        # 4.2 Cambiar los apellidos a mayusculas
        list_final_pilotos = []
        for piloto in list_pilotos:
            nombre = piloto.split(" ")[0]
            apellido = piloto.split(" ")[1].upper()
            list_final_pilotos.append(nombre + " " + apellido)

        d = {
            "posicion": list_position,
            "nacionalidad": list_nacionalidad,
            "Driver": list_abbreviation,
            "piloto": list_final_pilotos,
            "equipo": list_teams,
            "LapTime": list_laps,
        }
        df_results = pd.DataFrame(d)
        df_results = df_results.sort_values(by="LapTime", ascending=True)

        # 6) Create deltas
        df_results = add_gap(df_results)
        return df_results
    except Exception as e:
        raise e


def format_td(td):
    try:
        total_seconds = td.total_seconds()
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:06.3f}"
    except ValueError:
        return td
    except AttributeError:
        return td


def format_timedelta(td):
    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def format_timedelta_microseconds(td):
    total_seconds = td.total_seconds()
    seconds = total_seconds % 60
    return f"{seconds:0.3f}"


def format_columns(df):
    cols = df.columns.to_list()
    for col in cols:
        df[col] = df[col].apply(lambda i: format_td(i))


def format_column(df, col):
    df[col] = df[col].apply(lambda i: format_td(i))


def format_column_datetime(df, col):
    if sum(df[col] != "NaN") > 0:
        df[col] = df[col].apply(lambda i: format_timedelta(i))


def format_column_microseconds(df, col):
    df[col] = df[col].apply(lambda i: format_timedelta_microseconds(i))


def get_tiempo_decorte(df, column):
    if column == "Q1":
        df = df.sort_values(by=column, ascending=True)
        time = df[column].to_list()[14]
        return time
    elif column == "Q2":
        df = df.sort_values(by=column, ascending=True)
        time = df[column].to_list()[9]
        return time
    elif column == "Q3":
        return None
    else:
        raise Exception(f"Invalid {column}")


def get_tiempo_decorte_column(df):
    Q_cols = ["Q3", "Q2", "Q1"]
    values_time = []
    for col in Q_cols:
        values_time.append(get_tiempo_decorte(df, col))
    return values_time


def fastest_lap_session_info(results, session, Qsession):
    fastest_lap = results.loc[results[Qsession] == results[Qsession].min(),]
    fastest_lap = fastest_lap[["Abbreviation", "TeamName", Qsession]]
    driver = fastest_lap["Abbreviation"].iloc[0]
    laptime = fastest_lap[Qsession].iloc[0]
    driver_laps = session.laps.pick_drivers(driver)
    time_of_lap = driver_laps.loc[
        driver_laps["LapTime"] == laptime, "LapStartTime"
    ].iloc[0]
    compound = driver_laps.loc[driver_laps["LapTime"] == laptime, "Compound"].iloc[0]

    # Fill basic columns
    fastest_lap["Compound"] = compound
    fastest_lap["Time"] = time_of_lap
    fastest_lap["LapTime"] = laptime
    fastest_lap["Driver"] = driver

    # Fill weather columns
    try:
        df_weather = session.weather_data
        weather_lap = df_weather[
            (df_weather["Time"] >= time_of_lap)
            & (df_weather["Time"] <= time_of_lap + laptime)
        ]
        weather_lap["WeatherTime"] = weather_lap["Time"]
        weather_lap.pop("Time")
        for col in weather_lap.columns.to_list():
            fastest_lap[col] = weather_lap[col].iloc[0]
    except DataNotLoadedError:
        for col in WEATHER_COLUMNS:
            fastest_lap[col] = "NaN"

    fastest_lap = fastest_lap[
        ["LapTime", "Driver", "Time", "TeamName"] + WEATHER_COLUMNS
    ]
    return fastest_lap


def add_lap_info(lap, session):
    # Extract info of the lap
    driver = lap["Driver"]
    laptime = lap["LapTime"]
    team = lap["Team"]

    driver_laps = session.laps.pick_drivers(driver)
    time_of_lap = driver_laps.loc[
        driver_laps["LapTime"] == laptime, "LapStartTime"
    ].iloc[0]
    compound = driver_laps.loc[driver_laps["LapTime"] == laptime, "Compound"].iloc[0]

    # Save new columns
    lap_more_data = {}
    lap_more_data["Compound"] = compound
    lap_more_data["Time"] = time_of_lap
    lap_more_data["LapTime"] = laptime
    lap_more_data["Driver"] = driver
    lap_more_data["TeamName"] = team

    # Weather extraction
    try:
        df_weather = session.weather_data
        weather_lap = df_weather[
            (df_weather["Time"] >= time_of_lap)
            & (df_weather["Time"] <= time_of_lap + laptime)
        ]
        weather_lap["WeatherTime"] = weather_lap["Time"]
        weather_lap.pop("Time")

        # Save weather columns
        for col in weather_lap.columns.to_list():
            lap_more_data[col] = weather_lap[col].iloc[0]
    except DataNotLoadedError:
        # Save weather columns
        for col in WEATHER_COLUMNS:
            lap_more_data[col] = "NaN"

    # Select only this columns
    df_lap = pd.DataFrame(lap_more_data, index=[0])
    df_lap = df_lap[["LapTime", "Driver", "Time", "TeamName"] + WEATHER_COLUMNS]
    return df_lap


def convert_to_df_to_save(df_Q1):
    df_results = df_Q1.copy()
    df_results["LapTime"] = df_results["Q1"].fillna("NoTime")
    df_results["Driver"] = df_results["Abbreviation"]
    df_results.sort_values(by="LapTime", ascending=True, inplace=True)
    df_results.reset_index(inplace=True)

    # 6) Create deltas
    df_results = add_gap(df_results)
    return df_results


def show_resumes_table_qualyfying(session, use_Q1_data=False):
    results = session.results

    # Resume of Q3, Q2, Q1
    df_resume = results[["Position", "Abbreviation", "Q3", "Q2", "Q1"]]
    list_abbreviation = df_resume["Abbreviation"].to_list()
    df_resume["Nacionality"] = [
        DICT_PILOTOS_NACIONALIDAD[list_abbreviation[i]]
        if nacionalidad == ""
        else nacionalidad
        for i, nacionalidad in enumerate(results["CountryCode"].to_list())
    ]
    df_resume = df_resume[["Position", "Nacionality", "Abbreviation", "Q3", "Q2", "Q1"]]

    # Track stats improvements
    Q_cols = ["Q1", "Q2", "Q3"]
    Q_cols_ordered = ["Q3", "Q2", "Q1"]
    statistical_cols = ["min", "max", "mean", "std"]
    df_stats = df_resume[Q_cols].agg(statistical_cols)
    df_stats = df_stats.T.loc[Q_cols_ordered]
    df_stats["TiempoDeCorte"] = get_tiempo_decorte_column(df_resume)
    df_stats["TrackImprove"] = df_stats["mean"].apply(lambda time: time.total_seconds())
    df_stats["TrackImprove"] = df_stats["TrackImprove"].diff(-1)
    df_stats["TrackImprove"] = df_stats["TrackImprove"].apply(lambda i: round(i, 3))

    # Fastest Lap info
    fastest_lap = results.loc[results["Q3"] == results["Q3"].min(),]
    fastest_lap = fastest_lap[["Abbreviation", "TeamName", "Q3"]]
    driver = fastest_lap["Abbreviation"].iloc[0]
    laptime = fastest_lap["Q3"].iloc[0]
    driver_laps = session.laps.pick_drivers(driver)
    time_of_lap = driver_laps.loc[
        driver_laps["LapTime"] == laptime, "LapStartTime"
    ].iloc[0]
    compound = driver_laps.loc[driver_laps["LapTime"] == laptime, "Compound"].iloc[0]

    # Fill important info
    fastest_lap["Compound"] = compound
    fastest_lap["Time"] = time_of_lap
    fastest_lap["LapTime"] = laptime
    fastest_lap["Driver"] = driver

    # Fill weather columns
    try:
        df_weather = session.weather_data
        weather_lap = df_weather[
            (df_weather["Time"] >= time_of_lap)
            & (df_weather["Time"] <= time_of_lap + laptime)
        ]
        weather_lap["WeatherTime"] = weather_lap["Time"]
        weather_lap.pop("Time")
        for col in weather_lap.columns.to_list():
            fastest_lap[col] = weather_lap[col].iloc[0]
    except DataNotLoadedError:
        for col in WEATHER_COLUMNS:
            fastest_lap[col] = "NaN"

    fastest_lap = fastest_lap[
        ["LapTime", "Driver", "Time", "TeamName"] + WEATHER_COLUMNS
    ]

    # SHOW
    ## Resume
    print("\nResumen de la Clasificación:")
    format_column(df_resume, "Q1")
    format_column(df_resume, "Q2")
    format_column(df_resume, "Q3")
    print(tabulate(df_resume, headers="keys", showindex=False, tablefmt="grid"))

    if use_Q1_data:
        # Convert to df to save
        df_results = convert_to_df_to_save(df_resume)

        # Save
        race_num = session.session_info["Meeting"]["Number"]
        country_code = session.session_info["Meeting"]["Country"]["Code"]
        save_time_gap(df_results, race_num, country_code, use_Q1_data)
        print("Saved Time Gap with Q1 data")

    ## Stats
    print("\nEstadisticas de la Clasificación:")
    format_column_microseconds(df_stats, "std")
    format_columns(df_stats)
    print(tabulate(df_stats, headers="keys", showindex=True, tablefmt="grid"))

    ## FastestLap Q1
    print("\nDatos de la FastestLap Q1:")
    fastest_lap_Q1 = fastest_lap_session_info(results, session, "Q1")
    format_column(fastest_lap_Q1, "LapTime")
    format_column_datetime(fastest_lap_Q1, "Time")
    format_column_datetime(fastest_lap_Q1, "WeatherTime")
    print(tabulate(fastest_lap_Q1, headers="keys", showindex=False, tablefmt="grid"))

    ## FastestLap Q2
    print("\nDatos de la FastestLap Q2:")
    fastest_lap_Q2 = fastest_lap_session_info(results, session, "Q2")
    format_column(fastest_lap_Q2, "LapTime")
    format_column_datetime(fastest_lap_Q2, "Time")
    format_column_datetime(fastest_lap_Q2, "WeatherTime")
    print(tabulate(fastest_lap_Q2, headers="keys", showindex=False, tablefmt="grid"))

    ## FastestLap Q3 / Pole
    print("\nDatos de la Pole:")
    fastest_lap_Q3 = fastest_lap_session_info(results, session, "Q3")
    format_column(fastest_lap_Q3, "LapTime")
    format_column_datetime(fastest_lap_Q3, "Time")
    format_column_datetime(fastest_lap_Q3, "WeatherTime")
    print(tabulate(fastest_lap_Q3, headers="keys", showindex=False, tablefmt="grid"))


def get_wind_med_direction_sesion(session):
    df_weather = session.weather_data
    val_min = min(df_weather["WindDirection"])
    val_max = max(df_weather["WindDirection"])
    return (val_min + val_max) / 2


def show_meteo_resume(meteo_list):
    cols = []
    col_Q1 = []
    col_Q2 = []
    col_Q3 = []
    for meteo in meteo_list:
        cols.append(f"{meteo['name']}\n{meteo['name']}med")
        col_Q1.append(
            f"{meteo['Q1min']}-{meteo['Q1max']}{meteo['units']}\n{round(meteo['Q1med'], 1)}{meteo['units']}"
        )
        col_Q2.append(
            f"{meteo['Q2min']}-{meteo['Q2max']}{meteo['units']}\n{round(meteo['Q2med'], 1)}{meteo['units']}"
        )
        col_Q3.append(
            f"{meteo['Q3min']}-{meteo['Q3max']}{meteo['units']}\n{round(meteo['Q3med'], 1)}{meteo['units']}"
        )

    d = {"Variables": cols, "Q3": col_Q3, "Q2": col_Q2, "Q1": col_Q1}
    df_meteo = pd.DataFrame(d)
    print(tabulate(df_meteo, headers="keys", showindex=False, tablefmt="grid"))


def save_time_gap(
    df: pd.DataFrame, race_num: int, country_code: str, use_Q1_data=False
):
    # Load Table
    if not use_Q1_data:
        file_path = os.path.join(os.getcwd(), "src", "outputs", "gap_pilotos.csv")
    else:
        file_path = os.path.join(os.getcwd(), "src", "outputs", "gap_pilotos_Q1.csv")

    if os.path.exists(file_path):
        try:
            df_gap = pd.read_csv(file_path, sep=",")

            if sum(df_gap["Carrera"] == race_num) > 0:
                # Repeated no añadir
                pass
            else:
                # New row
                new_row = {}
                new_row["Carrera"] = race_num
                new_row["Country"] = country_code

                for i in range(df.shape[0]):
                    driver = df["Driver"].iloc[i]
                    gap = df["Gap (%)"].iloc[i]

                    # Actualizar
                    new_row[driver] = gap

                df_gap.loc[len(df_gap)] = new_row
        except pd.errors.EmptyDataError:
            df_gap = pd.DataFrame({"Carrera": [race_num], "Country": [country_code]})

            for driver, gap in zip(df["Driver"], df["Gap (%)"]):
                df_gap[driver] = gap
    else:
        df_gap = pd.DataFrame({"Carrera": [race_num], "Country": [country_code]})

        for driver, gap in zip(df["Driver"], df["Gap (%)"]):
            df_gap[driver] = gap

    # Save table
    df_gap.sort_values(by="Carrera", ascending=True, inplace=True)
    df_gap.to_csv(file_path, sep=",", index=False)


def get_final_sesion_table(session, save_gap=False, use_Q1_data=False):
    # ----------------------------------------------- QUALY ---------------------------------------------------------------------------------------
    print(session.session_info["Type"])
    if session.session_info["Type"] == "Qualifying":
        # DF results
        df_results = get_final_global_table_qualyfying(session)

        # Save gap
        if save_gap and not use_Q1_data:
            race_num = session.session_info["Meeting"]["Number"]
            country_code = session.session_info["Meeting"]["Country"]["Code"]
            save_time_gap(df_results, race_num, country_code)
            print("Saved Time Gap")
        elif save_gap and use_Q1_data:
            show_resumes_table_qualyfying(session, use_Q1_data)

        # DF resume
        show_resumes_table_qualyfying(session)

    # ----------------------------------------------- QUALY ---------------------------------------------------------------------------------------
    elif session.session_info["Type"] == "Practice":
        df_results = get_final_global_table_practice(session)

    # ----------------------------------------------- RACE ---------------------------------------------------------------------------------------
    elif session.session_info["Type"] == "Race":
        try:
            # Get Data
            results = session.results[
                [
                    "Position",
                    "Abbreviation",
                    "FullName",
                    "Status",
                    "TeamName",
                    "CountryCode",
                    "Points",
                    "GridPosition",
                ]
            ]
            """
            position = [1.0, 3.0, 2.0, 4.0, 18.0, 17.0, 11.0, 6.0, 16.0, 7.0, 5.0, 15.0, 20.0, 9.0, 10.0, 19.0, 8.0, 13.0, 14.0, 12.0]
            points = [25.0, 15.0, 18.0, 12.0, 0.0, 0.0, 0.0, 8.0, 0.0, 6.0, 10.0, 0.0, 0.0, 2.0, 1.0, 0.0, 4.0, 0.0, 0.0, 0.0]
            results['Position'] = position
            results['Points'] = points
            """

            # Get df_penalties and df_pending_penalties
            df_penalties = get_df_penalties(session)
            df_pending_penalties = get_df_pending_penalties(df_penalties)

            # Check if Points is empty
            if results["Points"].isna().all():
                print(
                    "\nLa columna 'Points' es completamente NaN. Se cancela el paso por la función."
                )
                columnas_todas_nan = results.columns[results.isna().all()]
                print("Las Columnas completamente NaN son:", list(columnas_todas_nan))
                print(
                    tabulate(results, headers="keys", showindex=False, tablefmt="grid")
                )
                return None
            else:
                list_abbreviation = results["Abbreviation"].to_list()
                list_position = results["Position"].to_list()
                list_points = results["Points"].to_list()
                list_teams = results["TeamName"].to_list()
                list_pilotos = results["FullName"].to_list()
                list_nacionalidad = [
                    DICT_PILOTOS_NACIONALIDAD[list_abbreviation[i]]
                    if nacionalidad == ""
                    else nacionalidad
                    for i, nacionalidad in enumerate(results["CountryCode"].to_list())
                ]

                # 2) Get Laps
                laps = session.laps

                # 3) Get Laps of the first driver
                fastest_driver = results.loc[
                    results["Position"] == 1.0, "Abbreviation"
                ].iloc[0]
                mask_driver = laps["Driver"] == fastest_driver
                subset_laps_driver = laps.loc[mask_driver,].copy()
                laps_fastest_driver = generate_correct_laptimes(
                    subset_laps_driver, session
                )

                laps_fastest_driver = [
                    laptime.total_seconds() for laptime in laps_fastest_driver
                ]
                totaltime_fastest_driver = sum(laps_fastest_driver)

                # 4) Get distances to the other drivers
                time_distance = []
                total_fast_driver_str = str(
                    timedelta(seconds=totaltime_fastest_driver)
                )[:-3]
                time_distance.append(total_fast_driver_str)  # For the fastest

                for i in range(1, results.shape[0]):
                    driver = results["Abbreviation"].iloc[i]
                    status = results["Status"].iloc[i]

                    if status == "Finished":
                        # Calculate time distances by times in the race
                        mask_driver = laps["Driver"] == driver
                        subset_laps_driver = laps.loc[mask_driver,].copy()
                        laps_driver = generate_correct_laptimes(
                            subset_laps_driver, session
                        )

                        laps_driver = [
                            laptime.total_seconds() for laptime in laps_driver
                        ]
                        totaltime_driver = sum(laps_driver)
                        final_diff = totaltime_driver - totaltime_fastest_driver

                        # Check pending penalties
                        mask_driver_penalties = df_pending_penalties["Driver"] == driver
                        if sum(mask_driver_penalties) == 0:
                            final_diff = pass_to_finaltime(final_diff)
                            time_distance.append(final_diff)
                        else:
                            # Pending penalties
                            adding_time_for_penalty = sum(
                                df_pending_penalties.loc[
                                    mask_driver_penalties, "QuantityPenalty"
                                ].to_list()
                            )
                            final_diff += adding_time_for_penalty
                            final_diff = pass_to_finaltime(final_diff)
                            time_distance.append(final_diff)
                    else:
                        time_distance.append(DICT_STATUS[status])

                # 5) Create df
                # 5.1 Traducir equipos a nombre normal
                list_teams = [DICT_TEAMS[team] for team in list_teams]

                # 5.2 Cambiar los apellidos a mayusculas
                list_final_pilotos = []
                for piloto in list_pilotos:
                    nombre = piloto.split(" ")[0]
                    apellido = piloto.split(" ")[1].upper()
                    list_final_pilotos.append(nombre + " " + apellido)

                # Cambiar posicion a entero
                list_position = [int(posicion) for posicion in list_position]

                # Obtener lista de cambios de posicio
                list_cambios = []
                for i, drv in enumerate(list_abbreviation):
                    grid_posicion = results.loc[
                        results["Abbreviation"] == drv, "GridPosition"
                    ].iloc[0]
                    final_posicion = results.loc[
                        results["Abbreviation"] == drv, "Position"
                    ].iloc[0]
                    cambio = final_posicion - grid_posicion
                    if cambio > 0.0:
                        string = f"+{int(cambio)}"
                        list_cambios.append(str(string))
                    elif cambio < 0.0:
                        string = f"-{int(-cambio)}"
                        list_cambios.append(str(string))
                    else:
                        string = int(cambio)
                        list_cambios.append(str(string))

                d = {
                    "posicion": list_position,
                    "cambios": list_cambios,
                    "nacionalidad": list_nacionalidad,
                    "Driver": list_abbreviation,
                    "piloto": list_final_pilotos,
                    "equipo": list_teams,
                    "tiempo": time_distance,
                    "puntos": list_points,
                }
                df_results = pd.DataFrame(d)

                # Ordenar
                df_results.sort_values(by="posicion", ascending=True, inplace=True)

                # Show
                if session.session_info["Type"] == "Race":
                    # Penalizaciones
                    print(f"\nPenalizaciones cometidas en carrera:")
                    print(
                        tabulate(
                            df_penalties,
                            headers="keys",
                            showindex=False,
                            tablefmt="grid",
                        )
                    )
                    print(f"\nPenalizaciones pendientes de servir:")
                    print(
                        tabulate(
                            df_pending_penalties,
                            headers="keys",
                            showindex=False,
                            tablefmt="grid",
                        )
                    )

                    # Resultados
                    print(f"\nResultados de la {session.session_info['Type']}")
                    print(
                        tabulate(
                            df_results, headers="keys", showindex=False, tablefmt="grid"
                        )
                    )
                else:
                    # Resultados
                    print(f"\nResultados de la {session.session_info['Type']}")
                    print(
                        tabulate(
                            df_results, headers="keys", showindex=False, tablefmt="grid"
                        )
                    )

                return df_results
        except Exception as e:
            raise e

    # Ordenar, tabulate and return
    # df_results.sort_values(by='Position', ascending=True, inplace=True)
    print(f"\nResultados de la {session.session_info['Type']}")
    print(tabulate(df_results, headers="keys", showindex=False, tablefmt="grid"))
    return df_results


def update_qualy_stats(df_results):
    # Read Stats
    outputs_path = os.path.join(os.getcwd(), "src", "outputs")
    df_stats_pilotos = pd.read_csv(
        os.path.join(outputs_path, "stats_pilotos.csv"), sep=","
    )
    df_stats_constructores = pd.read_csv(
        os.path.join(outputs_path, "stats_constructores.csv"), sep=","
    )

    # Obtener datos del piloto mas rapido
    mask_fastest = df_results["posicion"] == 1
    subset_fastest = df_results.loc[mask_fastest,].copy()

    piloto = subset_fastest["Driver"].iloc[0]
    equipo = subset_fastest["equipo"].iloc[0]

    # Actualizar stats de poles
    df_stats_pilotos.loc[df_stats_pilotos["abreviatura"] == piloto, "poles"] += 1
    df_stats_constructores.loc[df_stats_constructores["equipo"] == equipo, "poles"] += 1

    # Save stats
    path = os.path.join(os.path.join(outputs_path, "stats_pilotos.csv"))
    df_stats_pilotos.to_csv(path, sep=",", index=False)

    path = os.path.join(os.path.join(outputs_path, "stats_constructores.csv"))
    df_stats_constructores.to_csv(path, sep=",", index=False)


def calcula_tasa_victorias(victorias, numero_carreras):
    tasa_victorias = victorias / numero_carreras
    tasa_victorias = tasa_victorias * 100
    tasa_victorias = int(round(tasa_victorias, 0))
    return tasa_victorias


def update_race_stats(session, df_results):
    # Read Stats
    outputs_path = os.path.join(os.getcwd(), "src", "outputs")
    df_stats_pilotos = pd.read_csv(
        os.path.join(outputs_path, "stats_pilotos.csv"), sep=","
    )
    df_stats_constructores = pd.read_csv(
        os.path.join(outputs_path, "stats_constructores.csv"), sep=","
    )

    # ACTUALIZAR ESTADISTICAS GLOBALES
    # ------------------------------------------------------------------------------------------------------------------
    if session.name != "Sprint":
        for i in range(df_results.shape[0]):
            driver = df_results["Driver"].iloc[i]
            puntos = df_results["puntos"].iloc[i]
            status = session.results.loc[
                session.results["Abbreviation"] == driver, "Status"
            ].iloc[0]

            ## Puntos & Puntos Totales
            df_stats_pilotos.loc[
                df_stats_pilotos["abreviatura"] == driver, "puntos"
            ] += puntos
            df_stats_pilotos.loc[
                df_stats_pilotos["abreviatura"] == driver, "puntos_totales"
            ] += puntos

            ## GPs_totales
            df_stats_pilotos.loc[
                df_stats_pilotos["abreviatura"] == driver, "gps_totales"
            ] += 1

            ## DNFs
            if status not in ("Finished", "Lapped"):
                df_stats_pilotos.loc[
                    df_stats_pilotos["abreviatura"] == driver, "dnfs"
                ] += 1

        # ACTUALIZAR ESTADISTICAS DE TOP 10s, PODIO y VICTORIAS
        # ------------------------------------------------------------------------------------------------------------------
        ## TOP 10s, Podios & Podios totales y Victorias
        mask_top_tens = df_results["posicion"] <= 10
        subset_top_tens = df_results.loc[mask_top_tens,].copy()

        for i in range(subset_top_tens.shape[0]):
            piloto = subset_top_tens["Driver"].iloc[i]
            equipo = subset_top_tens["equipo"].iloc[i]
            posicion = subset_top_tens["posicion"].iloc[i]

            # Actualizar top 10s
            df_stats_pilotos.loc[
                df_stats_pilotos["abreviatura"] == piloto, "top_tens"
            ] += 1

            # Check if Podium
            if posicion <= 3:
                df_stats_pilotos.loc[
                    df_stats_pilotos["abreviatura"] == piloto, "podios"
                ] += 1
                df_stats_pilotos.loc[
                    df_stats_pilotos["abreviatura"] == piloto, "podios_totales"
                ] += 1
                df_stats_constructores.loc[
                    df_stats_constructores["equipo"] == equipo, "podios"
                ] += 1

                # Check victorias
                if posicion == 1:
                    df_stats_pilotos.loc[
                        df_stats_pilotos["abreviatura"] == piloto, "victorias"
                    ] += 1
                    df_stats_pilotos.loc[
                        df_stats_pilotos["abreviatura"] == piloto, "victorias_totales"
                    ] += 1
                    df_stats_constructores.loc[
                        df_stats_constructores["equipo"] == equipo, "victorias"
                    ] += 1

        # ACTUALIZAR ESTADISTICAS TASA DE VICTORIAS
        # ------------------------------------------------------------------------------------------------------------------
        num_races = session.event.RoundNumber
        df_stats_pilotos["tasa_victorias"] = df_stats_pilotos["victorias"].apply(
            lambda i: calcula_tasa_victorias(i, num_races)
        )

    elif session.name == "Sprint":
        for i in range(df_results.shape[0]):
            driver = df_results["Driver"].iloc[i]
            puntos = df_results["puntos"].iloc[i]

            ## Puntos & Puntos Totales
            df_stats_pilotos.loc[
                df_stats_pilotos["abreviatura"] == driver, "puntos"
            ] += puntos
            df_stats_pilotos.loc[
                df_stats_pilotos["abreviatura"] == driver, "puntos_totales"
            ] += puntos

    # ORDENAR POR PUNTOS
    # ------------------------------------------------------------------------------------------------------------------
    df_stats_pilotos.sort_values(by=["puntos"], ascending=False, inplace=True)

    # SAVE STATS
    # ------------------------------------------------------------------------------------------------------------------
    path = os.path.join(os.path.join(outputs_path, "stats_pilotos.csv"))
    df_stats_pilotos.to_csv(path, sep=",", index=False)

    path = os.path.join(os.path.join(outputs_path, "stats_constructores.csv"))
    df_stats_constructores.to_csv(path, sep=",", index=False)


def convert_img_to_ig(img_path, img_name):
    # Dimensions max in insta 1080 x 566
    # 0) Load image
    """pil_image_lst = convert_from_path(os.path.join(img_path, img_name))  # This returns a list even for a 1 page pdf
    pil_image = pil_image_lst[0]
    img = pil_image"""
    img = Image.open(os.path.join(img_path, img_name))

    # 1) Calculate factor
    length = img.size[0]
    width = img.size[1]

    print(length, width)
    alpha = 1080 / length

    # 2) Save image
    print(f"Nuevas Dimensiones con alpha = ({length * alpha}, {width * alpha})")
    img_resized_alpha = img.resize(
        (int(length * alpha), int(width * alpha)), Image.LANCZOS
    )

    name_sin_extension = img_name.split(".")[0]
    img_resized_alpha.save(
        os.path.join(img_path, f"{name_sin_extension}_reduced.png"), format="png"
    )


def resize_and_pad_images(img_path, output_path, target_size):
    # Obtener la lista de imagenes dentro de img_path
    list_images = os.listdir(img_path)
    list_images = [image for image in list_images if image != ".DS_Store"]

    for image in list_images:
        resize_and_pad_image(
            img_path=img_path,
            img_name=image,
            output_path=output_path,
            target_size=target_size,
        )


def convert_image_tiktok_margin(img_path, img_name, transparent):
    """
    Ajusta una imagen añadiendo rectángulos de color alrededor para convertirla al formato 9:16.

    Parámetros:
        imagen_path (str): Ruta de la imagen original.
        salida_path (str): Ruta donde se guardará la imagen ajustada.
    """
    if transparent:
        # Cargar la imagen original
        imagen = Image.open(os.path.join(img_path, img_name))
        ancho, alto = imagen.size

        # Calcular el ancho de los rectángulos izquierdo y derecho
        ancho_rect_lados = int((ancho * 80) / 1320)

        # Crear nueva imagen con márgenes laterales
        nuevo_ancho = ancho + 2 * ancho_rect_lados
        alto_destino = int((nuevo_ancho * 16) / 9)
        margen_superior_inferior = (alto_destino - alto) // 2

        # Crear la imagen final con formato 9:16
        imagen_final = Image.new("RGBA", (nuevo_ancho, alto_destino), (36, 34, 33, 255))

        # Crear una versión escalada y transparente de la imagen original
        imagen_escalada = imagen.resize((nuevo_ancho, alto_destino), Image.LANCZOS)
        imagen_transparente = ImageEnhance.Brightness(imagen_escalada).enhance(
            0.5
        )  # Reducir brillo para dar transparencia
        imagen_transparente.putalpha(
            int(0.5 * 255)
        )  # Establecer transparencia (alpha=0.3)
        imagen_final.paste(imagen_transparente, (0, 0), imagen_transparente)

        # Pegar la imagen original centrada en la imagen final
        imagen_final.paste(
            imagen, (ancho_rect_lados, margen_superior_inferior), imagen.convert("RGBA")
        )

        # Guardar la imagen ajustada
        image_name = img_name.split(".")[0]  # Quitar extension
        new_img_name = image_name + "_tiktok.png"
        imagen_final.save(os.path.join(img_path, new_img_name))
        print(f"Imagen {new_img_name} guardada en {img_path}")

    else:
        # Cargar la imagen original
        imagen = Image.open(os.path.join(img_path, img_name))
        ancho, alto = imagen.size

        # Color de los márgenes
        color_margen = (41, 38, 37)

        # Calcular el ancho de los rectángulos izquierdo y derecho
        ancho_rect_lados = int((ancho * 150) / 1320)

        # Crear nueva imagen con márgenes laterales
        nuevo_ancho = ancho + 2 * ancho_rect_lados
        imagen_con_margenes = Image.new("RGB", (nuevo_ancho, alto), color_margen)
        imagen_con_margenes.paste(imagen, (ancho_rect_lados, 0))

        # Calcular las dimensiones necesarias para el formato 9:16
        alto_destino = int((nuevo_ancho * 16) / 9)
        margen_superior_inferior = (alto_destino - alto) // 2

        # Crear imagen final con formato 9:16
        imagen_final = Image.new("RGB", (nuevo_ancho, alto_destino), color_margen)
        imagen_final.paste(imagen_con_margenes, (0, margen_superior_inferior))

        # Guardar la imagen ajustada
        image_name = img_name.split(".")[0]  # Quitar extension
        new_img_name = image_name + "_tiktok.png"
        imagen_final.save(os.path.join(img_path, new_img_name))
        print(f"Imagen {new_img_name} guardada en {img_path}")


def convert_tiktok_image(img_path, image, divisiones):
    # Load image
    img = Image.open(os.path.join(img_path, image))
    ancho, alto = img.size
    image_name = image.split(".")[0]  # Quitar extension

    # Target width
    ancho_division = ancho // divisiones
    margen = 0
    # color = (36, 34, 33)
    color = (41, 38, 37)

    # Aspect ratio 9:16
    ratio_ancho = 9
    ratio_alto = 16

    for i in range(divisiones):
        # Definir el área de la división
        x_inicio = i * ancho_division
        x_fin = (i + 1) * ancho_division if (i + 1) < divisiones else ancho
        division = img.crop((x_inicio, 0, x_fin, alto))

        # Calcular la altura deseada en función del ancho y el aspecto 9:16
        altura_deseada = int(division.width * (ratio_alto / ratio_ancho))

        # Si la altura de la división es menor que la altura deseada, agregar padding
        """if division.size[1] < altura_deseada:
            division = ImageOps.pad(division, (division.width, altura_deseada), color=(36, 34, 33),
                                    centering=(0.5, 0.5))"""

        # Añadir margen vertical (izquierda y derecha)
        division = ImageOps.expand(division, border=(margen, 0), fill=color)

        """# Calcular el margen horizontal basado en el aspecto
        margen_horizontal = int(margen * (ratio_alto / ratio_ancho))

        # Añadir margen horizontal (arriba y abajo)
        division = ImageOps.expand(division, border=(0, margen_horizontal), fill=(36, 34, 33))"""

        # Guardar la imagen
        nombre_division = f"{image_name}_{i + 1}.png"
        ruta_guardado = os.path.join(img_path, nombre_division)
        division.save(ruta_guardado)
        print(f"Guardado: {ruta_guardado}")


def convert_tiktok_image_by_rangos(img_path, image, rangos):
    # Load image
    img = Image.open(os.path.join(img_path, image))
    ancho, alto = img.size
    image_name = image.split(".")[0]  # Quitar extension

    # Marge
    margen = 0
    # color = (36, 34, 33)
    color = (41, 38, 37)

    # Aspect ratio 9:16
    ratio_ancho = 9
    ratio_alto = 16

    for i, (x_inicio, x_fin) in enumerate(rangos):
        # Definir el area de la division
        division = img.crop((x_inicio, 0, x_fin, alto))

        # Calcular la altura deseada en función del ancho y el aspecto 9:16
        altura_deseada = int(division.width * (ratio_alto / ratio_ancho))

        # Si la altura de la división es menor que la altura deseada, agregar padding
        """if division.size[1] < altura_deseada:
            division = ImageOps.pad(division, (division.width, altura_deseada), color=(36, 34, 33),
                                    centering=(0.5, 0.5))"""

        # Añadir margen vertical (izquierda y derecha)
        division = ImageOps.expand(division, border=(margen, 0), fill=color)

        """# Calcular el margen horizontal basado en el aspecto
        margen_horizontal = int(margen * (ratio_alto / ratio_ancho))

        # Añadir margen horizontal (arriba y abajo)
        division = ImageOps.expand(division, border=(0, margen_horizontal), fill=color)"""

        # Guardar la imagen
        nombre_division = f"{image_name}_{i + 1}.png"
        ruta_guardado = os.path.join(img_path, nombre_division)
        division.save(ruta_guardado)
        print(f"Guardado: {ruta_guardado}")


def convert_tiktok_image_by_rangos_with_tiktok(img_path, image, rangos):
    # Load image
    img = Image.open(os.path.join(img_path, image))
    ancho, alto = img.size
    image_name = image.split(".")[0]  # Quitar extension

    # Marge
    margen = 0
    color = (41, 38, 37)
    color_margen = (41, 38, 37)

    for i, (x_inicio, x_fin) in enumerate(rangos):
        # Definir el area de la division
        division = img.crop((x_inicio, 0, x_fin, alto))

        # Añadir margen vertical (izquierda y derecha)
        division = ImageOps.expand(division, border=(margen, 0), fill=color)

        # Apply tiktok Margin
        # Dimensiones
        ancho, alto = division.size

        # Calcular el ancho de los rectángulos izquierdo y derecho
        ancho_rect_lados = int((ancho * 80) / 1320)

        # Crear nueva imagen con márgenes laterales
        nuevo_ancho = ancho + 2 * ancho_rect_lados
        imagen_con_margenes = Image.new("RGB", (nuevo_ancho, alto), color_margen)
        imagen_con_margenes.paste(division, (ancho_rect_lados, 0))

        # Calcular las dimensiones necesarias para el formato 9:16
        alto_destino = int((nuevo_ancho * 16) / 9)
        margen_superior_inferior = (alto_destino - alto) // 2

        # Crear imagen final con formato 9:16
        imagen_final = Image.new("RGB", (nuevo_ancho, alto_destino), color_margen)
        imagen_final.paste(imagen_con_margenes, (0, margen_superior_inferior))

        # Guardar la imagen ajustada
        new_img_name = f"{image_name}_tiktok_{i + 1}.png"
        imagen_final.save(os.path.join(img_path, new_img_name))
        print(f"Imagen {new_img_name} guardada en {img_path}")


def resize_and_pad_image(img_path, img_name, output_path, target_size=(1080, 1080)):
    img = Image.open(os.path.join(img_path, img_name))

    # Calcular la relación de aspecto y redimensionar la imagen para que quepa en el tamaño objetivo
    img.thumbnail(target_size, Image.LANCZOS)

    # Calcular la posición para centrar la imagen en el lienzo
    delta_w = target_size[0] - img.size[0]
    delta_h = target_size[1] - img.size[1]
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )

    # Crear una nueva imagen con el fondo blanco y pegar la imagen redimensionada en el centro
    img_padded = ImageOps.expand(img, padding, (30, 28, 28))  # Color blanco

    # Guardar la imagen final
    img_padded.save(os.path.join(output_path, f"Padded_{img_name}"), format="png")

    print(
        f"Imagen '{img_name}' redimensionada y ajustada a {target_size}, guardada como 'Padded_{img_name}'."
    )


def update_mundial_pilotos(df_mundial):
    # CARGAR STATS DE PILOTOS
    # ------------------------------------------------------------------------------------------------------------------
    outputs_path = os.path.join(os.getcwd(), "src", "outputs")
    df_stats_pilotos = pd.read_csv(
        os.path.join(outputs_path, "stats_pilotos.csv"), sep=","
    )

    # ACTUALIZAR PUNTOS EN BASE A STATS PILOTOS
    # ------------------------------------------------------------------------------------------------------------------
    for i in range(df_stats_pilotos.shape[0]):
        driver = df_stats_pilotos["abreviatura"].iloc[i]
        puntos = df_stats_pilotos["puntos"].iloc[i]

        # Actualizar
        df_mundial.loc[df_mundial["abreviatura"] == driver, "puntos"] = puntos

    # REORDENAR POSICIONES
    # ------------------------------------------------------------------------------------------------------------------
    df_mundial.sort_values(by=["puntos"], ascending=False, inplace=True)

    # REACTUALIZAR POSICION
    # ------------------------------------------------------------------------------------------------------------------
    list_posicion = [i + 1 for i in range(df_mundial.shape[0])]
    df_mundial["posicion"] = list_posicion

    # SHOW RESULTS
    # ------------------------------------------------------------------------------------------------------------------
    print("\nMundial de pilotos")
    print(tabulate(df_mundial, headers="keys", showindex=False, tablefmt="grid"))

    # SAVE
    # ------------------------------------------------------------------------------------------------------------------
    df_mundial.to_csv(
        os.path.join(outputs_path, "mundial_pilotos.csv"), sep=",", index=False
    )


def update_mundial_pilotos_historic(df_historic):
    # CARGAR STATS DE PILOTOS
    # ------------------------------------------------------------------------------------------------------------------
    df_stats_pilotos = pd.read_csv(
        os.path.join(os.getcwd(), "src", "outputs", "stats_pilotos.csv"), sep=","
    )

    # ACTUALIZAR PUNTOS EN BASE A STATS PILOTOS
    # ------------------------------------------------------------------------------------------------------------------
    nueva_fila = {}

    ## Check if there are registers in Carrera
    if len(df_historic["Carrera"].to_list()) == 0:
        new_carrera = 1
    else:
        carrera = df_historic["Carrera"].iloc[-1]
        new_carrera = int(carrera) + 1

    ## Carrera
    nueva_fila["Carrera"] = new_carrera
    for i in range(df_stats_pilotos.shape[0]):
        driver = df_stats_pilotos["abreviatura"].iloc[i]
        puntos = df_stats_pilotos["puntos"].iloc[i]

        # Actualizar
        nueva_fila[driver] = puntos

    df_historic.loc[len(df_historic)] = nueva_fila

    # SAVE
    # ------------------------------------------------------------------------------------------------------------------
    df_historic.to_csv(
        os.path.join(os.getcwd(), "src", "outputs", "historic_mundial_pilotos.csv"),
        sep=",",
        index=False,
    )


def update_posicion_pilotos_historic(df, df_results):
    nueva_fila = {}

    ## Check if there are registers in Carrera
    if len(df["Carrera"].to_list()) == 0:
        new_carrera = 1
    else:
        carrera = df["Carrera"].iloc[-1]
        new_carrera = int(carrera) + 1
    nueva_fila["Carrera"] = new_carrera

    for i in range(df_results.shape[0]):
        driver = df_results["Driver"].iloc[i]
        puntos = df_results["posicion"].iloc[i]

        # Actualizar
        nueva_fila[driver] = puntos

    df.loc[len(df)] = nueva_fila

    # SAVE
    # ------------------------------------------------------------------------------------------------------------------
    df.to_csv(
        os.path.join(os.getcwd(), "src", "outputs", "historic_posiciones_pilotos.csv"),
        sep=",",
        index=False,
    )


def update_mundial_equipos_historic(df_mundial, df_historic):
    # ACTUALIZAR PUNTOS EN BASE A STATS EQUIPOS
    # ------------------------------------------------------------------------------------------------------------------
    nueva_fila = {}

    ## Check if there are registers in Carrera
    if len(df_historic["Carrera"].to_list()) == 0:
        new_carrera = 1
    else:
        carrera = df_historic["Carrera"].iloc[-1]
        new_carrera = int(carrera) + 1

    ## Carrera
    nueva_fila["Carrera"] = new_carrera
    for i in range(df_mundial.shape[0]):
        equipo = df_mundial["equipo"].iloc[i]
        puntos = df_mundial["puntos"].iloc[i]

        # Actualizar
        nueva_fila[equipo] = puntos

    df_historic.loc[len(df_historic)] = nueva_fila

    # SAVE
    # ------------------------------------------------------------------------------------------------------------------
    df_historic.to_csv(
        os.path.join(
            os.getcwd(), "src", "outputs", "historic_mundial_constructores.csv"
        ),
        sep=",",
        index=False,
    )


def update_mundial_equipos(session, df_mundial):
    # Extraer de la sesion los resultados
    df_results = session.results

    # Obtener los equipos
    list_teams = np.unique(df_results["TeamName"].to_list())

    # Iterar los equipos para actualizar los puntos
    for team in list_teams:
        # Obtener los puntos por equipo
        mask_team = df_results["TeamName"] == team
        list_points = df_results.loc[mask_team, "Points"].to_list()
        total_points = sum(list_points)

        # Sumar al df_mundial
        mundial_team = DICT_TEAMS[team]
        df_mundial.loc[df_mundial["equipo"] == mundial_team, "puntos"] += total_points

    # Reordenar las posiciones
    df_mundial.sort_values(by=["puntos"], ascending=False, inplace=True)

    # Reactualizar posicion
    list_posicion = [i + 1 for i in range(df_mundial.shape[0])]
    df_mundial["posicion"] = list_posicion

    # Mostrar resultado
    print("\nMundial de equipos")
    print(tabulate(df_mundial, headers="keys", showindex=False, tablefmt="grid"))

    # Guardar
    df_mundial.to_csv(
        os.path.join(os.getcwd(), "src", "outputs", "mundial_equipos.csv"),
        sep=",",
        index=False,
    )


def analisis_degradacion(session):
    # 1) OBTENER LAPS
    laps = session.laps

    # 1.1) Obtener listas de equipos y de stints usados
    list_teams = np.unique(laps["Team"].to_list())
    list_stints = np.unique(laps["Stint"].to_list())

    # 2) CALCULAR


def reject_outliers(data):
    m = 2
    u = np.mean(data)
    s = np.std(data)
    filtered = [e for e in data if (u - 2 * s < e < u + 2 * s)]
    return filtered


def filter_sc(laps_driver, df_race_control):
    # Calculate laps of SC
    mask_safetycar_deployed = df_race_control["Message"] == "SAFETY CAR DEPLOYED"
    mask_safetycar_inthislap = df_race_control["Message"] == "SAFETY CAR IN THIS LAP"

    laps_sc_deployed = df_race_control.loc[mask_safetycar_deployed, "Lap"].to_list()
    laps_sc_in_this_lap = df_race_control.loc[mask_safetycar_inthislap, "Lap"].to_list()

    # Search in evry SC window and delete laps
    mask = pd.Series([True] * len(laps_driver), index=laps_driver.index)

    for inicio, fin in zip(laps_sc_deployed, laps_sc_in_this_lap):
        mask &= ~laps_driver["LapNumber"].between(inicio, fin)

    return laps_driver[mask]


def filter_vsc(laps_driver, df_race_control):
    # Calculate laps of VSC
    mask_vsafetycardeployed = (
        df_race_control["Message"] == "VIRTUAL SAFETY CAR DEPLOYED"
    )
    mask_vsafetycarending = df_race_control["Message"] == "VIRTUAL SAFETY CAR ENDING"

    laps_vsc_deployed = df_race_control.loc[mask_vsafetycardeployed, "Lap"].to_list()
    laps_vsc_ending = df_race_control.loc[mask_vsafetycarending, "Lap"].to_list()

    # Search in evry SC window and delete laps
    mask = pd.Series([True] * len(laps_driver), index=laps_driver.index)

    for inicio, fin in zip(laps_vsc_deployed, laps_vsc_ending):
        mask &= ~laps_driver["LapNumber"].between(inicio, fin)

    return laps_driver[mask]


def filter_bad_event(laps_driver, df_race_control):
    # SafetyCar
    laps_driver = filter_sc(laps_driver, df_race_control)

    # Virtual SafetyCar
    laps_driver = filter_vsc(laps_driver, df_race_control)

    return laps_driver


def analiza_stint_compound(session, compound, stint):
    # Constantes
    delta_t_fuel = 0.3

    # 1) HACER UN SUBSET DE LAPS CON COMPOUND Y STINT
    laps = session.laps
    df_race_control = session.race_control_messages

    ## Filtrar las vueltas que no son de SafetyCar
    mask_safetycar_deployed = df_race_control["Message"] == "SAFETY CAR DEPLOYED"
    mask_safetycar_inthislap = df_race_control["Message"] == "SAFETY CAR IN THIS LAP"
    laps_sc_deployed = df_race_control.loc[mask_safetycar_deployed, "Lap"].to_list()
    laps_sc_inthislap = df_race_control.loc[mask_safetycar_inthislap, "Lap"].to_list()

    """mask_stint = laps['Stint'] == stint
    mask_compound = laps['Compound'] == compound
    subset_stint_compound = laps.loc[mask_stint&mask_compound,].copy()"""
    mask_compound = laps["Compound"] == compound
    mask_limit_lap = laps["LapNumber"] > 0.0

    subset_stint_compound = laps.loc[mask_limit_lap & mask_compound,].copy()

    # 2) RECORRER EL SUBSET POR EQUIPOS
    data_global = {
        "Stint": stint,
        "Compound": compound,
    }
    list_teams = np.unique(subset_stint_compound["Team"].to_list())
    top_teams = ["Aston Martin", "Ferrari", "Red Bull Racing", "Mercedes", "McLaren"]
    list_teams = top_teams
    list_ritmos = []
    for team in list_teams:
        # Subset vueltas de ese equipo
        mask_equipo = subset_stint_compound["Team"] == team
        subset_equipo = subset_stint_compound.loc[mask_equipo,].copy()

        # Pilotos de ese equipo
        list_pilotos = np.unique(subset_equipo["Driver"].to_list())

        if len(list_pilotos) > 1:
            # Extract laps
            laps_driver1 = subset_equipo.loc[
                subset_equipo["Driver"] == list_pilotos[0],
            ]
            laps_driver2 = subset_equipo.loc[
                subset_equipo["Driver"] == list_pilotos[1],
            ]

            # Check the number of laps
            if laps_driver2.shape[0] < 2:
                # Only driver 1
                laps_driver1["DeltaTyreLife"] = laps_driver1["TyreLife"].diff()
                if sum(laps_driver1["DeltaTyreLife"] < 0.0) == 0:
                    lap_limit_d1 = laps_driver1["LapNumber"].iloc[-2]
                else:
                    lap_limit_d1 = laps_driver1.loc[
                        laps_driver1["DeltaTyreLife"] < 0.0, "LapNumber"
                    ].to_list()[0]

                laps_driver1 = laps_driver1.loc[
                    laps_driver1["LapNumber"] < lap_limit_d1,
                ]
                laps_driver1 = filter_sc(
                    laps_driver1, laps_sc_deployed, laps_sc_inthislap
                )
                laps_driver1 = generate_correct_laptimes(laps_driver1, session)
                laps_driver1 = [lap.total_seconds() for lap in laps_driver1]

                # Calculate mean
                mean_driver1 = np.median(reject_outliers(laps_driver1))

                # Save
                n_laps = len(laps_driver1)
                list_ritmos.append(
                    {
                        "Team": team,
                        "Driver": list_pilotos[0],
                        "Laps": laps_driver1,
                        "Ritmo": mean_driver1,
                        "n_laps": n_laps,
                    }
                )
            else:
                if laps_driver1.shape[0] < 2:
                    # Only driver 2
                    laps_driver2["DeltaTyreLife"] = laps_driver2["TyreLife"].diff()
                    if sum(laps_driver2["DeltaTyreLife"] < 0.0) == 0:
                        lap_limit_d1 = laps_driver2["LapNumber"].iloc[-2]
                    else:
                        lap_limit_d1 = laps_driver2.loc[
                            laps_driver2["DeltaTyreLife"] < 0.0, "LapNumber"
                        ].to_list()[0]

                    laps_driver2 = laps_driver2.loc[
                        laps_driver2["LapNumber"] < lap_limit_d1,
                    ]
                    laps_driver2 = filter_sc(
                        laps_driver2, laps_sc_deployed, laps_sc_inthislap
                    )
                    laps_driver2 = generate_correct_laptimes(laps_driver2, session)
                    laps_driver2 = [lap.total_seconds() for lap in laps_driver2]

                    # Calculate mean
                    mean_driver2 = np.median(reject_outliers(laps_driver2))

                    # Save
                    n_laps = len(laps_driver2)
                    list_ritmos.append(
                        {
                            "Team": team,
                            "Driver": list_pilotos[1],
                            "Laps": laps_driver2,
                            "Ritmo": mean_driver2,
                            "n_laps": n_laps,
                        }
                    )
                else:
                    # TODO FILTrar primera vuelta de delta negativa de tyrelife
                    laps_driver1["DeltaTyreLife"] = laps_driver1["TyreLife"].diff()
                    if sum(laps_driver1["DeltaTyreLife"] < 0.0) == 0:
                        lap_limit_d1 = laps_driver1["LapNumber"].iloc[-2]
                    else:
                        lap_limit_d1 = laps_driver1.loc[
                            laps_driver1["DeltaTyreLife"] < 0.0, "LapNumber"
                        ].to_list()[0]

                    laps_driver2["DeltaTyreLife"] = laps_driver2["TyreLife"].diff()
                    if sum(laps_driver2["DeltaTyreLife"] < 0.0) == 0:
                        lap_limit_d2 = laps_driver2["LapNumber"].iloc[-2]
                    else:
                        lap_limit_d2 = laps_driver2.loc[
                            laps_driver2["DeltaTyreLife"] < 0.0, "LapNumber"
                        ].to_list()[0]

                    laps_driver1 = laps_driver1.loc[
                        laps_driver1["LapNumber"] < lap_limit_d1,
                    ]
                    laps_driver2 = laps_driver2.loc[
                        laps_driver2["LapNumber"] < lap_limit_d2,
                    ]

                    # Filter possible SC
                    laps_driver1 = filter_sc(
                        laps_driver1, laps_sc_deployed, laps_sc_inthislap
                    )
                    laps_driver2 = filter_sc(
                        laps_driver2, laps_sc_deployed, laps_sc_inthislap
                    )

                    # Correct the NaT laps using the start tiem of the lap and the startinf time of the second lap
                    laps_driver1 = generate_correct_laptimes(laps_driver1, session)
                    laps_driver2 = generate_correct_laptimes(laps_driver2, session)

                    # Convert laps in seconds
                    laps_driver1 = [lap.total_seconds() for lap in laps_driver1]
                    laps_driver2 = [lap.total_seconds() for lap in laps_driver2]

                    # Calcular la media de tiempo y mediana
                    # mean_driver1 = np.mean(laps_driver1[6:])
                    # mean_driver1 = np.median(reject_outliers(laps_driver1[:19]))
                    mean_driver1 = np.median(reject_outliers(laps_driver1))

                    # mean_driver2 = np.mean(laps_driver2[6:])
                    # mean_driver2 = np.median(reject_outliers(laps_driver2[:19]))
                    mean_driver2 = np.median(reject_outliers(laps_driver2))

                    # Tomar como equipo el de valor más pequeño
                    if mean_driver1 <= mean_driver2:
                        # Mas rapido Driver 1
                        n_laps = len(laps_driver1)
                        list_ritmos.append(
                            {
                                "Team": team,
                                "Driver": list_pilotos[0],
                                "Laps": laps_driver1,
                                "Ritmo": mean_driver1,
                                "n_laps": n_laps,
                            }
                        )

                    elif mean_driver1 > mean_driver2:
                        # Mas rapido Driver 2
                        n_laps = len(laps_driver2)
                        list_ritmos.append(
                            {
                                "Team": team,
                                "Driver": list_pilotos[1],
                                "Laps": laps_driver2,
                                "Ritmo": mean_driver2,
                                "n_laps": n_laps,
                            }
                        )

        else:
            # Extraer vueltas
            laps_driver1 = subset_equipo.loc[
                subset_equipo["Driver"] == list_pilotos[0],
            ]

            laps_driver1["DeltaTyreLife"] = laps_driver1["TyreLife"].diff()
            if sum(laps_driver1["DeltaTyreLife"] < 0.0) == 0:
                lap_limit_d1 = laps_driver1["LapNumber"].iloc[-2]
            else:
                lap_limit_d1 = laps_driver1.loc[
                    laps_driver1["DeltaTyreLife"] < 0.0, "LapNumber"
                ].to_list()[0]
            laps_driver1 = laps_driver1.loc[laps_driver1["LapNumber"] < lap_limit_d1,]

            # Correct the NaT laps using the start tiem of the lap and the startinf time of the second lap
            laps_driver1 = generate_correct_laptimes(laps_driver1, session)

            # Convert laps in seconds
            laps_driver1 = [lap.total_seconds() for lap in laps_driver1]

            # Calcular la media de tiempo y mediana
            # mean_driver1 = np.mean(laps_driver1)
            # mean_driver1 = np.median(reject_outliers(laps_driver1[:19]))
            mean_driver1 = np.median(reject_outliers(laps_driver1))

            # Mas rapido Driver 1
            n_laps = len(laps_driver1)
            list_ritmos.append(
                {
                    "Team": team,
                    "Driver": list_pilotos[0],
                    "Laps": laps_driver1,
                    "Ritmo": mean_driver1,
                    "n_laps": n_laps,
                }
            )

    # Save global
    data_global["info_equipos"] = list_ritmos

    # 3) RETURN
    return data_global


def analiza_stint(session, stint):
    # Constantes
    delta_t_fuel = 0.3

    # 1) HACER UN SUBSET DE LAPS CON COMPOUND Y STINT
    laps = session.laps
    df_race_control = session.race_control_messages

    mask_stint = laps["Stint"] == stint
    mask_limit_lap = laps["LapNumber"] > 0.0
    subset_stint = laps.loc[mask_limit_lap & mask_stint,].copy()

    if stint > 1.0:
        mask_pit_out = subset_stint["PitOutTime"].isna()
        subset_stint = subset_stint.loc[mask_pit_out,].copy()

    # 2) RECORRER EL SUBSET POR EQUIPOS
    data_global = {"Stint": stint}
    list_teams = np.unique(subset_stint["Team"].to_list())
    top_teams = ["Aston Martin", "Ferrari", "Red Bull Racing", "Mercedes", "McLaren"]
    list_teams = top_teams
    list_ritmos = []
    for team in list_teams:
        # Subset vueltas de ese equipo
        mask_equipo = subset_stint["Team"] == team
        subset_equipo = subset_stint.loc[mask_equipo,].copy()

        # Pilotos de ese equipo
        list_pilotos = np.unique(subset_equipo["Driver"].to_list())

        if len(list_pilotos) > 1:
            # Extract laps
            laps_driver1 = subset_equipo.loc[
                subset_equipo["Driver"] == list_pilotos[0],
            ]
            laps_driver2 = subset_equipo.loc[
                subset_equipo["Driver"] == list_pilotos[1],
            ]

            # Check the number of laps
            if laps_driver2.shape[0] < 2:
                # Only driver 1
                laps_driver1["DeltaTyreLife"] = laps_driver1["TyreLife"].diff()
                if sum(laps_driver1["DeltaTyreLife"] < 0.0) == 0:
                    lap_limit_d1 = laps_driver1["LapNumber"].iloc[-2]
                else:
                    lap_limit_d1 = laps_driver1.loc[
                        laps_driver1["DeltaTyreLife"] < 0.0, "LapNumber"
                    ].to_list()[0]

                laps_driver1 = laps_driver1.loc[
                    laps_driver1["LapNumber"] < lap_limit_d1,
                ]
                laps_driver1 = filter_bad_event(laps_driver1, df_race_control)
                laps_driver1 = generate_correct_laptimes(laps_driver1, session)
                laps_driver1 = [lap.total_seconds() for lap in laps_driver1]
                laps_driver1 = reject_outliers(laps_driver1)  #

                # Calculate mean
                # mean_driver2 = np.median(reject_outliers(laps_driver1))
                mean_driver1 = np.median(laps_driver1)

                # Save
                n_laps = len(laps_driver1)
                list_ritmos.append(
                    {
                        "Team": team,
                        "Driver": list_pilotos[0],
                        "Laps": laps_driver1,
                        "Ritmo": mean_driver1,
                        "n_laps": n_laps,
                    }
                )
            else:
                if laps_driver1.shape[0] < 2:
                    # Only driver 2
                    laps_driver2["DeltaTyreLife"] = laps_driver2["TyreLife"].diff()
                    if sum(laps_driver2["DeltaTyreLife"] < 0.0) == 0:
                        lap_limit_d1 = laps_driver2["LapNumber"].iloc[-2]
                    else:
                        lap_limit_d1 = laps_driver2.loc[
                            laps_driver2["DeltaTyreLife"] < 0.0, "LapNumber"
                        ].to_list()[0]

                    laps_driver2 = laps_driver2.loc[
                        laps_driver2["LapNumber"] < lap_limit_d1,
                    ]
                    laps_driver2 = filter_bad_event(laps_driver2, df_race_control)

                    laps_driver2 = generate_correct_laptimes(laps_driver2, session)
                    laps_driver2 = [lap.total_seconds() for lap in laps_driver2]
                    laps_driver2 = reject_outliers(laps_driver2)  #

                    # Calculate mean
                    # mean_driver2 = np.median(reject_outliers(laps_driver2))
                    mean_driver2 = np.median(laps_driver2)

                    # Save
                    n_laps = len(laps_driver2)
                    list_ritmos.append(
                        {
                            "Team": team,
                            "Driver": list_pilotos[1],
                            "Laps": laps_driver2,
                            "Ritmo": mean_driver2,
                            "n_laps": n_laps,
                        }
                    )
                else:
                    # TODO FILTrar primera vuelta de delta negativa de tyrelife
                    laps_driver1["DeltaTyreLife"] = laps_driver1["TyreLife"].diff()
                    if sum(laps_driver1["DeltaTyreLife"] < 0.0) == 0:
                        lap_limit_d1 = laps_driver1["LapNumber"].iloc[-2]
                    else:
                        lap_limit_d1 = laps_driver1.loc[
                            laps_driver1["DeltaTyreLife"] < 0.0, "LapNumber"
                        ].to_list()[0]

                    laps_driver2["DeltaTyreLife"] = laps_driver2["TyreLife"].diff()
                    if sum(laps_driver2["DeltaTyreLife"] < 0.0) == 0:
                        lap_limit_d2 = laps_driver2["LapNumber"].iloc[-2]
                    else:
                        lap_limit_d2 = laps_driver2.loc[
                            laps_driver2["DeltaTyreLife"] < 0.0, "LapNumber"
                        ].to_list()[0]

                    laps_driver1 = laps_driver1.loc[
                        laps_driver1["LapNumber"] < lap_limit_d1,
                    ]
                    laps_driver2 = laps_driver2.loc[
                        laps_driver2["LapNumber"] < lap_limit_d2,
                    ]

                    # Filter possible SC
                    laps_driver1 = filter_bad_event(laps_driver1, df_race_control)
                    laps_driver2 = filter_bad_event(laps_driver2, df_race_control)

                    # Correct the NaT laps using the start tiem of the lap and the startinf time of the second lap
                    laps_driver1 = generate_correct_laptimes(laps_driver1, session)
                    laps_driver2 = generate_correct_laptimes(laps_driver2, session)

                    # Convert laps in seconds
                    laps_driver1 = [lap.total_seconds() for lap in laps_driver1]
                    laps_driver2 = [lap.total_seconds() for lap in laps_driver2]

                    # Reject Outliers
                    laps_driver1 = reject_outliers(laps_driver1)  #
                    laps_driver2 = reject_outliers(laps_driver2)  #

                    # Calcular la media de tiempo y mediana
                    # mean_driver1 = np.median(reject_outliers(laps_driver1))
                    # mean_driver2 = np.median(reject_outliers(laps_driver2))
                    mean_driver1 = np.median(laps_driver1)
                    mean_driver2 = np.median(laps_driver2)

                    # Tomar como equipo el de valor más pequeño
                    if mean_driver1 <= mean_driver2:
                        # Mas rapido Driver 1
                        n_laps = len(laps_driver1)
                        list_ritmos.append(
                            {
                                "Team": team,
                                "Driver": list_pilotos[0],
                                "Laps": laps_driver1,
                                "Ritmo": mean_driver1,
                                "n_laps": n_laps,
                            }
                        )

                    elif mean_driver1 > mean_driver2:
                        # Mas rapido Driver 2
                        n_laps = len(laps_driver2)
                        list_ritmos.append(
                            {
                                "Team": team,
                                "Driver": list_pilotos[1],
                                "Laps": laps_driver2,
                                "Ritmo": mean_driver2,
                                "n_laps": n_laps,
                            }
                        )

        else:
            # Extraer vueltas
            laps_driver1 = subset_equipo.loc[
                subset_equipo["Driver"] == list_pilotos[0],
            ]

            laps_driver1["DeltaTyreLife"] = laps_driver1["TyreLife"].diff()
            if sum(laps_driver1["DeltaTyreLife"] < 0.0) == 0:
                lap_limit_d1 = laps_driver1["LapNumber"].iloc[-2]
            else:
                lap_limit_d1 = laps_driver1.loc[
                    laps_driver1["DeltaTyreLife"] < 0.0, "LapNumber"
                ].to_list()[0]
            laps_driver1 = laps_driver1.loc[laps_driver1["LapNumber"] < lap_limit_d1,]

            # Filter bas events and Correct the NaT
            laps_driver1 = filter_bad_event(laps_driver1, df_race_control)
            laps_driver1 = generate_correct_laptimes(laps_driver1, session)

            # Convert laps in seconds
            laps_driver1 = [lap.total_seconds() for lap in laps_driver1]

            # Reject Outliers
            laps_driver1 = reject_outliers(laps_driver1)  #

            # Calcular la media de tiempo y mediana
            # mean_driver1 = np.median(reject_outliers(laps_driver1))
            mean_driver1 = np.median(laps_driver1)

            # Mas rapido Driver 1
            n_laps = len(laps_driver1)
            list_ritmos.append(
                {
                    "Team": team,
                    "Driver": list_pilotos[0],
                    "Laps": laps_driver1,
                    "Ritmo": mean_driver1,
                    "n_laps": n_laps,
                }
            )

    # Save global
    data_global["info_equipos"] = list_ritmos

    # 3) RETURN
    return data_global


def get_list_sorted(df_historic):
    # Create transpose historic df
    df_historic_transpose = df_historic
    df_historic_transpose = df_historic_transpose.set_index("Carrera")
    df_historic_transpose = df_historic_transpose.T

    # Sort by the last one column (which is the last one race)
    last_carrera = df_historic_transpose.columns.to_list()[-1]
    df_historic_transpose = df_historic_transpose.sort_values(
        by=last_carrera, ascending=False
    )

    # Extract the index list (now sorted) and return it
    list_index_sorted = df_historic_transpose.index.to_list()
    return list_index_sorted


def show_ranking_column(dict_results, column, option, session):
    if option == "Equipos":
        # Create dataframe of column
        list_index = []
        list_column = []
        for key, value in dict_results.items():
            list_index.append(DICT_TEAMS.get(key))
            list_column.append(value[column])

        df_column = pd.DataFrame({"Equipos": list_index, column: list_column})
        df_column.sort_values(by=column, ascending=False, inplace=True)
        df_column["Posicion"] = [i + 1 for i in range(df_column.shape[0])]
        df_column = df_column[["Posicion", "Equipos", column]]

        # Show
        print(f"{column}:")
        print(tabulate(df_column, headers="keys", showindex=False, tablefmt="grid"))

        # Save table as a graph
        plot_results_table(df_column, column, session, save=True)

        # Talk de equipos concretos
        talk_of_single_team_with_ranking(df_column, "Mclaren", column)
        talk_of_single_team_with_ranking(df_column, "RedBull", column)
        talk_of_single_team_with_ranking(df_column, "Mercedes", column)
        talk_of_single_team_with_ranking(df_column, "Ferrari", column)
        talk_of_single_team_with_ranking(df_column, "AstonMartin", column)
        talk_of_single_team_with_ranking(df_column, "Williams", column)
        print("\n")


def talk_of_single_team_with_ranking(df_ranking, team, column):
    position_team = df_ranking.loc[df_ranking["Equipos"] == team, "Posicion"].iloc[0]
    column_team = df_ranking.loc[df_ranking["Equipos"] == team, column].iloc[0]

    # Check position first
    if column != "Bajo Drag":
        if position_team == 1:
            print(f"{team} es el equipo con Mayor {column}")
        elif position_team == df_ranking.shape[0]:
            print(f"{team} es el equipo con Menor {column}")
        else:
            if column_team > 0.1:
                print(f"{team} es un equipo con mejor {column} que la media de Equipos")
            elif column_team < -0.1:
                print(f"{team} es un equipo con peor {column} que la media de Equipos")
            else:
                print(f"{team} es un equipo con {column} en la media de Equipos")
    else:
        column = column.split(" ")[1]
        if position_team == 1:
            print(f"{team} es el equipo con Menor {column}")
        elif position_team == df_ranking.shape[0]:
            print(f"{team} es el equipo con Mayor {column}")
        else:
            if column_team > 0.1:
                print(f"{team} es un equipo con mejor {column} que la media de Equipos")
            elif column_team < -0.1:
                print(f"{team} es un equipo con peor {column} que la media de Equipos")
            else:
                print(f"{team} es un equipo con {column} en la media de Equipos")


def proyect_puntos(vmed, vmax, xc, yc):
    # Restar el centroide para trasladar el punto al origen
    x_prime = vmed - xc
    y_prime = vmax - yc

    # Definir los ejes rotados -45°
    eje1 = np.array([np.cos(-np.pi / 4), np.sin(-np.pi / 4)])  # Eje +45°
    eje2 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])  # Eje -45°

    # Vector del punto trasladado
    punto_vec = np.array([x_prime, y_prime])

    # Proyección escalar sobre cada eje
    proy1 = np.dot(punto_vec, eje1) * eje1
    proy2 = np.dot(punto_vec, eje2) * eje2

    # Transformar de vuelta al espacio original
    x_proy1, y_proy1 = xc + proy1
    x_proy2, y_proy2 = xc + proy2

    return (x_proy1, y_proy1), (x_proy2, y_proy2)


def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


def get_circuito(eventname):
    file_path = os.path.join(os.getcwd(), "src", "outputs", "horarios_calendario.csv")
    df_calendario = pd.read_csv(file_path, sep=",")
    return df_calendario.loc[df_calendario["Race"] == eventname, "Location"].iloc[0]


def get_corners_df(quali):
    try:
        corners = quali.get_circuit_info().corners
    except AttributeError:
        corners = []

    fastest_lap = quali.laps.pick_fastest()
    telemetry = fastest_lap.get_telemetry().add_distance()
    corners = create_corner_tableclasif(corners, telemetry)
    return corners


def get_lap_stats(quali):
    fastest_lap = quali.laps.pick_fastest()
    telemetry = fastest_lap.get_telemetry().add_distance()

    # Percentage flaout
    lap_stats = {}
    mask_flatout = telemetry["Throttle"] >= 95
    lap_stats["%_Flatout"] = (sum(mask_flatout) / telemetry.shape[0]) * 100

    # Zonas DRS
    mask_DRS_on = telemetry["DRS"] > 8
    subset_DRS = telemetry.loc[mask_DRS_on, ["Distance", "Speed", "Throttle"]]
    index_DRS_on = subset_DRS.index.tolist()
    delta_index_DRS_on = np.diff(index_DRS_on)
    index_con_cambio_mayor_de_1 = np.where(delta_index_DRS_on > 1)[0]
    lap_stats["Zonas_DRS"] = len(index_con_cambio_mayor_de_1) + 1
    return lap_stats


def get_last_year_pole_time(session_hoy, next_race):
    info = session_hoy.session_info
    year = info.get("StartDate").year - 1
    quali = ff1.get_session(year, next_race, "Q")
    quali.load()
    pole_lap = quali.laps.pick_fastest()["LapTime"]
    return pole_lap
