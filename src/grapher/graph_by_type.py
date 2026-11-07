# Librerias
import numpy as np
import pandas as pd
import fastf1 as ff1

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.text import TextPath
import seaborn as sns

from fastf1 import plotting
from fastf1 import utils
from datetime import timedelta

from src.utils.functions import format_time


def find_index_corner(distance, telemetry):
    # Hacemos una copia del Df para cañadir Dist_corner sobre distance
    aux_telemetry = telemetry.copy()
    aux_telemetry['Dist_corner'] = abs(aux_telemetry['Distance'] - distance)

    # Obtenemos el indice como idxmin()
    index_corner = aux_telemetry['Dist_corner'].idxmin()

    return index_corner


def find_index_HardBraking(corners, telemetry, corner_number):
    # 1) Buscamos el indice de la curva, de su curva anterior y de su curva siguiente
    mask_corner = corners['Number'] == corner_number
    index_df = corners.loc[mask_corner].index.tolist()
    index_df = index_df[0]
    index_corner = corners.loc[index_df, 'Index_corner']

    ## Miramos si la curva es la curva 1
    last_corner = corners.loc[corners.shape[0]-1, 'Number']
    if corner_number == 1:
        # En este caso es la primera curva
        index_beforecorner = 2
        # No es la ultima curva, ponemos el indice de curva sumando 1 al index_df
        index_aftercorner = corners.loc[index_df + 1, 'Index_corner']

    elif corner_number == last_corner:
        # No es la curva 1, restamos 1 al index_df
        index_beforecorner = corners.loc[index_df - 1, 'Index_corner']
        # Es la ultima curva, no hay curva siguiente, le asignamos el ultimo indice de la telemetria
        index_aftercorner = telemetry.shape[0] - 1

    else:
        # No es la curva 1, restamos 1 al index_df
        index_beforecorner = corners.loc[index_df - 1, 'Index_corner']
        # No es la ultima curva, ponemos el indice de curva sumando 1 al index_df
        index_aftercorner = corners.loc[index_df + 1, 'Index_corner']

    # 2) El inicio de la curva será 5 indices antes del index_inicio_lift o de index_inicio_brake
    #    el que sea menor.
    ## Buscamoe el inicio de la levantada
    subset = telemetry.loc[index_beforecorner:index_corner].copy()
    mask_lift = subset['Throttle'] < 99
    index_lift = subset.loc[mask_lift].index.tolist()

    # Miramos que no halla una recta de por medio
    delta_index_lift = np.diff(index_lift)
    index_con_cambio_mayor_de_1 = np.where(delta_index_lift > 1)[0]

    if len(index_con_cambio_mayor_de_1) > 0:
        # Hay recta de por medio
        index_con_cambio_mayor_de_1 = index_con_cambio_mayor_de_1[0]
        index_inicio_lift = index_con_cambio_mayor_de_1 + 1
        index_inicio_lift = index_lift[index_inicio_lift]
    else:
        index_inicio_lift = min(index_lift)

    ## Buscamos el indice de inicio de frenada, que sera el primer indice donde Brake == True
    mask_brake = subset['Brake'] == True
    index_brake = subset.loc[mask_brake].index.tolist()

    # Miramos que no halla una recta de por medio
    delta_index_brake = np.diff(index_brake)
    index_con_cambio_mayor_de_1 = np.where(delta_index_brake > 1)[0]

    if len(index_con_cambio_mayor_de_1) > 0:
        # Hay recta de por medio
        index_con_cambio_mayor_de_1 = index_con_cambio_mayor_de_1[0]
        index_inicio_brake = index_con_cambio_mayor_de_1 + 1
        index_inicio_brake = index_brake[index_inicio_brake]
    else:
        index_inicio_brake = min(index_brake)

    index_final_brake = max(index_brake)

    # Miramos que ocurre antes
    if index_inicio_lift < index_inicio_brake:
        index_inicio_corner = index_inicio_lift - 5
        # Checkeamos si el index_inicio_corner es negativo, lo que indicaría que la
        # primera curva esta muy al inicio de la vuelta
        if index_inicio_corner < 0:
            # Le asignnamosel primer indice de la curva
            index_inicio_corner = index_beforecorner
    else:
        index_inicio_corner = index_inicio_brake - 5
        # Checkeamos si el index_inicio_corner es negativo, lo que indicaría que la
        # primera curva esta muy al inicio de la vuelta
        if index_inicio_corner < 0:
            # Le asignnamosel primer indice de la curva
            index_inicio_corner = index_beforecorner

    # 4) Buscamos el indice de final de la curva, que será 5 indices despues del primer
    #    indice que consigue poner Throttle al maximo , tenemos que buscarlo entre el indice
    #    de la curva actual y de la curva siguiente
    subset_throttle = telemetry.loc[index_final_brake:index_aftercorner]
    mask_fullthrottle = subset_throttle['Throttle'] == 100
    index_full_throttle = subset_throttle.loc[mask_fullthrottle].index.tolist()

    if len(index_full_throttle) > 0:
        # Llega a poner acelerador al maximo
        index_inicio_fullthrottle = min(index_full_throttle)
        index_final_corner = index_inicio_fullthrottle + 5

    else:
        # No llega a poner el acelerador al maximo, tomamos el punto de throttle maximo
        mask_fullthrottle = subset_throttle['Throttle'] == max(subset_throttle['Throttle'])
        index_full_throttle = subset_throttle.loc[mask_fullthrottle].index.tolist()
        index_final_corner = index_full_throttle[0] + 5

    # Return de los valores
    return index_inicio_corner, index_final_corner


def find_index_Chicane(corners, telemetry, corner_number):
    # 1) Buscamos el indice de la curva, de su curva anterior y de su curva siguiente
    #    , la curva siguiente esta 1 index_df despues de corner_number[1]
    mask_corner1 = corners['Number'] == corner_number[0]
    mask_corner2 = corners['Number'] == corner_number[1]
    index_df1 = corners.loc[mask_corner1].index.tolist()
    index_df1 = index_df1[0]
    index_corner1 = corners.loc[index_df1, 'Index_corner']

    index_df2 = corners.loc[mask_corner2].index.tolist()
    index_df2 = index_df2[0]
    index_corner2 = corners.loc[index_df2, 'Index_corner']

    ## Miramos si la curva es la curva 1
    last_corner = corners.loc[corners.shape[0]-1, 'Number']
    if corner_number[0] == 1:
        # En este caso es la primera curva
        index_beforecorner = 2
        # No es la ultima curva, ponemos el indice de curva sumando 1 al index_df
        index_aftercorner = corners.loc[index_df2 + 1, 'Index_corner']

    else:
        # No es la curva 1, restamos 1 al index_df
        index_beforecorner = corners.loc[index_df1 - 1, 'Index_corner']
        # No es la ultima curva, ponemos el indice de curva sumando 1 al index_df
        index_aftercorner = corners.loc[index_df2 + 1, 'Index_corner']

    # 2) El inicio de la curva será 5 indices antes del index_inicio_lift o de index_inicio_brake
    #    el que sea menor.
    ## Buscamoe el inicio de la levantada
    subset = telemetry.loc[index_beforecorner:index_corner1]
    mask_lift = subset['Throttle'] < 100
    index_lift = subset.loc[mask_lift].index.tolist()
    index_inicio_lift = min(index_lift)

    ##   Buscamos el indice de inicio de frenada, que sera el primer indice donde Brake == True
    subset = telemetry.loc[index_beforecorner:index_corner1]
    mask_brake = subset['Brake'] == True
    index_brake = subset.loc[mask_brake].index.tolist()
    index_inicio_brake = min(index_brake)
    index_final_brake = max(index_brake)

    # Miramos que ocurre antes
    if index_inicio_lift < index_inicio_brake:
        index_inicio_corner = index_inicio_lift - 5
    else:
        index_inicio_corner = index_inicio_brake - 5

    ## Miramos que el punto de inicio no este a mas de 500 metros, eso indicaria se toma el valor
    ## de la otra frenada
    if telemetry.loc[index_inicio_brake,'Distance'] >= 500:
        # Lo esta cogiendo de la curva anterior, tomo la velocidad maxima en ese tramo como referencia
        mask_maxspeed = subset['Speed'] == max(subset['Speed'])
        index_max_speed = subset.loc[mask_maxspeed].index.tolist()
        index_max_speed = index_max_speed[0]

        subset2 = telemetry.loc[index_max_speed:index_corner1]
        mask_brake2 = subset2['Brake'] == True
        index_brake2 = subset2.loc[mask_brake2].index.tolist()
        index_inicio_brake = min(index_brake2)

        index_inicio_corner = index_inicio_brake - 5
        index_final_brake = max(index_brake2)

    else:
        index_inicio_corner = index_inicio_brake - 5
        index_final_brake = max(index_brake)

    # 4) Buscamos el indice de final de la curva, que será 5 indices despues del primer
    #    indice que consigue poner Throttle al maximo , tenemos que buscarlo entre el indice
    #    de la curva 2 actual y de la curva siguiente
    subset_throttle = telemetry.loc[index_corner2:index_aftercorner]
    mask_fullthrottle = subset_throttle['Throttle'] == 100
    index_full_throttle = subset_throttle.loc[mask_fullthrottle].index.tolist()

    if len(index_full_throttle) > 0:
        # Llega a poner acelerador al maximo
        index_inicio_fullthrottle = min(index_full_throttle)
        index_final_corner = index_inicio_fullthrottle + 5

    else:
        # No llega a poner el acelerador al maximo, tomamos el punto de throttle maximo
        mask_fullthrottle = subset_throttle['Throttle'] >= 99
        index_full_throttle = subset_throttle.loc[mask_fullthrottle].index.tolist()
        index_final_corner = index_full_throttle[0] + 5

    # Return de los valores
    return index_inicio_corner, index_final_corner


def graf_HardBraking(corners, telemetry, corner_number, index_inicio_corner, index_final_corner, color):
    # Tamaño del grafico
    plt.figure(figsize=(12,4))

    # Representacion de la velocidad grafico
    # --------------------------------------
    distances = telemetry.loc[index_inicio_corner:index_final_corner, 'Distance']
    speeds = telemetry.loc[index_inicio_corner:index_final_corner, 'Speed']

    plt.plot(distances, speeds, color=color)

    # Representar el area de frenada
    # ------------------------------
    subset = telemetry.loc[index_inicio_corner:index_final_corner]
    mask_brake = subset['Brake']==True
    index_brake = subset.loc[mask_brake, 'Distance'].index.tolist()

    index_inicio_brake = min(index_brake)
    index_final_brake = max(index_brake)

    frenada = telemetry.loc[index_inicio_brake:index_final_brake, 'Distance']
    plt.fill_between(frenada , min(speeds), max(speeds), color='red', alpha=0.3, label='Frenada')

    # Añadimos levantada de acelerador en la curva(si levanta el acelerador antes en la curva)
    # ----------------------------------------------------------------------------------------
    subset = telemetry.loc[index_inicio_corner:index_inicio_brake]
    mask_lift = subset['Throttle'] < 100
    index_lift = subset.loc[mask_lift].index.tolist()

    if len(index_lift) > 0:
        # Hay levantada previa a frenar, la dibujamos
        index_lift_initial = min(index_lift) - 1# Para alinearlo con el inicio de la curva
        index_lift_final = max(index_lift)
        lift = telemetry.loc[index_lift_initial:index_lift_final, 'Distance']
        plt.fill_between(lift, min(speeds), max(speeds), color='yellow', alpha=0.3, label='Lift')

    # Representar el area de aceleracion
    # ----------------------------------
    ## Buscamos el indice de inicio de aceleracion entre el index_final_brake:index_inicio_fullthrottle
    subset = telemetry.loc[index_final_brake:index_final_corner]

    mask_fullthrottle = subset['Throttle'] >= 99
    index_full_throttle = subset.loc[mask_fullthrottle].index.tolist()

    if len(index_full_throttle) > 0:
        # Se llega a poner el acelerador a fondo
        index_inicio_fullthrottle = min(index_full_throttle)
    else:
        # No llega a meter acelerador a fondo, nos quedamos con el maximo que pone
        mask_fullthrottle = subset['Throttle'] == max(subset['Throttle'])
        index_full_throttle = subset.loc[mask_fullthrottle].index.tolist()
        index_inicio_fullthrottle = index_full_throttle[0]

    subset_aceleracion = telemetry.loc[index_final_brake:index_inicio_fullthrottle]
    mask_throttle = subset_aceleracion['Throttle'] >= 1
    index_throttle = subset_aceleracion.loc[mask_throttle].index.tolist()

    aceleracion = telemetry.loc[index_throttle, 'Distance']
    plt.fill_between(aceleracion, min(speeds), max(speeds), color='green', alpha=0.3, label='Aceleración')

    # Añadimos Acelerador a fondo
    # ---------------------------
    if len(index_full_throttle) > 0:
        # Se puede graficar
        full_throttle = telemetry.loc[index_full_throttle, 'Distance']
        plt.fill_between(full_throttle, min(speeds), max(speeds), color='purple', alpha=0.3, label='Full Throttle')

    # Añadimos La franja que delimita el inicio de la curva y el final
    # ----------------------------------------------------------------
    ## Inicio -> Último punto con acelerador a fondo en la entrada de curva
    if len(index_lift) > 0:
        # Si hay levantada lo saco de lift
        index_inicio_curva = index_lift[0] - 1
        inicio_curva = telemetry.loc[index_inicio_curva, 'Distance']
    else:
        # Al no haber levantada previa, lo saco de la frenada
        index_inicio_curva = index_inicio_brake
        inicio_curva = telemetry.loc[index_inicio_curva, 'Distance']

    ## Final -> Primer punto con acelerador a fondo a la salida
    index_final_curva = index_full_throttle[0]
    final_curva = telemetry.loc[index_final_curva, 'Distance']

    plt.axvline(x=inicio_curva, ymin=0, ymax=400, linestyle='--', color='orange')# inicio de curva
    plt.axvline(x=final_curva,  ymin=0, ymax=400, linestyle='--', color='orange')# final de curva

    # Añadir información de la curva
    # ------------------------------
    vmin=min(speeds)
    ventry = telemetry.loc[index_inicio_curva,'Speed']
    vexit = telemetry.loc[index_final_curva,'Speed']

    t1 = telemetry.loc[index_inicio_curva, 'Time']
    t2 = telemetry.loc[index_final_curva, 'Time']
    delta_curva = format_time(t2 - t1)

    t1 = telemetry.loc[index_inicio_brake, 'Time']
    t2 = telemetry.loc[index_final_brake, 'Time']
    delta_frenada = format_time(t2 - t1)

    if len(index_throttle) > 1:
        t1 = telemetry.loc[min(index_throttle), 'Time']
        t2 = telemetry.loc[max(index_throttle), 'Time']
        delta_aceleracion = format_time(t2 - t1)
    else:
        delta_aceleracion = 0.0

    ## Tipo de corner type
    mask = corners['Number'] == corner_number
    corner_type = corners.loc[mask, 'Type_Corner_Speed'].values[0]

    max_distances = max(distances)
    min_distances = min(distances)
    offset = 0.07*(max_distances - min_distances)

    if corner_type == 'Slow':
        plt.text(max(distances) + offset, max(speeds), corner_type, color='red', fontsize=18,
                 fontweight='bold', verticalalignment='center')

    elif corner_type == 'Medium':
        plt.text(max(distances) + offset, max(speeds), corner_type, color='yellow', fontsize=18,
                 fontweight='bold', verticalalignment='center')

    elif corner_type == 'Fast':
        plt.text(max(distances) + offset, max(speeds), corner_type, color='purple', fontsize=18,
                 fontweight='bold', verticalalignment='center')

    ## Valores adicionales
    max_speed = max(speeds)
    min_speed = min(speeds)
    offset_speed = 0.1*(max_speed - min_speed)
    texto = (f'Velocidades relevantes:\n  '
             f'v_min = {vmin} km/h\n  '
             f'v_entry = {ventry} km/h\n  '
             f'v_exit = {vexit} km/h\n\n'
             f'Corner Time = {delta_curva}s\n'
             f'Brake Time = {delta_frenada}s\n'
             f'Acc. Time = {delta_aceleracion}s')
    plt.text(max(distances) + offset, max(speeds) - offset_speed, texto, fontsize=10, color='white',
             verticalalignment='top')

    # Añadir titulos y ejes
    # ---------------------
    plt.xlabel('Distance (m)')
    plt.ylabel('Speed (km/h)')

    plt.title('Curva {} - Hard Braking'.format(corner_number), loc='left', fontweight='bold', fontsize = 18)
    plt.legend(loc='lower left')

    # Muestro grafico
    # ---------------
    plt.show()


def graf_Chicane(corners, telemetry, corner_number, index_inicio_corner, index_final_corner, color):
    # Tamaño del grafico
    plt.figure(figsize=(12,4))

    # Representacion de la velocidad grafico
    # --------------------------------------
    distances = telemetry.loc[index_inicio_corner:index_final_corner, 'Distance']
    speeds = telemetry.loc[index_inicio_corner:index_final_corner, 'Speed']

    plt.plot(distances, speeds, color=color)

    # Representar el area de frenada
    # ------------------------------
    subset = telemetry.loc[index_inicio_corner:index_final_corner]
    mask_brake = subset['Brake']==True
    index_brake = subset.loc[mask_brake, 'Distance'].index.tolist()

    index_inicio_brake = min(index_brake)
    index_final_brake = max(index_brake)

    frenada = telemetry.loc[index_inicio_brake:index_final_brake, 'Distance']
    plt.fill_between(frenada , min(speeds), max(speeds), color='red', alpha=0.3, label='Frenada')

    # Añadimos levantada de acelerador en la curva(si levanta el acelerador antes en la curva)
    # ----------------------------------------------------------------------------------------
    subset = telemetry.loc[index_inicio_corner:index_inicio_brake]
    mask_lift = subset['Throttle'] < 99
    index_lift = subset.loc[mask_lift].index.tolist()

    if len(index_lift) > 0:
        # Hay levantada previa a frenar, la dibujamos
        index_lift_initial = min(index_lift) - 1# Para alinearlo con el inicio de la curva
        index_lift_final = max(index_lift)
        lift = telemetry.loc[index_lift_initial:index_lift_final, 'Distance']
        plt.fill_between(lift , min(speeds), max(speeds), color='yellow', alpha=0.3, label='Lift entry')

    # Representar el area de aceleracion
    # ----------------------------------
    ## Buscamos el indice de inicio de aceleracion entre el index_final_brake:index_inicio_fullthrottle
    subset = telemetry.loc[index_final_brake:index_final_corner]

    mask_fullthrottle = subset['Throttle'] >= 99
    index_full_throttle = subset.loc[mask_fullthrottle].index.tolist()

    if len(index_full_throttle) > 0:
        # Se llega a poner el acelerador a fondo
        index_inicio_fullthrottle = min(index_full_throttle)
    else:
        # No llega a meter acelerador a fondo, nos quedamos con el maximo que pone
        mask_fullthrottle = subset['Throttle'] == max(subset['Throttle'])
        index_full_throttle = subset.loc[mask_fullthrottle].index.tolist()
        index_inicio_fullthrottle = index_full_throttle[0]

    ## Ahora desde index_final_brake:index_inicio_fullthrottle buscaremos los puntos donde se acelera
    telemetry['Diff_Throttle'] = telemetry['Throttle'].diff()

    ## Añadimos las franjas verdes de aceleracion
    subset = telemetry.loc[index_final_brake:index_inicio_fullthrottle]
    mask_positivo = subset['Diff_Throttle'] > 0

    index_positivo = subset.loc[mask_positivo].index.tolist()

    if len(index_positivo) > 0:
        index_inicio = index_positivo[0]
        cont = 1
        for i in range(1, len(index_positivo)):
            if index_positivo[i] == (index_positivo[i-1]+1):
                # Son consecutivos
                index_fin = index_positivo[i]

                # Ver si estamos en el ultimo caso
                if i == (len(index_positivo) - 1):
                    index_fin = index_positivo[i] + 1
                    aceleracion = subset.loc[index_inicio:index_fin, 'Distance']
                    text = 'Aceleracion ' + str(cont)
                    plt.fill_between(aceleracion, min(speeds), max(speeds), color='green', alpha=0.3, label=text)
                    cont += 1

            else:
                # No son consecutivos
                index_fin = index_positivo[i-1] + 1
                # Nota, tambien valdría index_fin = index_fin + 1

                # Ver si estamos en el ultimo caso
                if i == (len(index_positivo) -1):
                    aceleracion = subset.loc[index_inicio:index_fin, 'Distance']
                    text = 'Aceleracion ' + str(cont)
                    plt.fill_between(aceleracion, min(speeds), max(speeds), color='green', alpha=0.3, label=text)
                    cont += 1
                else:
                    aceleracion = subset.loc[index_inicio:index_fin, 'Distance']
                    text = 'Aceleracion ' + str(cont)
                    plt.fill_between(aceleracion, min(speeds), max(speeds), color='green', alpha=0.3, label=text)
                    cont += 1

                    # Reseteamos el inicio
                    index_inicio = index_positivo[i]

    ## Añadimos las franjas amarillas de lift
    mask_negativo = subset['Diff_Throttle'] < 0
    index_negativo = subset.loc[mask_negativo].index.tolist()

    if len(index_negativo) > 0:
        index_inicio = index_negativo[0]
        cont = 1
        for i in range(1, len(index_negativo)):
            if index_negativo[i] == (index_negativo[i-1]+1):
                # Son consecutivos
                index_fin = index_negativo[i]

                # Ver si estamos en el ultimo caso
                if i == (len(index_negativo) -1):
                    index_fin = index_negativo[i] + 1
                    lift = subset.loc[index_inicio:index_fin, 'Distance']
                    text = 'Lift ' + str(cont)
                    plt.fill_between(lift, min(speeds), max(speeds), color='yellow', alpha=0.3, label=text)
                    cont += 1

            else:
                # No son consecutivos
                index_fin = index_negativo[i-1] + 1
                # Nota, tambien valdría index_fin = index_fin + 1

                # Ver si estamos en el ultimo caso
                if i == (len(index_negativo) - 1):
                    lift = subset.loc[index_inicio:index_fin, 'Distance']
                    text = 'Lift ' + str(cont)
                    plt.fill_between(lift, min(speeds), max(speeds), color='yellow', alpha=0.3, label=text)
                    cont += 1
                else:
                    lift = subset.loc[index_inicio:index_fin, 'Distance']
                    text = 'Lift ' + str(cont)
                    plt.fill_between(lift, min(speeds), max(speeds), color='yellow', alpha=0.3, label=text)
                    cont += 1

                    # Reseteamos el inicio
                    index_inicio = index_negativo[i]

    ## Eliminamos la columna Diff_Throttle
    telemetry.pop('Diff_Throttle')

    # Añadimos La franja que delimita el inicio de la curva y el final
    # ----------------------------------------------------------------
    ## Inicio -> Último punto con acelerador a fondo en la entrada de curva
    if len(index_lift) > 0:
        # Si hay levantada lo saco de lift
        index_inicio_curva = index_lift[0] - 1# Para alinearlo con el grafico
        inicio_curva = telemetry.loc[index_inicio_curva, 'Distance']
    else:
        # Al no haber levantada previa, lo saco de la frenada
        index_inicio_curva = index_inicio_brake
        inicio_curva = telemetry.loc[index_inicio_curva, 'Distance']

    ## Final -> Primer punto con acelerador a fondo a la salida
    index_final_curva = index_full_throttle[0]
    final_curva = telemetry.loc[index_final_curva, 'Distance']

    plt.axvline(x=inicio_curva, ymin=0, ymax=400, linestyle='--', color='orange')# inicio de curva
    plt.axvline(x=final_curva,  ymin=0, ymax=400, linestyle='--', color='orange')# final de curva

    # Representar los puntos de las curvas de la Chicane
    # --------------------------------------------------
    for i in range(len(corner_number)):
        mask = corners['Number'] == corner_number[i]
        distance_corner = corners.loc[mask, 'Distance'].values[0]
        plt.axvline(x=distance_corner, ymin=0, ymax=400, linestyle='--', color='black')

    # Añadimos Acelerador a fondo
    # ---------------------------
    if len(index_full_throttle) > 0:
        # Se puede graficar
        full_throttle = telemetry.loc[index_full_throttle, 'Distance']
        plt.fill_between(full_throttle, min(speeds), max(speeds), color='purple', alpha=0.3, label='Full Throttle')

    # Añadir información de la curva
    # ------------------------------
    vmin=min(speeds)
    ventry = telemetry.loc[index_inicio_curva,'Speed']
    vexit = telemetry.loc[index_final_curva,'Speed']

    t1 = telemetry.loc[index_inicio_curva, 'Time']
    t2 = telemetry.loc[index_final_curva, 'Time']
    delta_curva = format_time(t2 - t1)

    t1 = telemetry.loc[index_inicio_brake, 'Time']
    t2 = telemetry.loc[index_final_brake, 'Time']
    delta_frenada = format_time(t2 - t1)

    #t1 = telemetry.loc[min(index_throttle), 'Time']
    #t2 = telemetry.loc[max(index_throttle), 'Time']
    #delta_aceleracion = format_time(t2 - t1)

    ## Tipo de corner type
    mask = corners['Number'] == corner_number[0]
    corner_type = corners.loc[mask, 'Type_Corner_Speed'].values[0]

    max_distances = max(distances)
    min_distances = min(distances)
    offset = 0.07*(max_distances - min_distances)

    if corner_type == 'Slow':
        plt.text(max(distances) + offset, max(speeds), corner_type, color='red', fontsize=18,
                 fontweight='bold', verticalalignment='center')

    elif corner_type == 'Medium':
        plt.text(max(distances) + offset, max(speeds), corner_type, color='yellow', fontsize=18,
                 fontweight='bold', verticalalignment='center')

    elif corner_type == 'Fast':
        plt.text(max(distances) + offset, max(speeds),corner_type, color='purple', fontsize=18,
                 fontweight='bold', verticalalignment='center')

    max_speed = max(speeds)
    min_speed = min(speeds)
    offset_speed = 0.1*(max_speed - min_speed)

    texto = (f'Velocidades relevantes:\n  '
             f'v_min = {vmin} km/h\n  '
             f'v_entry = {ventry} km/h\n  '
             f'v_exit = {vexit} km/h\n\n'
             f'Corner Time = {delta_curva}s\n'
             f'Brake Time = {delta_frenada}s')
    plt.text(max(distances) + offset, max(speeds) - offset_speed, texto, fontsize=10, color='white',
             verticalalignment='top')

    # Añadir titulos y ejes
    # ---------------------
    plt.xlabel('Distance (m)')
    plt.ylabel('Speed (km/h)')
    plt.title('Curvas {} - Chicane'.format(corner_number), loc='left', fontweight='bold', fontsize=18)
    plt.legend(loc='lower left')

    # Muestro grafico
    # ---------------
    plt.show()


def find_index_Scorner(corners, telemetry, corner_number):
    # 1) Buscamos el indice de la curva, de su curva anterior y de su curva siguiente
    #    , la curva siguiente esta 1 index_df despues de corner_number[1]
    mask_corner_first = corners['Number'] == corner_number[0]
    mask_corner_last = corners['Number'] == corner_number[len(corner_number)-1]
    index_df_first = corners.loc[mask_corner_first].index.tolist()
    index_df_first = index_df_first[0]
    index_corner_first = corners.loc[index_df_first, 'Index_corner']

    index_df_last = corners.loc[mask_corner_last].index.tolist()
    index_df_last = index_df_last[0]
    index_corner_last = corners.loc[index_df_last, 'Index_corner']

    ## Miramos si la curva es la curva 1
    last_corner = corners.loc[corners.shape[0]-1, 'Number']
    if corner_number[0] == 1:
        # En este caso es la primera curva
        index_beforecorner = 2
        # No es la ultima curva, ponemos el indice de curva sumando 1 al index_df
        index_aftercorner = corners.loc[index_df_last + 1, 'Index_corner']

    else:
        # No es la curva 1, restamos 1 al index_df
        index_beforecorner = corners.loc[index_df_first - 1, 'Index_corner']
        # No es la ultima curva, ponemos el indice de curva sumando 1 al index_df
        index_aftercorner = corners.loc[index_df_last + 1, 'Index_corner']

    # 2) El inicio de la curva será 5 indices antes del index_inicio_brake.
    ##   Buscamos el indice de inicio de frenada, que sera el primer indice donde Brake == True
    ##   Sino cogeremos un punto a 50metoros antes de la curva 1
    subset = telemetry.loc[index_beforecorner:index_corner_first]
    mask_brake = subset['Brake'] == True
    index_brake = subset.loc[mask_brake].index.tolist()

    ## Miramos que se frene antes de entrar antes de la curva, algo que puede no se el caso
    if len(index_brake) > 0:
        index_inicio_brake = min(index_brake)
        index_inicio_corner = index_inicio_brake - 5
    else:
        ### Se toma la 1ª curva de la S a fondo => Tomamos como punto de inicio, uno a 50m antes
        distance_corner = telemetry.loc[index_corner_first, 'Distance']
        distance_50mbefore = distance_corner - 50

        telemetry['DiffDistance'] = abs(telemetry['Distance']-distance_50mbefore)
        index_inicio_corner = telemetry['DiffDistance'].idxmin()
        telemetry.pop('DiffDistance')

    # 4) Buscamos el indice de final de la curva, que será 5 indices despues del primer
    #    indice que consigue poner Throttle al maximo , tenemos que buscarlo entre el indice
    #    de la curva 2 actual y de la curva siguiente
    subset_throttle = telemetry.loc[index_corner_last:index_aftercorner]
    mask_fullthrottle = subset_throttle['Throttle'] >= 99
    index_full_throttle = subset_throttle.loc[mask_fullthrottle].index.tolist()

    if len(index_full_throttle) > 0:
        # Llega a poner acelerador al maximo
        index_inicio_fullthrottle = min(index_full_throttle)

        # Miramos que el indice no sea el de la ultima curva de la S
        if index_inicio_fullthrottle == index_corner_last:
            # En ese caso al pasarse toda la curva final a fondo, cogemos el punto a 50m
            # mas adelante
            distance_corner = telemetry.loc[index_corner_last, 'Distance']
            distance_50mafter = distance_corner + 50
            telemetry['DiffDistance'] = abs(telemetry['Distance'] - distance_50mafter)
            index_final_corner = telemetry['DiffDistance'].idxmin()
            telemetry.pop('DiffDistance')

        else:
            index_final_corner = index_inicio_fullthrottle + 5

    else:
        # No llega a poner el acelerador al maximo, tomamos el punto de throttle maximo
        mask_fullthrottle = subset_throttle['Throttle'] == max(subset_throttle['Throttle'])
        index_full_throttle = subset_throttle.loc[mask_fullthrottle].index.tolist()
        index_final_corner = index_full_throttle[0] + 5

    # Return de los valores
    return index_inicio_corner, index_final_corner

def graf_Scorner(corners, telemetry, corner_number, index_inicio_corner, index_final_corner, color):
    # Tamaño del grafico
    plt.figure(figsize=(12,4))

    # Representacion de la velocidad grafico
    # --------------------------------------
    distances = telemetry.loc[index_inicio_corner:index_final_corner, 'Distance']
    speeds = telemetry.loc[index_inicio_corner:index_final_corner, 'Speed']

    plt.plot(distances, speeds, color=color)

    # Representacion de frenada si la hubiera
    # ---------------------------------------
    ## Recorremos las curvas que conforman la S, veremos si en cada transición
    ## de curva a curva hay o no frenada.
    for i in range(len(corner_number)):
        if i==0:
            # Primera curva
            mask_corner = corners['Number'] == corner_number[i]
            index_corner = corners.loc[mask_corner, 'Index_corner'].tolist()
            index_corner = index_corner[0]
            index_corner_anterior = index_corner

            subset = telemetry.loc[index_inicio_corner:index_corner]
            mask_brake = subset['Brake'] == True
            index_brake = subset.loc[mask_brake, 'Distance'].index.tolist()

            if len(index_brake) > 0:
                # Hay frenada
                index_inicio_brake = min(index_brake)
                index_final_brake = max(index_brake)

                frenada = telemetry.loc[index_inicio_brake:index_final_brake, 'Distance']
                plt.fill_between(frenada , min(speeds), max(speeds), color='red', alpha=0.3, label='Frenada')

        else:
            # No es la primera curva
            mask_corner = corners['Number'] == corner_number[i]
            index_corner = corners.loc[mask_corner, 'Index_corner'].tolist()
            index_corner = index_corner[0]
            index_corner_anterior = index_corner

            subset = telemetry.loc[index_corner_anterior:index_corner]
            mask_brake = subset['Brake'] == True
            index_brake = subset.loc[mask_brake, 'Distance'].index.tolist()

            if len(index_brake) > 0:
                # Hay frenada (NOTA: Ver como )
                index_inicio_brake = min(index_brake)
                index_final_brake = max(index_brake)

                frenada = telemetry.loc[index_inicio_brake:index_final_brake, 'Distance']
                plt.fill_between(frenada, min(speeds), max(speeds), color='red', alpha=0.3, label='Frenada')

    # Representacion de la aceleracion si la hubiera
    # ----------------------------------------------
    for i in range(len(corner_number)):
        if i == (len(corner_number)-1):
            # Es el ultimo caso, la aceleracion sería entre la posición de la ultima curvay la posicion final
            mask_corner = corners['Number'] == corner_number[i]
            index_corner = corners.loc[mask_corner, 'Index_corner'].tolist()
            index_corner = index_corner[0]

            subset = telemetry.loc[index_corner:index_final_corner]
            mask_throttle = subset['Throttle'] > 0
            mask_not_fullthrottle = subset['Throttle'] < 99
            index_aceleracion = subset.loc[mask_throttle & mask_not_fullthrottle].index.tolist()

            # Chequeamos que haya caso de aceleración
            if len(index_aceleracion) > 0:
                # Hay aceleracion => Lo graficamos
                aceleracion = telemetry.loc[index_aceleracion, 'Distance']
                plt.fill_between(aceleracion, min(speeds), max(speeds), color='green', alpha=0.3, label='Aceleración')

        else:
            # Calculamos el indice de la curva actual
            mask_corner = corners['Number'] == corner_number[i]
            index_corner = corners.loc[mask_corner, 'Index_corner'].tolist()
            index_corner = index_corner[0]

            # Ahora para la curva siguiente
            mask_corner_next = corners['Number'] == corner_number[i+1]
            index_corner_next = corners.loc[mask_corner_next, 'Index_corner'].tolist()
            index_corner_next = index_corner_next[0]

            subset = telemetry.loc[index_corner:index_corner_next]
            mask_throttle = subset['Throttle'] > 0
            mask_not_fullthrottle = subset['Throttle'] < 99
            index_aceleracion = subset.loc[mask_throttle & mask_not_fullthrottle].index.tolist()

            # Chequeamos que haya caso de aceleración
            if len(index_aceleracion) > 0:
                # Hay aceleracion => Lo graficamos
                aceleracion = telemetry.loc[index_aceleracion, 'Distance']
                plt.fill_between(aceleracion, min(speeds), max(speeds), color='green', alpha=0.3, label='Aceleración')

    # Representacion de la levantada si la hubiera
    # --------------------------------------------
    for i in range(len(corner_number)):
        if i==0:
            # Primera curva
            mask_corner = corners['Number'] == corner_number[i]
            index_corner = corners.loc[mask_corner, 'Index_corner'].tolist()
            index_corner = index_corner[0]
            index_corner_anterior = index_corner

            subset = telemetry.loc[index_inicio_corner:index_corner]
            mask_lift = subset['Throttle'] < 99
            index_lift = subset.loc[mask_lift].index.tolist()

            if len(index_lift) > 0:
                # Hay levantada previa a frenar, la dibujamos
                lift = telemetry.loc[index_lift, 'Distance']
                plt.fill_between(lift, min(speeds), max(speeds), color='yellow', alpha=0.3, label='Lift')

        else:
            # No es primera curva
            mask_corner = corners['Number'] == corner_number[i]
            index_corner = corners.loc[mask_corner, 'Index_corner'].tolist()
            index_corner = index_corner[0]
            index_corner_anterior = index_corner

            subset = telemetry.loc[index_corner_anterior:index_corner]
            mask_lift = subset['Throttle'] < 99
            index_lift = subset.loc[mask_lift].index.tolist()

            if len(index_lift) > 0:
                # Hay levantada previa a frenar, la dibujamos
                lift = telemetry.loc[index_lift, 'Distance']
                plt.fill_between(lift, min(speeds), max(speeds), color='yellow', alpha=0.3, label='Lift')

    # Representacion de la aceleracion a fondo
    # ----------------------------------------------
    ## Primero miramos que en toda la seccion de curvas la aceleración este afondo
    subset = telemetry.loc[index_inicio_corner:index_final_corner]
    mask_full_throttle = subset['Throttle'] >= 99
    index_full_throttle = subset.loc[mask_full_throttle].index.tolist()

    len_section = index_final_corner - index_inicio_corner + 1

    if len(index_full_throttle) == len_section:
        # Pasa toda la sección a fondo
        full_throttle = telemetry.loc[index_full_throttle, 'Distance']
        plt.fill_between(full_throttle, min(speeds), max(speeds), color='purple', alpha=0.3, label='Full Throttle')

    else:
        # No pasa toda la sección a fondo, hay que recorrer de curva a curva
        cont = 1
        for i in range(len(corner_number) + 1):
            if i==0:
                # Es el cacho de index_inicio_corner:index_corner
                mask_corner = corners['Number'] == corner_number[i]
                index_corner = corners.loc[mask_corner, 'Index_corner'].tolist()
                index_corner = index_corner[0]
                index_corner_anterior = index_corner

                subset = telemetry.loc[index_inicio_corner:index_corner]
                mask_full_throttle = subset['Throttle'] >= 99
                index_full_throttle = subset.loc[mask_full_throttle].index.tolist()

                if len(index_full_throttle) > 0:
                    # Se llega a poner el aceleracion a fondo
                    full_throttle = telemetry.loc[index_full_throttle, 'Distance']
                    text = 'Full Throttle ' + str(cont)
                    plt.fill_between(full_throttle, min(speeds), max(speeds), color='purple', alpha=0.3, label=text)
                    cont += 1

            elif i==len(corner_number):
                # Es el ultimo termino, es el trozo desde index_corner_anterior:index_final_corner
                subset = telemetry.loc[index_corner_anterior:index_final_corner]
                mask_full_throttle = subset['Throttle'] >= 99
                index_full_throttle = subset.loc[mask_full_throttle].index.tolist()

                if len(index_full_throttle) > 0:
                    # Se llega a poner el aceleracion a fondo
                    full_throttle = telemetry.loc[index_full_throttle, 'Distance']
                    text = 'Full Throttle ' + str(cont)
                    plt.fill_between(full_throttle, min(speeds), max(speeds), color='purple', alpha=0.3, label=text)
                    cont += 1

            else:
                # Esta entre curva y curva
                mask_corner = corners['Number'] == corner_number[i]
                index_corner = corners.loc[mask_corner, 'Index_corner'].tolist()
                index_corner = index_corner[0]
                index_corner_anterior = index_corner

                subset = telemetry.loc[index_corner_anterior:index_corner]
                mask_full_throttle = subset['Throttle'] >= 99
                index_full_throttle = subset.loc[mask_full_throttle].index.tolist()

                if len(index_full_throttle) > 0:
                    # Se llega a poner el aceleracion a fondo
                    full_throttle = telemetry.loc[index_full_throttle, 'Distance']
                    text = 'Full Throttle ' + str(cont)
                    plt.fill_between(full_throttle, min(speeds), max(speeds), color='purple', alpha=0.3, label=text)
                    cont += 1

    # Representacion de los limites de la curva en S
    # ----------------------------------------------
    # Seran los index_inicio_corner y index_final_corner
    inicio_curva = telemetry.loc[index_inicio_corner, 'Distance']
    final_curva = telemetry.loc[index_final_corner, 'Distance']

    plt.axvline(x=inicio_curva, ymin=0, ymax=400, linestyle='--', color='orange')# inicio de curva
    plt.axvline(x=final_curva,  ymin=0, ymax=400, linestyle='--', color='orange')# final de curva

    # Ahora añadimos los puntos que indican cada curva
    for i in range(len(corner_number)):
        mask = corners['Number'] == corner_number[i]
        distance_corner = corners.loc[mask, 'Distance'].values[0]
        plt.axvline(x=distance_corner, ymin=0, ymax=400, linestyle='--', color='black')

    # Añadir información de la curva
    # ------------------------------
    vmin = min(speeds)
    ventry = telemetry.loc[index_inicio_corner, 'Speed']
    vexit = telemetry.loc[index_final_corner, 'Speed']

    t1 = telemetry.loc[index_inicio_corner, 'Time']
    t2 = telemetry.loc[index_final_corner, 'Time']
    delta_curva = format_time(t2 - t1)

    ## Tipo de corner type
    mask = corners['Number'] == corner_number[0]
    corner_type = corners.loc[mask, 'Type_Corner_Speed'].values[0]

    max_distances = max(distances)
    min_distances = min(distances)
    offset = 0.07*(max_distances - min_distances)

    if corner_type == 'Slow':
        plt.text(max(distances) + offset, max(speeds), corner_type, color='red', fontsize=18,
                 fontweight='bold', verticalalignment='center')

    elif corner_type == 'Medium':
        plt.text(max(distances) + offset, max(speeds), corner_type, color='yellow', fontsize=18,
                 fontweight='bold', verticalalignment='center')

    elif corner_type == 'Fast':
        plt.text(max(distances) + offset, max(speeds), corner_type, color='purple', fontsize=18,
                 fontweight='bold', verticalalignment='center')

    max_speed = max(speeds)
    min_speed = min(speeds)
    offset_speed = 0.1*(max_speed - min_speed)

    texto = (f'Velocidades relevantes:\n  '
             f'v_min = {vmin} km/h\n  '
             f'v_entry = {ventry} km/h\n  '
             f'v_exit = {vexit} km/h\n\n'
             f'Corner Time = {delta_curva}s')
    plt.text(max(distances) + offset, max(speeds) - offset_speed, texto, fontsize=10, color='white',
             verticalalignment='top')

    # Añadir titulos y ejes
    # ---------------------
    plt.xlabel('Distance (m)')
    plt.ylabel('Speed (km/h)')
    plt.title('Curvas {} - S corner'.format(corner_number), loc='left', fontweight='bold', fontsize = 18)
    plt.legend(loc='best')

    # Muestro grafico
    # ---------------
    plt.show()


def find_index_LittleBraking(corners, telemetry, corner_number):
    # 1) Buscamos el indice de la curva, de su curva anterior y de su curva siguiente
    mask_corner = corners['Number'] == corner_number
    index_df = corners.loc[mask_corner].index.tolist()
    index_df = index_df[0]
    index_corner = corners.loc[index_df, 'Index_corner']

    ## Miramos si la curva es la curva 1
    last_corner = corners.loc[corners.shape[0]-1, 'Number']
    if corner_number == 1:
        # En este caso es la primera curva
        index_beforecorner = 2
        # No es la ultima curva, ponemos el indice de curva sumando 1 al index_df
        index_aftercorner = corners.loc[index_df + 1, 'Index_corner']

    elif corner_number == last_corner:
        # No es la curva 1, restamos 1 al index_df
        index_beforecorner = corners.loc[index_df - 1, 'Index_corner']

        # Es la ultima curva, no hay curva siguiente, le asignamos el ultimo indice de la telemetria
        index_aftercorner = telemetry.shape[0]+1

    else:
        # No es la curva 1, restamos 1 al index_df
        index_beforecorner = corners.loc[index_df - 1, 'Index_corner']
        # No es la ultima curva, ponemos el indice de curva sumando 1 al index_df
        index_aftercorner = corners.loc[index_df + 1, 'Index_corner']

    # 2) El inicio de la curva será 5 indices antes del index_inicio_lift o de index_inicio_brake
    #    el que sea menor.
    subset = telemetry.loc[index_beforecorner:index_corner]

    ## Miramos que haya acelerador a fondo
    mask_full = subset['Throttle'] >= 99
    index_full = subset.loc[mask_full].index.tolist()

    if len(index_full) > 0:
        ### Hay acelerador a fondo
        ## Buscamos el inicio de la levantada
        mask_lift = subset['Throttle'] < 99
        index_lift = subset.loc[mask_lift].index.tolist()
        index_inicio_lift = min(index_lift)

        ## Buscamos el indice de inicio de frenada, que sera el primer indice donde Brake == True
        subset = telemetry.loc[index_beforecorner:index_corner]
        mask_brake = subset['Brake'] == True
        index_brake = subset.loc[mask_brake].index.tolist()
        index_inicio_brake = min(index_brake)
        index_final_brake = max(index_brake)

        # Miramos que ocurre antes
        if index_inicio_lift < index_inicio_brake:
            index_inicio_corner = index_inicio_lift - 5
        else:
            index_inicio_corner = index_inicio_brake - 5

    else:
        # No se pone el acelerador a fondo entre la curva anterior y la actual
        # Buscamos el inicio de la frenada y lo tomaremos como extremo mayor de busqueda
        subset_brake = telemetry.loc[index_beforecorner:index_corner]
        mask_brake = subset_brake['Brake'] == True
        index_brake = subset_brake.loc[mask_brake].index.tolist()
        index_inicio_brake = min(index_brake)
        index_final_brake = max(index_brake)

        # Ahora buscamos el indice de acelerador maximo entre index_beforecorner:index_inicio_brake
        subset2 = telemetry.loc[index_beforecorner:index_inicio_brake]
        index_maxthrottle = subset2['Throttle'].idxmax()

        index_inicio_corner = index_maxthrottle - 5

    # 4) Buscamos el indice de final de la curva, que será 5 indices despues del primer
    #    indice que consigue poner Throttle al maximo , tenemos que buscarlo entre el indice
    #    de la curva actual y de la curva siguiente
    subset_throttle = telemetry.loc[index_final_brake:index_aftercorner]
    mask_fullthrottle = subset_throttle['Throttle'] >= 99
    index_full_throttle = subset_throttle.loc[mask_fullthrottle].index.tolist()

    if len(index_full_throttle) > 0:
        # Llega a poner acelerador al maximo
        index_inicio_fullthrottle = min(index_full_throttle)
        index_final_corner = index_inicio_fullthrottle + 5

    else:
        # No llega a poner el acelerador al maximo, tomamos el punto de throttle maximo
        mask_fullthrottle = subset_throttle['Throttle'] == max(subset_throttle['Throttle'])
        index_full_throttle = subset_throttle.loc[mask_fullthrottle].index.tolist()
        index_final_corner = index_full_throttle[0] + 5

    # Return de los valores
    return index_inicio_corner, index_final_corner

def graf_LittleBraking(corners, telemetry, corner_number, index_inicio_corner, index_final_corner, color):
    # Tamaño del grafico
    plt.figure(figsize=(12,4))

    # Representacion de la velocidad grafico
    # --------------------------------------
    distances = telemetry.loc[index_inicio_corner:index_final_corner, 'Distance']
    speeds = telemetry.loc[index_inicio_corner:index_final_corner, 'Speed']

    plt.plot(distances, speeds, color = color)

    # Representar el area de frenada
    # ------------------------------
    subset = telemetry.loc[index_inicio_corner:index_final_corner]
    mask_brake = subset['Brake'] == True
    index_brake = subset.loc[mask_brake, 'Distance'].index.tolist()

    index_inicio_brake = min(index_brake)
    index_final_brake = max(index_brake)

    frenada = telemetry.loc[index_inicio_brake:index_final_brake, 'Distance']
    plt.fill_between(frenada, min(speeds), max(speeds), color='red', alpha=0.3, label='Frenada')

    # Añadimos levantada de acelerador en la curva(si levanta el acelerador antes en la curva)
    # ----------------------------------------------------------------------------------------
    subset = telemetry.loc[index_inicio_corner:index_inicio_brake]

    ## Miramos que se venga de acelerador al 100 antes de la frenada, ya que es punto delimita la curva
    maximo_throttle = max(subset['Throttle'])

    if maximo_throttle == 100:
        # Viene de acelerador al maximo
        mask_lift = subset['Throttle'] < 99
        index_lift = subset.loc[mask_lift].index.tolist()

        if len(index_lift) > 0:
            # Hay levantada previa a frenar, la dibujamos
            index_lift_initial = min(index_lift) - 1# Para alinearlo con el inicio de la curva
            index_lift_final = max(index_lift)
            lift = telemetry.loc[index_lift_initial:index_lift_final, 'Distance']
            plt.fill_between(lift , min(speeds), max(speeds), color='yellow', alpha=0.3, label='Lift')

    else:
        # No viene de acelerador al maximo, marcamos el punto en donde va a acelerador maximo
        index_max_throttle = subset['Throttle'].idxmax()

        if (index_inicio_brake - index_max_throttle) > 0:
            # Hay punto de levantada entre el punto de acelerador al maximo y el inicio de la frenada
            index_lift = list(range(index_max_throttle, index_inicio_brake + 1))
            lift = telemetry.loc[index_lift, 'Distance']
            plt.fill_between(lift, min(speeds), max(speeds), color='yellow', alpha=0.3, label='Lift')

    # Representar el area de aceleracion
    # ----------------------------------
    ## Buscamos el indice de inicio de aceleracion entre el index_final_brake:index_inicio_fullthrottle
    subset = telemetry.loc[index_final_brake:index_final_corner]

    mask_fullthrottle = subset['Throttle'] >= 99
    index_full_throttle = subset.loc[mask_fullthrottle].index.tolist()

    if len(index_full_throttle) > 0:
        # Se llega a poner el acelerador a fondo
        index_inicio_fullthrottle = min(index_full_throttle)
    else:
        # No llega a meter acelerador a fondo, nos quedamos con el maximo que pone
        mask_fullthrottle = subset['Throttle'] == max(subset['Throttle'])
        index_full_throttle = subset.loc[mask_fullthrottle].index.tolist()
        index_inicio_fullthrottle = index_full_throttle[0]

    subset_aceleracion = telemetry.loc[index_final_brake:index_inicio_fullthrottle]
    mask_throttle = subset_aceleracion['Throttle'] >= 1
    index_throttle = subset_aceleracion.loc[mask_throttle].index.tolist()

    aceleracion = telemetry.loc[index_throttle, 'Distance']
    plt.fill_between(aceleracion, min(speeds), max(speeds), color='green', alpha=0.3, label='Aceleración')

    # Añadimos Acelerador a fondo
    # ---------------------------
    if len(index_full_throttle) > 0:
        # Se puede graficar
        full_throttle = telemetry.loc[index_full_throttle, 'Distance']
        plt.fill_between(full_throttle, min(speeds), max(speeds), color='purple', alpha=0.3, label='Full Throttle')

    # Añadimos La franja que delimita el inicio de la curva y el final
    # ----------------------------------------------------------------
    ## Inicio -> Último punto con acelerador a fondo en la entrada de curva
    if len(index_lift) > 0:
        # Vemos si se llega de acelerador a fondo antes de la frenada
        subset = telemetry.loc[index_inicio_corner:index_inicio_brake]
        maximo_throttle = max(subset['Throttle'])

        if maximo_throttle == 100:
            # Viene de maximo acelerador, lo tomo desde lift
            index_inicio_curva = index_lift[0] - 1
            inicio_curva = telemetry.loc[index_inicio_curva, 'Distance']

        else:
            # No viene de acelerador a 100, lo coloco en el punto de maximo acelerador
            index_max_throttle = subset['Throttle'].idxmax()
            index_inicio_curva = index_max_throttle
            inicio_curva = telemetry.loc[index_inicio_curva, 'Distance']

    else:
        # Al no haber levantada previa, lo saco de la frenada
        index_inicio_curva = index_inicio_brake
        inicio_curva = telemetry.loc[index_inicio_curva, 'Distance']

    ## Final -> Primer punto con acelerador a fondo a la salida
    index_final_curva = index_full_throttle[0]
    final_curva = telemetry.loc[index_final_curva, 'Distance']

    plt.axvline(x=inicio_curva, ymin=0, ymax=400, linestyle='--', color='orange')# inicio de curva
    plt.axvline(x=final_curva,  ymin=0, ymax=400, linestyle='--', color='orange')# final de curva

    # Añadir información de la curva
    # ------------------------------
    vmin = min(speeds)
    ventry = telemetry.loc[index_inicio_curva, 'Speed']
    vexit = telemetry.loc[index_final_curva, 'Speed']

    t1 = telemetry.loc[index_inicio_curva, 'Time']
    t2 = telemetry.loc[index_final_curva, 'Time']
    delta_curva = format_time(t2 - t1)

    t1 = telemetry.loc[index_inicio_brake, 'Time']
    t2 = telemetry.loc[index_final_brake, 'Time']
    delta_frenada = format_time(t2 - t1)

    t1 = telemetry.loc[min(index_throttle), 'Time']
    t2 = telemetry.loc[max(index_throttle), 'Time']
    delta_aceleracion = format_time(t2 - t1)

    ## Tipo de corner type
    mask = corners['Number'] == corner_number
    corner_type = corners.loc[mask, 'Type_Corner_Speed'].values[0]

    max_distances = max(distances)
    min_distances = min(distances)
    offset = 0.07*(max_distances - min_distances)

    if corner_type == 'Slow':
       plt.text(max(distances) + offset, max(speeds), corner_type, color='red', fontsize=18,
                fontweight='bold', verticalalignment='center')

    elif corner_type == 'Medium':
        plt.text(max(distances) + offset, max(speeds), corner_type, color='yellow', fontsize=18,
                 fontweight='bold', verticalalignment='center')

    elif corner_type == 'Fast':
        plt.text(max(distances) + offset, max(speeds), corner_type, color='purple', fontsize=18,
                 fontweight='bold', verticalalignment='center')

    max_speed = max(speeds)
    min_speed = min(speeds)
    offset_speed = 0.1*(max_speed - min_speed)

    texto = (f'Velocidades relevantes:\n  '
             f'v_min = {vmin} km/h\n  '
             f'v_entry = {ventry} km/h\n  '
             f'v_exit = {vexit} km/h\n\n'
             f'Corner Time = {delta_curva}s\n'
             f'Brake Time = {delta_frenada}s\n'
             f'Acc. Time = {delta_aceleracion}s')
    plt.text(max(distances) + offset, max(speeds) - offset_speed, texto, fontsize=10, color='white',
             verticalalignment='top')

    # Añadir titulos y ejes
    # ---------------------
    plt.xlabel('Distance (m)')
    plt.ylabel('Speed (km/h)')
    plt.title('Curva {} - Little Braking '.format(corner_number), loc='left', fontweight='bold', fontsize=18)
    plt.legend(loc='best')

    # Muestro grafico
    # ---------------
    plt.show()


def find_index_Flatout(corners, telemetry, corner_number):
    # 1) Buscamos el indice de la curva
    mask_corner = corners['Number'] == corner_number
    index_df = corners.loc[mask_corner].index.tolist()
    index_df = index_df[0]
    index_corner = corners.loc[index_df, 'Index_corner']

    # 2) El inicio de la curva será 75m antes del punto de la curva
    distance_corner = telemetry.loc[index_corner, 'Distance']
    distance_ini = distance_corner - 75

    telemetry['DiffDistance'] = abs(telemetry['Distance'] - distance_ini)
    index_inicio_corner = telemetry['DiffDistance'].idxmin()

    telemetry.pop('DiffDistance')

    # 3) El final de la curva será 75m despues del punto de la curva
    distance_fin = distance_corner + 75

    telemetry['DiffDistance'] = abs(telemetry['Distance'] - distance_fin)
    index_final_corner = telemetry['DiffDistance'].idxmin()

    telemetry.pop('DiffDistance')

    # Return de los valores
    return index_inicio_corner, index_final_corner

def graf_Flatout(corners, telemetry, corner_number, index_inicio_corner, index_final_corner, color):
    # Tamaño del grafico
    plt.figure(figsize=(12,4))

    # Representacion de la velocidad grafico
    # --------------------------------------
    distances = telemetry.loc[index_inicio_corner:index_final_corner, 'Distance']
    speeds = telemetry.loc[index_inicio_corner:index_final_corner, 'Speed']

    plt.plot(distances, speeds, color = color)

    # Añadimos la franja de acelerador a fondo
    # -----------------------------------------
    ## Primero miramos que en toda la seccion de curvas la aceleración este afondo
    subset = telemetry.loc[index_inicio_corner:index_final_corner]
    mask_full_throttle = subset['Throttle'] >= 99
    index_full_throttle = subset.loc[mask_full_throttle].index.tolist()

    len_section = index_final_corner - index_inicio_corner + 1

    if len(index_full_throttle) == len_section:
        # Pasa toda la sección a fondo
        full_throttle = telemetry.loc[index_full_throttle, 'Distance']
        plt.fill_between(full_throttle, min(speeds), max(speeds), color='purple', alpha=0.3, label='Full Throttle')

    else:
        # Entonces vamos a dividir el caso entre inicio curva:indice_curva y indice_curva:final:curva
        mask = corners['Number'] == corner_number
        index_corner = corners.loc[mask, 'Index_corner'].values[0]

        subset1 = telemetry.loc[index_inicio_corner:index_corner]
        mask_full_throttle1 = subset1['Throttle'] >= 99
        index_full_throttle1 = subset1.loc[mask_full_throttle1].index.tolist()

        if len(index_full_throttle1) > 0:
            # Hay acelerador a fondo
            full_throttle1 = telemetry.loc[index_full_throttle1, 'Distance']
            plt.fill_between(full_throttle1, min(speeds), max(speeds), color='purple',
                             alpha=0.3, label='Full Throttle 1')

        subset2 = telemetry.loc[index_corner:index_final_corner]
        mask_full_throttle2 = subset2['Throttle'] >= 99
        index_full_throttle2 = subset2.loc[mask_full_throttle2].index.tolist()

        if len(index_full_throttle2) > 0:
            # Hay acelerador a fondo
            full_throttle2 = telemetry.loc[index_full_throttle2, 'Distance']
            plt.fill_between(full_throttle2, min(speeds), max(speeds), color='purple',
                             alpha=0.3, label='Full Throttle 2')

    # Añadimos los puntos de inicio de la curva
    # -----------------------------------------
    inicio_curva = telemetry.loc[index_inicio_corner,'Distance']

    ## Para el final de curva, vamos a ver si final index_final_curva, el acelerador esta a fondo,
    ## sino buscamos el ultimo
    ## punto con acelerador a fondo, probablemente se junte con otra curva
    if telemetry.loc[index_final_corner,'Throttle'] < 99:
        # No termina con acelerador a fondo
        # Miramos que se llegue a poner el acelerador a fondo en la salida de la curva
        if len(index_full_throttle2) > 0:
            index_final_curva = max(index_full_throttle2)
            final_curva = telemetry.loc[index_final_curva, 'Distance']
        else:
            # No se llega a poner el acelerador al maximo, pongo como final de la curva el index_final_corner
            index_final_curva = index_final_corner
            final_curva = telemetry.loc[index_final_curva, 'Distance']
    else:
        # El ultimo punto esta con el acelerador a fondo, sera nuestro indice de final de curva
        index_final_curva = index_final_corner
        final_curva = telemetry.loc[index_final_curva, 'Distance']

    plt.axvline(x=inicio_curva, ymin=0, ymax=400, linestyle='--', color='orange')# inicio de curva
    plt.axvline(x=final_curva,  ymin=0, ymax=400, linestyle='--', color='orange')# final de curva

    # Añadimos las franjas de acelaracion y de lift
    # ---------------------------------------------
    telemetry['Diff_Throttle'] = telemetry['Throttle'].diff()
    subset = telemetry.loc[index_inicio_corner:index_final_curva]

    ## Añadimos las franjas verdes de aceleracion
    mask_positivo = subset['Diff_Throttle'] > 0
    mask_notfullthrottle = subset['Throttle'] < 99
    index_positivo = subset.loc[mask_positivo & mask_notfullthrottle].index.tolist()

    if len(index_positivo) > 0:
        index_inicio = index_positivo[0]
        cont = 1
        for i in range(1, len(index_positivo)):
            if index_positivo[i] == (index_positivo[i-1]+1):
                # Son consecutivos
                index_fin = index_positivo[i]

                # Ver si estamos en el ultimo caso
                if i == (len(index_positivo) - 1):
                    index_fin = index_positivo[i] + 1
                    aceleracion = subset.loc[index_inicio:index_fin, 'Distance']
                    text = 'Aceleracion ' + str(cont)
                    plt.fill_between(aceleracion, min(speeds), max(speeds), color='green', alpha=0.3, label=text)
                    cont += 1

            else:
                # No son consecutivos
                index_fin = index_positivo[i-1] + 1
                # Nota, tambien valdría index_fin = index_fin + 1

                # Ver si estamos en el ultimo caso
                if i == (len(index_positivo) - 1):
                    aceleracion = subset.loc[index_inicio:index_fin, 'Distance']
                    text = 'Aceleracion ' + str(cont)
                    plt.fill_between(aceleracion, min(speeds), max(speeds), color='green', alpha=0.3, label=text)
                    cont += 1
                else:
                    aceleracion = subset.loc[index_inicio:index_fin, 'Distance']
                    text = 'Aceleracion ' + str(cont)
                    plt.fill_between(aceleracion, min(speeds), max(speeds), color='green', alpha=0.3, label=text)
                    cont += 1

                    # Reseteamos el inicio
                    index_inicio = index_positivo[i]

    ## Añadimos las franjas amarillas de lift
    mask_negativo = subset['Diff_Throttle'] < 0
    index_negativo = subset.loc[mask_negativo & mask_notfullthrottle].index.tolist()

    if len(index_negativo) > 0:
        index_inicio = index_negativo[0]
        cont = 1
        for i in range(1, len(index_negativo)):
            if index_negativo[i] == (index_negativo[i-1]+1):
                # Son consecutivos
                index_fin = index_negativo[i]

                # Ver si estamos en el ultimo caso
                if i == (len(index_negativo) - 1):
                    index_fin = index_negativo[i] + 1
                    lift = subset.loc[index_inicio:index_fin, 'Distance']
                    text = 'Lift ' + str(cont)
                    plt.fill_between(lift, min(speeds), max(speeds), color='yellow', alpha=0.3, label=text)
                    cont += 1

            else:
                # No son consecutivos
                index_fin = index_negativo[i-1] + 1
                # Nota, tambien valdría index_fin = index_fin + 1

                # Ver si estamos en el ultimo caso
                if i == (len(index_negativo) - 1):
                    lift = subset.loc[index_inicio:index_fin, 'Distance']
                    text = 'Lift ' + str(cont)
                    plt.fill_between(lift, min(speeds), max(speeds), color='yellow', alpha=0.3, label=text)
                    cont += 1
                else:
                    lift = subset.loc[index_inicio:index_fin, 'Distance']
                    text = 'Lift ' + str(cont)
                    plt.fill_between(lift, min(speeds), max(speeds), color='yellow', alpha=0.3, label=text)
                    cont += 1

                    # Reseteamos el inicio
                    index_inicio = index_negativo[i]

    ## Eliminamos la columna Diff_Throttle
    telemetry.pop('Diff_Throttle')

    # Añadimos el punto de la curva en negro
    # --------------------------------------
    mask = corners['Number'] == corner_number
    distance_corner = corners.loc[mask, 'Distance'].values[0]
    plt.axvline(x=distance_corner, ymin=0, ymax=400, linestyle='--', color='black')

    # Añadir información de la curva
    # ------------------------------
    vmin = min(speeds)
    ventry = telemetry.loc[index_inicio_corner, 'Speed']
    vexit = telemetry.loc[index_final_curva, 'Speed']

    t1 = telemetry.loc[index_inicio_corner, 'Time']
    t2 = telemetry.loc[index_final_curva, 'Time']
    delta_curva = format_time(t2 - t1)

    ## Tipo de corner type
    mask = corners['Number'] == corner_number
    corner_type = corners.loc[mask, 'Type_Corner_Speed'].values[0]

    max_distances = max(distances)
    min_distances = min(distances)
    offset = 0.07*(max_distances - min_distances)

    if corner_type == 'Slow':
        plt.text(max(distances) + offset, max(speeds), corner_type, color='red', fontsize=18,
                 fontweight='bold', verticalalignment='center')

    elif corner_type == 'Medium':
        plt.text(max(distances) + offset, max(speeds), corner_type, color='yellow', fontsize=18,
                 fontweight='bold', verticalalignment='center')

    elif corner_type == 'Fast':
        plt.text(max(distances) + offset, max(speeds), corner_type, color='purple', fontsize=18,
                 fontweight='bold', verticalalignment='center')

    max_speed = max(speeds)
    min_speed = min(speeds)
    offset_speed = 0.1*(max_speed - min_speed)

    texto = (f'Velocidades relevantes:\n  '
             f'v_min = {vmin} km/h\n  '
             f'v_entry = {ventry} km/h\n  '
             f'v_exit = {vexit} km/h\n\n'
             f'Corner Time = {delta_curva}s')
    plt.text(max(distances) + offset, max(speeds) - offset_speed, texto, fontsize=10, color='white',
             verticalalignment='top')

    # Añadir titulos y ejes
    # ---------------------
    plt.xlabel('Distance (m)')
    plt.ylabel('Speed (km/h)')
    plt.title('Curva {} - Flat out'.format(corner_number), loc='left', fontweight='bold', fontsize = 18)
    plt.legend(loc='best')

    # Muestro grafico
    # ---------------
    plt.show()

def graf_corners_onedriver(corners, telemetry, color):
    # 1) Añadimos a corners el indice que corresponde cada curva
    corners['Index_corner'] = corners['Distance'].apply(lambda x: find_index_corner(x, telemetry))

    # 2) Creamos una lista con los typos de curva sin repetir
    unique_corners = []
    unique_corners = corners['Type_Corner'].unique()
    corner_order = corners.groupby('Type_Corner')['Number'].min().sort_values().index
    unique_corners = sorted(unique_corners, key=lambda x: corner_order.get_loc(x))

    # 3) Recorremos cada tipo de curva unico para representar cada curva, segun su tipo
    for i in range(len(unique_corners)):
        ## Extraemos el tipo de curva, estructura = Tipo_Numero
        tipo_curva = unique_corners[i].split('_')[0]

        ## Extraemos los numeros de curva que estamos representando, puede ser uno o varios si es curva compuesta(chicane, Scorner)
        mask_number = corners['Type_Corner'] == unique_corners[i]
        corner_number = corners.loc[mask_number, 'Number'].tolist()

        ## Segmentamos por tipo de curva
        if tipo_curva == 'HardBraking':
            corner_number = corner_number[0]

            ### Buscamos los indices de inicio de curva y final, sabiendo que es HardBraking
            index_inicio_corner, index_final_corner = find_index_HardBraking(corners, telemetry, corner_number)

            ### Graficamos la curva HardBraking
            #graf_HardBraking(corners, telemetry, corner_number, index_inicio_corner, index_final_corner, color)

        elif tipo_curva == 'LittleBraking':
            corner_number = corner_number[0]

            ### Buscamos los indices de inicio de curva y final, sabiendo que es HardBraking
            index_inicio_corner, index_final_corner = find_index_LittleBraking(corners, telemetry, corner_number)

            ### Graficamos la curva HardBraking
            #graf_LittleBraking(corners, telemetry, corner_number, index_inicio_corner, index_final_corner, color)

        elif tipo_curva == 'Chicane':
            ### corner_number es una lista de dos elementos
            ### Buscamos los indices de inicio de curva y final, sabiendo que es Chicane
            index_inicio_corner, index_final_corner = find_index_Chicane(corners, telemetry, corner_number)

            ### Graficamos la Chicane
            graf_Chicane(corners, telemetry, corner_number, index_inicio_corner, index_final_corner, color)

        elif tipo_curva == 'Scorner':
            ### corner_number es una lista de tres elementos o mas
            ### Buscamos los indices de inicio de curva y final, sabiendo que es Chicane
            index_inicio_corner, index_final_corner = find_index_Scorner(corners, telemetry, corner_number)

            ### Graficamos la Chicane
            graf_Scorner(corners, telemetry, corner_number, index_inicio_corner, index_final_corner, color)

        elif tipo_curva == 'Flatout':
            ### corner_number tiene un elemento
            corner_number = corner_number[0]

            ### Buscamos los indices de la curva
            index_inicio_corner, index_final_corner = find_index_Flatout(corners, telemetry, corner_number)

            ### Graficamos la Flatout
            graf_Flatout(corners, telemetry, corner_number, index_inicio_corner, index_final_corner, color)

        else:
            print('Error, Tipo de curva: {} introducido NO es valido'.format(unique_corners[i]))

    # Quitamos la columna Index_corner
    corners.pop('Index_corner')


def calcula_recta(telemetry):
    # Buscamos las zonas de DRS y sus indices
    mask_DRS_driver1 = telemetry['DRS'] > 8
    mask_throttle_driver1 = telemetry['Throttle'] >= 99
    subset_DRS = telemetry.loc[mask_DRS_driver1 & mask_throttle_driver1, ['Distance', 'Speed', 'Time']]
    index_DRS_on = subset_DRS.index.tolist()

    # Vemos los indices de cambio de recta
    delta_index = np.diff(index_DRS_on)
    index_cambio_recta = np.where(delta_index > 1)[0]

    # Indices de rectas
    index_rectas = []
    index_rectas.append(index_DRS_on[0])  # Primer caso

    for cambio in index_cambio_recta:
        index_rectas.append(index_DRS_on[cambio])  # Cambios de recta
        index_rectas.append(index_DRS_on[cambio + 1])  # Indice siguiente recta

    index_rectas.append(index_DRS_on[-1])  # Ultimo caso

    return index_rectas


def find_recta_comun(subset1, subset2):
    # Copiamos los subsets para hacer calculos y no afectar al resto
    subset_d1 = subset1.copy()
    subset_d2 = subset2.copy()

    # INICIO DE LA RECTA
    # ------------------
    # Extraemos la primera distancia de cada piloto
    dist_ini_d1 = subset_d1.loc[subset_d1.index[0], 'Distance']
    dist_ini_d2 = subset_d2.loc[subset_d2.index[0], 'Distance']

    # Buscar la distancia mayor
    dist_ini_mayor = 0
    if dist_ini_d1 > dist_ini_d2:
        dist_ini_mayor = dist_ini_d1
        idx_ini_d1 = subset_d1.index[0]

        # Creamos una nueva columna en el otro piloto con la diff entre Distance y dist_mayor en valor abs
        subset_d2['Diff_dist_ini'] = abs(subset_d2['Distance'] - dist_ini_mayor)

        # Buscamos el indice min
        idx_ini_d2 = subset_d2['Diff_dist_ini'].idxmin()

    else:
        dist_ini_mayor = dist_ini_d2
        idx_ini_d2 = subset_d2.index[0]

        # Creamos una nueva columna en el otro piloto con la diff entre Distance y dist_mayor en valor abs
        subset_d1['Diff_dist_ini'] = abs(subset_d1['Distance'] - dist_ini_mayor)

        # Buscamos el indice min
        idx_ini_d1 = subset_d1['Diff_dist_ini'].idxmin()

    # FINAL DE LA RECTA
    # -----------------
    # Extraemos la ultima distancia de cada piloto
    dist_fin_d1 = subset_d1.loc[subset_d1.index[-1], 'Distance']
    dist_fin_d2 = subset_d2.loc[subset_d2.index[-1], 'Distance']

    # Buscamos la distancia menor de ambas
    if dist_fin_d1 > dist_fin_d2:
        dist_fin_menor = dist_fin_d2
        idx_fin_d2 = subset_d2.index[-1]

        # Creamos una nueva columna en el otro piloto con la diff entre Distance y dist_menor en valor abs
        subset_d1['Diff_dist_fin'] = abs(subset_d1['Distance'] - dist_fin_menor)

        # Buscamos el indice menor de esta columna
        idx_fin_d1 = subset_d1['Diff_dist_fin'].idxmin()

    else:
        dist_fin_menor = dist_fin_d1
        idx_fin_d1 = subset_d1.index[-1]

        # Creamos una nueva columna en el otro piloto con la diff entre Distance y dist_menor en valor abs
        subset_d2['Diff_dist_fin'] = abs(subset_d2['Distance'] - dist_fin_menor)

        # Buscamos el indice menor de esta columna
        idx_fin_d2 = subset_d2['Diff_dist_fin'].idxmin()

    # Hacemos el subset para ambos pilotos y los devolvemos
    #print("D1 Index = {}, {}".format(idx_ini_d1, idx_fin_d1))
    #print("D2 Index = {}, {}\n".format(idx_ini_d2, idx_fin_d2))
    subset1 = subset1.loc[idx_ini_d1:idx_fin_d1]
    subset2 = subset2.loc[idx_ini_d2:idx_fin_d2]

    return subset1, subset2


def graph_analisis_rectasDRS(telemetry_driver_1, fastest_driver_1, telemetry_driver_2, fastest_driver_2):
    # Calculamos el color
    team_driver = fastest_driver_1['Team']
    color_driver1 = ff1.plotting.team_color(team_driver)
    driver1 = fastest_driver_1['Driver']

    team_driver = fastest_driver_2['Team']
    color_driver2 = ff1.plotting.team_color(team_driver)
    driver2 = fastest_driver_2['Driver']

    # Buscamos las zonas de DRS y sus indices
    index_rectas_d1 = calcula_recta(telemetry_driver_1)
    index_rectas_d2 = calcula_recta(telemetry_driver_2)

    # Recorrer las rectas
    n_rectas = int(len(index_rectas_d1) / 2)
    recta = 0

    for i in range(0, n_rectas):
        # Tamaño del grafico
        plt.figure(figsize=(14, 6))

        # Calculamos los indices de las curvas
        index_inicio_recta = index_rectas_d1[recta]
        index_final_recta = index_rectas_d1[recta + 1]

        index_inicio_recta_d2 = index_rectas_d2[recta]
        index_final_recta_d2 = index_rectas_d2[recta + 1]
        #print('index_inicio = {}, index_final = {}'.format(index_inicio_recta, index_final_recta))

        # Paso al siguiente indice
        recta += 2

        # Generamos el subset para representar
        subset_d1 = telemetry_driver_1.loc[index_inicio_recta:index_final_recta, ['Distance', 'Speed', 'Time']]
        subset_d2 = telemetry_driver_2.loc[index_inicio_recta_d2:index_final_recta_d2, ['Distance', 'Speed', 'Time']]
        #print("Driver 1")
        #print(subset_d1)
        #print("Driver 2")
        #print(subset_d2)
        # Hacemos que ambos pilotos coincidan en la recta
        subset1, subset2 = find_recta_comun(subset_d1, subset_d2)

        # Plot de distance vs Speed
        plt.plot(subset1['Distance'], subset1['Speed'], color=color_driver1)
        plt.plot(subset2['Distance'], subset2['Speed'], color=color_driver2)

        inicio_recta = subset1.loc[subset1.index[0], 'Distance']
        final_recta = subset1.loc[subset1.index[-1], 'Distance']
        plt.axvline(x=inicio_recta, ymin=0, ymax=400, linestyle='--', color='orange')  # inicio de recta
        plt.axvline(x=final_recta, ymin=0, ymax=400, linestyle='--', color='orange')  # final de recta

        # Texto con la info de la recta
        vmax = max(subset1['Speed'])
        vmin = min(subset1['Speed'])
        #ventry = subset.loc[index_inicio_recta, 'Speed']
        #vexit = subset.loc[index_final_recta, 'Speed']

        offset = 0.07 * (final_recta - inicio_recta)
        offset_speed = 0.1 * (vmax - vmin)

        t1 = subset1.loc[subset1.index[0], 'Time']
        t2 = subset1.loc[subset1.index[-1], 'Time']
        delta_recta = format_time(t2 - t1)

        t1 = subset2.loc[subset2.index[0], 'Time']
        t2 = subset2.loc[subset2.index[-1], 'Time']
        delta_recta_d2 = format_time(t2 - t1)

        if delta_recta < delta_recta_d2:
            texto = (f'Tiempos en Recta:\n'
                     f'{driver1} Fastest\n'
                    # f'v_max = {vmax} km/h\n'
                   #  f'v_entry = {ventry} km/h\n'
                   #  f'v_exit = {vexit} km/h\n\n'
                     f'{driver1} = {delta_recta}s\n'
                     f'{driver2} = {delta_recta_d2}')

        else:
            texto = (f'Tiempos en Recta:\n'
                     f'{driver2} Fastest\n'
                     # f'v_max = {vmax} km/h\n'
                     #  f'v_entry = {ventry} km/h\n'
                     #  f'v_exit = {vexit} km/h\n\n'
                     f'{driver2} = {delta_recta_d2}s\n'
                     f'{driver1} = {delta_recta}')


        plt.text(max(subset1['Distance']) + offset, max(subset1['Speed']) - offset_speed, texto,
                 fontsize=10, color='white', verticalalignment='top')

        # Añadir titulos y ejes
        # ---------------------
        plt.xlabel('Distance (m)')
        plt.ylabel('Speed (km/h)')
        plt.title('Recta DRS {}'.format(i+1), loc='left', fontweight='bold', fontsize=18)

        plt.show()

