import os
import re
import json
import shutil
import numpy as np
import pandas as pd
import fastf1 as ff1

from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader
from pdf2image import convert_from_path

from src.s3_manager.s3_functions import upload_file_s3, download_file_s3
from src.common.const import *


def convert_and_save_in_pngs(input_path, output_path, name):
    # Convertir PDF a imagen (alta resolución con 300 DPI)
    images = convert_from_path(input_path, dpi=300)

    # Checkear si name empiza por Telemetria
    if name.split("_")[0] == "Telemetria":
        # Seleccionar la primera página del PDF y convertirla a una imagen de PIL
        image = images[0]

        # Obtener las dimensiones originales de la imagen
        width, height = image.size

        # Definir el recorte: (izquierda, arriba, derecha, abajo)
        crop_area = (0, 0, width, 1718)

        # Recortar la imagen
        cropped_image = image.crop(crop_area)

        # Guardar la imagen recortada como PNG
        outputname = name + '.png'
        path = os.path.join(output_path, outputname)
        cropped_image.save(path, 'PNG')

        # Guardar la imagen gloabla como PNG
        outputname = name + '_Total.png'
        path = os.path.join(output_path, outputname)
        images[0].save(path, 'PNG')

    else:
        # Guardar la imagen como PNG
        outputname = name + '.png'
        path = os.path.join(output_path, outputname)
        images[0].save(path, 'PNG')


def move_png(input_path, path_imagenes, name):
    output_path = os.path.join(path_imagenes, name)
    shutil.copy(input_path, output_path)


def convert_and_save_in_png(input_path, output_path, name):
    # Convert img in png
    inputname = name + ".pdf"
    file_path = os.path.join(input_path, inputname)
    image = convert_from_path(file_path, dpi=300)

    # Save in png
    outputname = name + ".png"
    file_path = os.path.join(output_path, outputname)
    image[0].save(file_path, 'PNG')


def update_clasification(df_mundial_pilotos, df_mundial_constructores, session):
    # Common 
    path_output = os.path.join(os.getcwd(), "src", "outputs")

    # STATS DEL PILOTO
    # Lectura
    df_stats_pilotos = pd.read_csv(os.path.join(path_output, 'stats_pilotos.csv'), sep=",")
    columns_stats = df_stats_pilotos.columns.to_list()
    columns_stats = [col for col in columns_stats if col != "abreviatura"]

    # EXTRACT CONTEXTO
    ## Mundial de Pilotos
    list_drivers_dict = []
    for i in range(df_mundial_pilotos.shape[0]):
        # Extract
        posicion = df_mundial_pilotos["posicion"].iloc[i]
        nombre = df_mundial_pilotos["piloto"].iloc[i].split(" ")[0]
        apellido = df_mundial_pilotos["piloto"].iloc[i].split(" ")[1]
        equipo = df_mundial_pilotos["equipo"].iloc[i]

        nacionalidad_img = DICT_NACIONALIDAD_IMG[df_mundial_pilotos["nacionalidad"].iloc[i]]
        nacionalidad = DICT_NACIONALIDAD[df_mundial_pilotos["nacionalidad"].iloc[i]]
        puntos = df_mundial_pilotos["puntos"].iloc[i]
        abreviatura = df_mundial_pilotos["abreviatura"].iloc[i]
        if abreviatura == "DOO":
            color = ff1.plotting.get_driver_color('GAS', session)
        else:
            color = ff1.plotting.get_driver_color(abreviatura, session)

        # Stats
        subset_piloto = df_stats_pilotos.loc[df_stats_pilotos["abreviatura"] == abreviatura,].copy()
        dict_stats = {}
        for col in columns_stats:
            dict_stats[col] = subset_piloto[col].iloc[0]

        # Create dict
        dict_aux = {
            "posicion": posicion,
            "abreviatura":abreviatura,
            "nombre": nombre,
            "apellido": apellido,
            "equipo": equipo,
            "nacionalidad_img": nacionalidad_img,
            "nacionalidad": nacionalidad,
            "puntos": puntos,
            "color": color,
            "filas_stats_piloto": dict_stats
        }

        list_drivers_dict.append(dict_aux)

    ## Mundial de Constructores
    # STATS DEL CONSTRUCTORES
    # Lectura
    df_stats_constructores = pd.read_csv(os.path.join(path_output, ('stats_constructores.csv')), sep=",")
    columns_stats_constructores = df_stats_constructores.columns.to_list()
    columns_stats_constructores = [col for col in columns_stats_constructores if col != "equipo"]
    
    list_constructores_dict = []
    for i in range(df_mundial_constructores.shape[0]):
        # Extract
        posicion = df_mundial_constructores["posicion"].iloc[i]
        equipo = df_mundial_constructores["equipo"].iloc[i]
        puntos = df_mundial_constructores["puntos"].iloc[i]
        color = color = ff1.plotting.get_team_color(equipo, session)

        # Stats constructores
        subset_constructores = df_stats_constructores.loc[df_stats_constructores['equipo']==equipo,].copy()
        dict_auxiliar = {}
        for col in columns_stats_constructores:
            dict_auxiliar[col] = subset_constructores[col].iloc[0]
        dict_auxiliar["puntos"] = df_mundial_constructores["puntos"].iloc[i]

        # Create dict
        dict_aux = {
            "posicion": posicion,
            "equipo": equipo,
            "color": color,
            "logo_name": f"{equipo}2",
            "puntos": puntos,
            "filas_stats_constructor": dict_auxiliar
        }

        list_constructores_dict.append(dict_aux)

    # CONTEXTO FINAL
    contexto = {
        "filas_pilotos": list_drivers_dict,
        "filas_constructores": list_constructores_dict
    }

    # CARGAR EL TEMPLATE DE LA PLANTILLA
    creators_path = os.path.join(os.getcwd(), "src", "creators")
    env = Environment(loader=FileSystemLoader(creators_path))
    template = env.get_template('plantilla_clasificacion.html')

    # RENDERIZAR DOCUMENT0
    html_documento = template.render(contexto)

    # SAVE
    filepath = os.path.join(os.getcwd(), "src", "html", "clasificacion.html")
    print(filepath)
    with open(filepath, "w") as file:
        file.write(html_documento)

    # 4) Subir al S3 el nueva clasificacion 
    s3_path = ""
    upload_file_s3(filepath, s3_path)


def create_race_sesion(session, df_results):
    # EXTRACT CONTEXTO
    list_dict = []

    ## Category 
    eventname = session.event.EventName.split(" ")[0]
    race_doc = f"{eventname.lower()}.html"
    dict_category = {
        "color" : DICT_EVENT[eventname]["color"],
        "category": DICT_EVENT[eventname]["category"],
        "year": session.event.year,
        "race_doc": race_doc
    }
    
    ## Results Table
    for i in range(df_results.shape[0]):
        # Extract
        posicion = df_results["posicion"].iloc[i]
        piloto = df_results["piloto"].iloc[i]
        equipo = df_results["equipo"].iloc[i]
        cambio = df_results["cambios"].iloc[i]

        # Check nacionalidad
        if df_results["nacionalidad"].iloc[i] == '':
            nacionalidad = DICT_NACIONALIDAD_IMG[DICT_NACIONALIDAD_DRIVER[df_results["Driver"].iloc[i]]]
        else:
            nacionalidad = DICT_NACIONALIDAD_IMG[df_results["nacionalidad"].iloc[i]]

        # Tiempo
        tiempo = df_results["tiempo"].iloc[i]

        # Cambiar cambio de +1
        symbol = cambio[0]
        if len(cambio) > 1:
            value = int(cambio[1:])
            if symbol == "+":
                # Suma posicion, se va para abajo 
                cambio = f"↓{value}"
                color = "red"
            elif symbol == "-":
                # Pierde posicion, se va para arriba 
                cambio = f"↑{value}"
                color = "green"
        else:
            cambio = f"={cambio}"
            color = "gray"
        
        # Create dict
        dict_aux = {
            "posicion": posicion,
            "cambios": cambio,
            "color": color,
            "piloto": piloto,
            "equipo": equipo,
            "nacionalidad": nacionalidad,
            "tiempo": tiempo
        }

        list_dict.append(dict_aux)

    # Crear subdirectorio con el nombre del pais_año para guardar los outputs
    eventname = session.event.EventName.split(" ")[0]
    path = f"{eventname}{session.event.year}"
    path = os.path.join(os.getcwd(), "src", "html", path)

    # Crear el subdirectorio de GP
    if not os.path.exists(path):
        os.makedirs(path)

    # Crear el subdirectorio de imagenes/race
    path_imagenes = os.path.join(path, "imagenes")
    path_imagenes = os.path.join(path_imagenes, "race")

    if not os.path.exists(path_imagenes):
        os.makedirs(path_imagenes)

    # COMPARATIVAS Y OTROS
    # Leer las ficheros que hay
    eventname = session.event.EventName.split(" ")[0]
    folder = f"{session.event.year}_{eventname}"
    path_outputs = os.path.join(os.getcwd(), "src", "outputs", folder, session.name)
    list_files = os.listdir(path_outputs)
    list_files = [file for file in list_files if file != ".DS_Store"]  # Quitar DS_Store
    list_files = [os.path.splitext(file)[0] for file in list_files]  # Quitarextension

    list_comparativas = []
    list_otros = []
    comienzo = f"{session.event.year} {session.event.EventName} {session.name} "
    comienzo_adicional = "Distancia en carrera "
    comienzo_race_data = "Race data "
    for filename in list_files:
        filename_troceado = filename.split("_")
        filename_joined = " ".join(filename_troceado)

        if filename_joined.startswith(comienzo):
            # Quitar comienzo
            resto_string = filename_joined[len(comienzo):]

            # Tomar solo los que  empiezan por distancia de carrera
            if resto_string.startswith(comienzo_adicional):
                # Quitar comienzo adicional
                resto_string = resto_string[len(comienzo_adicional):]

                # Splitear y listar por " y "
                resto_spliteado = resto_string.split(" y ")

                # Replazar a cada elemento, los espacios " " por "_"
                resto_spliteado = [elemento.replace(" ", "_") for elemento in resto_spliteado]

                # extraer driver1 y driver2
                driver1 = resto_spliteado[0]
                driver2 = resto_spliteado[1]

                # Append a lista de comparativas
                list_comparativas.append({"driver1": driver1, "driver2": driver2})

                # Save png
                file = filename + ".pdf"
                input_path = os.path.join(path_outputs, file)
                nombre = f"Delta_{driver1}_{driver2}"
                convert_and_save_in_pngs(input_path, path_imagenes, nombre)

            else:
                if "vs" in filename_troceado:
                    # Extract drivers
                    driver1 = filename_troceado[-3]
                    driver2 = filename_troceado[-1]

                    # Save png
                    file = filename + ".pdf"
                    input_path = os.path.join(path_outputs, file)
                    nombre = f"Comparativa_{driver1}_{driver2}"
                    convert_and_save_in_pngs(input_path, path_imagenes, nombre)

        elif filename_joined.startswith(comienzo_race_data):
           pass
        else:
            # Son otros formatos
            if filename_troceado[0] == "Meteorologia":
                list_otros.append({"nombre": "Meteorologia"})
                nombre = filename_troceado[0]

                # Save as png
                file = filename + ".pdf"
                input_path = os.path.join(path_outputs, file)
                convert_and_save_in_pngs(input_path, path_imagenes, nombre)

            elif filename_troceado[0] == "Sectors":
                tipo = filename_troceado[-1]
                nombre = f"Mejores_Sectores_{tipo}"
                list_otros.append({"nombre": nombre})

            elif filename_troceado[0][:-2] == "Speed":
                tipo = filename_troceado[-2:]
                nombre = f"Mejores_Velocidades_{tipo}"
                list_otros.append({"nombre": nombre})

            elif filename_troceado[0] == "Ritmos" and filename_troceado[1] != "degradacion":
                tipo = filename_troceado[2]
                nombre = f"Ritmos_Equipos_{tipo}"
                list_otros.append({"nombre": nombre})

                # Save as png
                file = filename + ".pdf"
                input_path = os.path.join(path_outputs, file)
                convert_and_save_in_pngs(input_path, path_imagenes, nombre)

            elif filename_troceado[0] == "Position":
                if filename_troceado[1] == "changes":
                    nombre = filename
                    list_otros.append({"nombre": nombre})

                    # Save as png
                    file = filename + ".pdf"
                    input_path = os.path.join(path_outputs, file)
                    convert_and_save_in_pngs(input_path, path_imagenes, nombre)
                else:
                    pass

            elif filename_troceado[0] == "Levantada":
                nombre = f"Levantada_RectaPrincipal"
                list_otros.append({"nombre": nombre})

                # Save as png
                file = filename + ".pdf"
                input_path = os.path.join(path_outputs, file)
                convert_and_save_in_pngs(input_path, path_imagenes, nombre)

            elif filename_troceado[0] == "Tiempos":
                if filename_troceado[1] == "Reaccion":
                    nombre = f"Tiempos_Reaccion"
                    list_otros.append({"nombre": nombre})

                    # Save as png
                    file = filename + ".pdf"
                    input_path = os.path.join(path_outputs, file)
                    convert_and_save_in_pngs(input_path, path_imagenes, nombre)
            elif filename_troceado[0] == "Race":
                if filename_troceado[1] == "Start":
                    nombre = filename
                    list_otros.append({"nombre": nombre})

                    # Save as png
                    file = filename + ".pdf"
                    input_path = os.path.join(path_outputs, file)
                    convert_and_save_in_pngs(input_path, path_imagenes, nombre)

    # Creacion del contexto
    contexto = {
        "race": dict_category,
        "filas_pilotos": list_dict,
        "filas_comparativas": list_comparativas,
        "filas_otros": list_otros
    }

    # CARGAR EL TEMPLATE DE LA PLANTILLA
    plantilla_path = os.path.join(os.getcwd(), "src", "creators", "PlantillaGP")
    env = Environment(loader=FileSystemLoader(plantilla_path))
    template = env.get_template('plantilla_race.html')

    # RENDERIZAR DOCUMENT0
    html_documento = template.render(contexto)

    # SAVE
    path = os.path.join(path, "race.html")
    with open(path, "w") as file:
        file.write(html_documento)


def create_race_document(session):
    # CONTEXT
    eventname = session.event.EventName.split(" ")[0]

    # color, category and name
    color = DICT_EVENT[eventname]["color"],
    category = DICT_EVENT[eventname]["category"]
    name = DICT_EVENT[eventname]["name"]
    folder = f"{eventname}{session.event.year}"

    # Read articles list of articles to count 
    list_articulos = []
    path_html = os.path.join(os.getcwd(), "src", "html")
    path_outputs = os.path.join(path_html, folder)
    list_files = os.listdir(path_outputs)
    list_files = [file for file in list_files if file.endswith("html")]

    for file in list_files:
        list_articulos.append({"document": file, "sesion": file.split(".")[0]})

    # Creacion del contexto
    contexto = {
        "nombre": name,
        "race_color": color,
        "category": category,
        "grand_prix": folder,
        "filas_articulos": list_articulos
    }

    # CARGAR EL TEMPLATE DE LA PLANTILLA
    creators_path = os.path.join(os.getcwd(), "src", "creators")
    env = Environment(loader=FileSystemLoader(creators_path))
    template = env.get_template('plantilla_sesion.html')

    # RENDERIZAR DOCUMENT0
    html_documento = template.render(contexto)

    # SAVE
    path = os.path.join(path_html, f"{eventname.lower()}.html")
    with open(path, "w") as file:
        file.write(html_documento)


def create_qualy_sesion(session, df_results):
    # CREATE STRUCTURE
    # Crear subdirectorio con el nombre del pais_año para guardar los outputs
    eventname = session.event.EventName.split(" ")[0]
    path = f"{eventname}{session.event.year}"
    path = os.path.join(os.getcwd(), "src", "html", path)

    # Crear el subdirectorio de NombreGPAño
    if not os.path.exists(path):
        os.makedirs(path)

    # Añadir directorio de imagenes y imagenes/qualy
    path_imagenes = os.path.join(path, "imagenes")
    if not os.path.exists(path_imagenes):
        os.makedirs(path_imagenes)

    path_imagenes = os.path.join(path_imagenes, "qualy")
    if not os.path.exists(path_imagenes):
        os.makedirs(path_imagenes)

    # RESULTADOS SESION
    ## Category 
    eventname = session.event.EventName.split(" ")[0]
    qualy_doc = f"{eventname.lower()}.html"
    dict_category = {
        "color" : DICT_EVENT[eventname]["color"],
        "category": DICT_EVENT[eventname]["category"],
        "year": session.event.year,
        "qualy_doc": qualy_doc,
    }

    list_dict = []

    for i in range(df_results.shape[0]):
        # Extract
        posicion = df_results["posicion"].iloc[i]
        piloto = df_results["piloto"].iloc[i]
        equipo = df_results["equipo"].iloc[i]

        if df_results["nacionalidad"].iloc[i] == '':
            nacionalidad = DICT_NACIONALIDAD_IMG[DICT_NACIONALIDAD_DRIVER[df_results["Driver"].iloc[i]]]
        else:
            nacionalidad = DICT_NACIONALIDAD_IMG[df_results["nacionalidad"].iloc[i]]

        vuelta = df_results["LapTime"].iloc[i]
        gap = df_results["Gap"].iloc[i]

        # Create dict
        dict_aux = {
            "posicion": posicion,
            "piloto": piloto,
            "equipo": equipo,
            "nacionalidad": nacionalidad,
            "vuelta": vuelta,
            "gap": gap
        }

        list_dict.append(dict_aux)

    # COMPARATIVAS y OTROS
    # Leer las ficheros que hay
    eventname = session.event.EventName.split(" ")[0]
    folder = f"{session.event.year}_{eventname}"
    path_outputs = os.path.join(os.getcwd(), "src", "outputs", folder, session.name)
    list_files = os.listdir(path_outputs)
    list_files = [file for file in list_files if file != ".DS_Store"] # Quitar DS_Store
    list_files_with_extensions = list_files
    list_files = [os.path.splitext(file)[0] for file in list_files] # Quitarextension

    list_comparativas = []
    list_otros = []
    list_minisectores = []
    comienzo = f"{session.event.year} {session.event.EventName} {session.name} "
    for i, filename in enumerate(list_files):
        filename_troceado = filename.split("_")
        filename_joined = " ".join(filename_troceado)

        if filename_joined.startswith(comienzo):
            if filename_troceado[-1] == "Minisectores":
                # Quitar final
                filename_joined_new = " ".join(filename_troceado[:-1])

                # Quitar comienzo
                resto_string = filename_joined_new[len(comienzo):]

                # Splitear y listar por " vs "
                resto_spliteado = resto_string.split(" vs ")

                # Replazar a cada elemento, los espacios " " por "_"
                resto_spliteado = [elemento.replace(" ", "_") for elemento in resto_spliteado]

                # extraer driver1 y driver2
                driver1 = resto_spliteado[0]
                driver2 = resto_spliteado[1]

                # Append a lista de comparativas
                list_minisectores.append({"driver1": driver1, "driver2": driver2})

                # Transformar a .png y guardar en imagenes/qualy
                extension = os.path.splitext(list_files_with_extensions[i])[1]
                if extension == ".png":
                    # Move
                    input_path = os.path.join(path_outputs, list_files_with_extensions[i])
                    move_png(input_path, path_imagenes, list_files_with_extensions[i])
                   
                elif extension == ".pdf":
                    # Convert to pdf
                    file = filename + ".pdf"
                    input_path = os.path.join(path_outputs, file)
                    nombre = f"Minisectores_{driver1}_{driver2}"
                    convert_and_save_in_pngs(input_path, path_imagenes, nombre)
                else:
                    print(f"Extension: {extension} of file: {list_files_with_extensions[i]} NOT Supported")
            else:
                # Quitar comienzo
                resto_string = filename_joined[len(comienzo):]

                # Splitear y listar por " vs "
                resto_spliteado = resto_string.split(" vs ")

                # Replazar a cada elemento, los espacios " " por "_"
                resto_spliteado = [elemento.replace(" ", "_") for elemento in resto_spliteado]

                # extraer driver1 y driver2
                driver1 = resto_spliteado[0]
                driver2 = resto_spliteado[1]

                # Append a lista de comparativas
                list_comparativas.append({"driver1": driver1, "driver2": driver2})

                # Transformar a .png y guardar en imagenes/qualy
                file = filename + ".pdf"
                input_path = os.path.join(path_outputs, file)
                nombre = f"Telemetria_{driver1}_{driver2}"
                convert_and_save_in_pngs(input_path, path_imagenes, nombre)
        else:
            # Son otros formatos
            if filename_troceado[0] == "Meteorologia":
                nombre = "Meteorologia"
                list_otros.append({"nombre": nombre})

                # Transformar a .png y guardar en imagenes/qualy
                file = filename + ".pdf"
                input_path = os.path.join(path_outputs, file)
                convert_and_save_in_pngs(input_path, path_imagenes, nombre)

            elif filename_troceado[0] == "Sectors":
                # Añadir a la lista de otros
                tipo = filename_troceado[-1]
                nombre = f"Mejores_Sectores_{tipo}"
                list_otros.append({"nombre": nombre})

                # Transformar a .png y guardar en imagenes/qualy
                file = filename + ".pdf"
                input_path = os.path.join(path_outputs, file)
                convert_and_save_in_pngs(input_path, path_imagenes, nombre)

            elif filename_troceado[0][:-2] == "Speed":
                tipo = filename_troceado[-2:]
                nombre = f"Mejores_Velocidades_{tipo}"
                list_otros.append({"nombre": nombre})

                # Transformar a .png y guardar en imagenes/qualy
                file = filename + ".pdf"
                input_path = os.path.join(path_outputs, file)
                convert_and_save_in_pngs(input_path, path_imagenes, nombre)

            elif filename_troceado[0] == "Ritmos":
                # Añadir a la lista de otros
                nombre = f"Ritmos_Stint1"
                list_otros.append({"nombre": nombre})

                # Transformar a .png y guardar en imagenes/qualy
                file = filename + ".pdf"
                input_path = os.path.join(path_outputs, file)
                convert_and_save_in_pngs(input_path, path_imagenes, nombre)

            elif filename_troceado[0] == "AeroAnalisis":
                # Añadir a la lista de otros
                nombre = f"Aeroanalisis"
                list_otros.append({"nombre": nombre})

                # Transformar a .png y guardar en imagenes/qualy
                file = filename + ".pdf"
                input_path = os.path.join(path_outputs, file)
                convert_and_save_in_pngs(input_path, path_imagenes, nombre)

    # Creacion del contexto os
    contexto = {
        "qualy": dict_category,
        "filas_pilotos": list_dict,
        "filas_comparativas": list_comparativas,
        "filas_minisectores": list_minisectores,
        "filas_otros": list_otros
    }

    # CARGAR EL TEMPLATE DE LA PLANTILLA
    plantilla_path = os.path.join(os.getcwd(), "src", "creators", "PlantillaGP")
    env = Environment(loader=FileSystemLoader(plantilla_path))
    template = env.get_template('plantilla_qualy.html')

    # RENDERIZAR DOCUMENT0
    html_documento = template.render(contexto)

    # SAVE
    path = os.path.join(path, "qualy.html")
    with open(path, "w") as file:
        file.write(html_documento)


def _get_distancia(distancia):
    distancia_km = distancia / 1000
    distancia_str = str(distancia_km)
    distancia_splited = distancia_str.split('.')
    return f"{distancia_splited[0]},{distancia_splited[1]}"


def _get_item_class(i, indice_next):
    if i < indice_next:
        return "calendar-item past"
    elif i == indice_next:
        return "calendar-item next"
    else:
        return "calendar-item"



def update_calendar(race):
    # 1) Read horarios_calendario.csv
    path_outputs = os.path.join(os.getcwd(), "src", "outputs")
    df_horarios = pd.read_csv(os.path.join(path_outputs, "horarios_calendario.csv"))
    df_horarios['FechaIni'] = pd.to_datetime(df_horarios['FechaIni'], format='%d/%m/%Y')
    df_horarios['FechaFin'] = pd.to_datetime(df_horarios['FechaFin'], format='%d/%m/%Y')

    # 2) Seleccionar la fila siguiente a race
    index_actual = df_horarios.loc[df_horarios['Race']==race,].index
    next_race = df_horarios.loc[index_actual + 1,]

    # 3) Crear el contexto
    ## Next event
    dict_next_event = {}
    race = next_race['Race'].iloc[0]
    circuito = next_race['Location'].iloc[0]
    circuitofile = next_race['CircuitoFile'].iloc[0]
    banderafile = next_race['BanderaFile'].iloc[0]
    fecha_ini = next_race['FechaIni'].iloc[0]
    fecha_fin = next_race['FechaFin'].iloc[0]

    race_translate = DICT_PAISES[race]
    day_ini = str(fecha_ini.day)
    month_ini = DICT_MONTHS_RED[fecha_ini.month]
    day_fin = str(fecha_fin.day)
    month_fin = DICT_MONTHS_RED[fecha_fin.month]
    year = str(fecha_ini.year)

    dict_next_event['href'] = DICT_PAISES_HREF[race]
    dict_next_event['image'] = circuitofile
    dict_next_event['background'] = banderafile
    dict_next_event['GPname'] = "GP de " + race_translate
    dict_next_event['circuito'] = circuito
    dict_next_event['day_ini'] = day_ini
    dict_next_event['day_fin'] = day_fin
    dict_next_event['month_ini'] = month_ini
    dict_next_event['month_fin'] = month_fin

    ## Horarios
    issprint = next_race['IsSprint'].iloc[0]
    if issprint == "Yes":
        list_horarios = []
        list_columns = ["FP1","FP2/SQF","FP3/Sprint","Qualy","Carrera"]
        list_firstday = ["FP1","FP2/SQF"]
        list_intermediate = ["FP3/Sprint","Qualy"]

        for column in list_columns:
            event = DICT_SPRINT_WEEKEND_SESSIONS[column]
            hora = next_race[column].iloc[0]
            if column in list_firstday:
                day = str(fecha_ini.day)
                month = DICT_MONTHS[fecha_ini.month]
                fecha = day + " de " + month
            elif column in list_intermediate:
                day = str(fecha_ini.day + 1)
                month = DICT_MONTHS[fecha_ini.month]
                fecha = day + " de " + month
            else:
                day = str(fecha_fin.day)
                month = DICT_MONTHS[fecha_fin.month]
                fecha = day + " de " + month

            list_horarios.append({"event": event, "fecha": fecha, "hora": hora})

    elif issprint == "No":
        list_horarios = []
        list_columns = ["FP1","FP2/SQF","FP3/Sprint","Qualy","Carrera"]
        list_firstday = ["FP1","FP2/SQF"]
        list_intermediate = ["FP3/Sprint","Qualy"]

        for column in list_columns:
            event = DICT_NORMAL_WEEKEND_SESSIONS[column]
            hora = next_race[column].iloc[0]
            if column in list_firstday:
                day = str(fecha_ini.day)
                month = DICT_MONTHS[fecha_ini.month]
                fecha = day + " de " + month
            elif column in list_intermediate:
                day = str(fecha_ini.day + 1)
                month = DICT_MONTHS[fecha_ini.month]
                fecha = day + " de " + month
            else:
                day = str(fecha_fin.day)
                month = DICT_MONTHS[fecha_fin.month]
                fecha = day + " de " + month

            list_horarios.append({"event": event, "fecha": fecha, "hora": hora})

    ## Next Date
    hora_inicial = str(next_race["FP1"].iloc[0])
    month = fecha_ini.month
    if month < 10:
        month_str = f"0{month}"
    else:
        month_str = f"{month}"
    
    if int(day_ini) < 10:
        day_ini_str = f"0{day_ini}"
    else:
        day_ini_str = f"{day_ini}"

    nexdate = f"{year}-{month_str}-{day_ini_str}T{hora_inicial}:00"

    ## Calendar objects
    calendar_rows = []
    df_stats_circuitos = pd.read_csv(os.path.join(os.getcwd(), "src", "outputs", "info_circuitos.csv"))

    for i in range(df_stats_circuitos.shape[0]):
        race = df_stats_circuitos['Race'].iloc[i]
        bandera = df_stats_circuitos['Bandera'].iloc[i]
        distancia = _get_distancia(df_stats_circuitos['Distancia'].iloc[i])
        vueltas = df_stats_circuitos['Vueltas'].iloc[i]
        distancia_total = _get_distancia(df_stats_circuitos['Distancia_total'].iloc[i])
        vuelta_rapida = str(df_stats_circuitos['Vuelta_rapida'].iloc[i])
        piloto_vuelta_rapida = df_stats_circuitos['Piloto_vuelta_rapida'].iloc[i]
        href = DICT_PAISES_HREF[race]
        calendar_item_class = _get_item_class(i, index_actual + 1)
        background = df_horarios['BanderaFile'].iloc[i]
        image = df_horarios['CircuitoFile'].iloc[i]
        gpname = "GP de " + DICT_PAISES[race]
        circuito = df_horarios['Location'].iloc[i]
        fecha_ini = df_horarios['FechaIni'].iloc[i]
        fecha_fin = df_horarios['FechaFin'].iloc[i]

        calendar_rows.append({
            "href": href,
            "class": calendar_item_class,
            "background": background,
            "image": image,
            "circuito": circuito,
            "GPname": gpname,
            "circuito": circuito,
            "day_ini": str(fecha_ini.day),
            "day_fin": str(fecha_fin.day),
            "month_ini": DICT_MONTHS_RED[fecha_ini.month],
            "month_fin": DICT_MONTHS_RED[fecha_fin.month],
            "pais": f"{DICT_PAISES[race]} {bandera}",
            "distancia": f"{distancia} km",
            "vueltas": vueltas,
            "distancia_total": f"{distancia_total} km",
            "vuelta_rapida": vuelta_rapida,
            "piloto": piloto_vuelta_rapida,
            "index": i
        })

    calendar_rows_blocks = [calendar_rows[i:i+3] for i in range(0, len(calendar_rows), 3)]

    ## Contexto final
    contexto = {
        "next_race": dict_next_event,
        "filas_horarios": list_horarios,
        "next_date": nexdate,
        "calendar_blocks": calendar_rows_blocks
    }

    # 3) Render plantilla con contexto
    creators_path = os.path.join(os.getcwd(), "src", "creators")
    env = Environment(loader=FileSystemLoader(creators_path))
    template = env.get_template('plantilla_calendario.html')

    # RENDERIZAR DOCUMENT0
    html_documento = template.render(contexto)

    # SAVE
    filepath = os.path.join(os.getcwd(), "src", "html", "calendario.html")
    with open(filepath, "w") as file:
        file.write(html_documento)

    # 4) Subir al S3 el nuevo calendario 
    s3_path = ""
    upload_file_s3(filepath, s3_path)


def _get_context_next_event(race):
    ## Common
    path_outputs = os.path.join(os.getcwd(), "src", "outputs")
    df_horarios = pd.read_csv(os.path.join(path_outputs, "horarios_calendario.csv"))
    df_horarios['FechaIni'] = pd.to_datetime(df_horarios['FechaIni'], format='%d/%m/%Y')
    df_horarios['FechaFin'] = pd.to_datetime(df_horarios['FechaFin'], format='%d/%m/%Y')

    ## Next Race
    index_actual = df_horarios.loc[df_horarios['Race']==race,].index
    next_race = df_horarios.loc[index_actual + 1,]

    dict_next_event = {}
    race = next_race['Race'].iloc[0]
    circuito = next_race['Location'].iloc[0]
    circuitofile = next_race['CircuitoFile'].iloc[0]
    banderafile = next_race['BanderaFile'].iloc[0]
    fecha_ini = next_race['FechaIni'].iloc[0]
    fecha_fin = next_race['FechaFin'].iloc[0]

    race_translate = DICT_PAISES[race]
    day_ini = str(fecha_ini.day)
    month_ini = DICT_MONTHS_RED[fecha_ini.month]
    day_fin = str(fecha_fin.day)
    month_fin = DICT_MONTHS_RED[fecha_fin.month]
    year = str(fecha_ini.year)

    dict_next_event['href'] = DICT_PAISES_HREF[race]
    dict_next_event['image'] = circuitofile
    dict_next_event['background'] = banderafile
    dict_next_event['GPname'] = "GP de " + race_translate
    dict_next_event['circuito'] = circuito
    dict_next_event['day_ini'] = day_ini
    dict_next_event['day_fin'] = day_fin
    dict_next_event['month_ini'] = month_ini
    dict_next_event['month_fin'] = month_fin


    ## List_horarios
    issprint = next_race['IsSprint'].iloc[0]
    if issprint == "Yes":
        list_horarios = []
        list_columns = ["FP1","FP2/SQF","FP3/Sprint","Qualy","Carrera"]
        list_firstday = ["FP1","FP2/SQF"]
        list_intermediate = ["FP3/Sprint","Qualy"]

        for column in list_columns:
            event = DICT_SPRINT_WEEKEND_SESSIONS[column]
            hora = next_race[column].iloc[0]
            if column in list_firstday:
                day = str(fecha_ini.day)
                month = DICT_MONTHS[fecha_ini.month]
                fecha = day + " de " + month
            elif column in list_intermediate:
                day = str(fecha_ini.day + 1)
                month = DICT_MONTHS[fecha_ini.month]
                fecha = day + " de " + month
            else:
                day = str(fecha_fin.day)
                month = DICT_MONTHS[fecha_fin.month]
                fecha = day + " de " + month

            list_horarios.append({"event": event, "fecha": fecha, "hora": hora})

    elif issprint == "No":
        list_horarios = []
        list_columns = ["FP1","FP2/SQF","FP3/Sprint","Qualy","Carrera"]
        list_firstday = ["FP1","FP2/SQF"]
        list_intermediate = ["FP3/Sprint","Qualy"]

        for column in list_columns:
            event = DICT_NORMAL_WEEKEND_SESSIONS[column]
            hora = next_race[column].iloc[0]
            if column in list_firstday:
                day = str(fecha_ini.day)
                month = DICT_MONTHS[fecha_ini.month]
                fecha = day + " de " + month
            elif column in list_intermediate:
                day = str(fecha_ini.day + 1)
                month = DICT_MONTHS[fecha_ini.month]
                fecha = day + " de " + month
            else:
                day = str(fecha_fin.day)
                month = DICT_MONTHS[fecha_fin.month]
                fecha = day + " de " + month

            list_horarios.append({"event": event, "fecha": fecha, "hora": hora})

    ## Next Date
    hora_inicial = str(next_race["FP1"].iloc[0])
    month = fecha_ini.month
    if month < 10:
        month_str = f"0{month}"
    else:
        month_str = f"{month}"
    
    if int(day_ini) < 10:
        day_ini_str = f"0{day_ini}"
    else:
        day_ini_str = f"{day_ini}"

    nexdate = f"{year}-{month_str}-{day_ini_str}T{hora_inicial}:00"

    return dict_next_event, list_horarios, nexdate


def _update_calendar_item(soup, contexto):
    # Get calendar item
    calendar_item = soup.find("div", class_="calendar-item")

    # 1. Actualizar bandera background
    image_container = calendar_item.find("div", class_="image-container")
    image_container["style"] = f"background-image: url('imagenes/banderas/{contexto['next_race']['background']}');"

    # 2. Actualizar imagen circuito
    circuito_img = image_container.find("img")
    circuito_img["src"] = f"imagenes/circuitos/{contexto['next_race']['image']}"
    circuito_img["alt"] = contexto['next_race']['circuito']

    # 3. Actualizar título, circuito y fechas
    calendar_item.find("h2").string = contexto['next_race']["GPname"]
    calendar_item.find("p").string = contexto['next_race']["circuito"]
    calendar_item.find("span").string = f"{contexto['next_race']['day_ini']} {contexto['next_race']['month_ini']} - {contexto['next_race']['day_fin']} {contexto['next_race']['month_fin']}"

    return soup


def _update_tabla_horarios(soup, contexto):
    # Get tabla and clean
    tabla = soup.find("table", id="horarios-table")
    for fila in tabla.find_all("tr"):
        fila.decompose()

    # Fill new tabla
    for fila_horarios in contexto['fila_horarios']:
        fila = soup.new_tag("tr")

        # Sesion
        td1 = soup.new_tag("td", **{"class": "neutro"})
        strong = soup.new_tag("strong")
        strong.string = fila_horarios['event']
        td1.append(strong)

        # Fecha
        td2 = soup.new_tag("td", **{"class": "neutro", "style": "color: gray"})
        td2.string = fila_horarios['fecha']

        # Hora
        td3 = soup.new_tag("td", **{"class": "neutro"})
        span = soup.new_tag("span")
        span.string = fila_horarios['hora']
        td3.append(span)

        # Save
        fila.extend([td1, td2, td3])
        tabla.append(fila)
    
    return soup


def _update_next_date(soup, contexto):
    # Load script tag
    script_tag = soup.find("script", string=re.compile("const targetDate"))

    # Update
    if script_tag:
        nueva_fecha = contexto['next_date']
        script_tag.string = re.sub(
            r'const targetDate = new Date\(".*?"\);',
            f'const targetDate = new Date("{nueva_fecha}");',
            script_tag.string
        )
    return soup

def update_index(race):
    # 1) Get index.html del s3
    s3_path = ""
    download_path = os.path.join(os.getcwd(), "src", "html", "index.html") 
    download_file_s3("index.html", s3_path, download_path)
    with open(download_path, "r", encoding="utf-8") as f:
        index_html = f.read()

    # 2) Generar contexto para actualizar evento y fecha próxima
    dict_next_event, list_horarios, nexdate = _get_context_next_event(race)

    contexto = {
        "next_race": dict_next_event,
        "fila_horarios": list_horarios,
        "next_date": nexdate
    }

    # 3) Sustituir las region y proxima fecha del documento
    soup = BeautifulSoup(index_html, "html.parser")

    ## Actualizar calendar_tem (Bloque Next Event)
    soup = _update_calendar_item(soup, contexto)

    ## Actualizar Tabla de Horarios
    soup = _update_tabla_horarios(soup, contexto)

    ## Actualizar Next fecha
    soup = _update_next_date(soup, contexto)

    # 3) Save in local
    nuevo_html = soup.prettify()

    save_path = os.path.join(os.getcwd(), "src", "html", "index.html")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(nuevo_html)

    # 4) Subir al s3 el nuevo index.html
    s3_path = ""
    upload_file_s3(save_path, s3_path)

    