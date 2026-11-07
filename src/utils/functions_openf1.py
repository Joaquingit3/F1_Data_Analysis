


def get_final_sesion_table(session, openf1_client, year, grand_prix, save_gap=False, use_Q1_data = False):
    # ----------------------------------------------- QUALY ---------------------------------------------------------------------------------------
    print(session.session_info['Type'])
    if session.session_info['Type'] == "Qualifying":
        # DF results
        df_results = openf1_client.get_qualifying_results(year, grand_prix)

        # Save gap
        if save_gap and not use_Q1_data:
            race_num = session.session_info['Meeting']['Number']
            country_code = session.session_info['Meeting']['Country']['Code']
            save_time_gap(df_results, race_num, country_code)
            print("Saved Time Gap")
        elif save_gap and use_Q1_data:
            show_resumes_table_qualyfying(session, use_Q1_data)

        # DF resume 
        show_resumes_table_qualyfying(session)

    # ----------------------------------------------- QUALY ---------------------------------------------------------------------------------------
    elif session.session_info['Type'] == "Practice":
        df_results = get_final_global_table_practice(session)

    # ----------------------------------------------- RACE ---------------------------------------------------------------------------------------
    elif session.session_info['Type'] == "Race":
        try:
            # Get Data
            results = session.results[['Position', 'Abbreviation', 'FullName', 'Status', 'TeamName', 'CountryCode', 'Points', 'GridPosition']]
            '''
            position = [1.0, 3.0, 2.0, 4.0, 18.0, 17.0, 11.0, 6.0, 16.0, 7.0, 5.0, 15.0, 20.0, 9.0, 10.0, 19.0, 8.0, 13.0, 14.0, 12.0]
            points = [25.0, 15.0, 18.0, 12.0, 0.0, 0.0, 0.0, 8.0, 0.0, 6.0, 10.0, 0.0, 0.0, 2.0, 1.0, 0.0, 4.0, 0.0, 0.0, 0.0]
            results['Position'] = position
            results['Points'] = points
            '''

            # Get df_penalties and df_pending_penalties 
            df_penalties = get_df_penalties(session)
            df_pending_penalties = get_df_pending_penalties(df_penalties)

            # Check if Points is empty
            if results['Points'].isna().all():
                print("\nLa columna 'Points' es completamente NaN. Se cancela el paso por la función.")
                columnas_todas_nan = results.columns[results.isna().all()]
                print("Las Columnas completamente NaN son:", list(columnas_todas_nan))
                print(tabulate(results, headers='keys', showindex=False, tablefmt='grid'))
                return None
            else:
                list_abbreviation = results['Abbreviation'].to_list()
                list_position = results['Position'].to_list()
                list_points = results['Points'].to_list()
                list_teams = results['TeamName'].to_list()
                list_pilotos = results['FullName'].to_list()
                list_nacionalidad = [DICT_PILOTOS_NACIONALIDAD[list_abbreviation[i]] if nacionalidad == "" else nacionalidad for i, nacionalidad in enumerate(results['CountryCode'].to_list())]

                # 2) Get Laps
                laps = session.laps

                # 3) Get Laps of the first driver
                fastest_driver = results.loc[results['Position'] == 1.0, 'Abbreviation'].iloc[0]
                mask_driver = laps['Driver'] == fastest_driver
                subset_laps_driver = laps.loc[mask_driver,].copy()
                laps_fastest_driver = generate_correct_laptimes(subset_laps_driver, session)

                laps_fastest_driver = [laptime.total_seconds() for laptime in laps_fastest_driver]
                totaltime_fastest_driver = sum(laps_fastest_driver)

                # 4) Get distances to the other drivers
                time_distance = []
                total_fast_driver_str = str(timedelta(seconds=totaltime_fastest_driver))[:-3]
                time_distance.append(total_fast_driver_str)# For the fastest

                for i in range(1, results.shape[0]):
                    driver = results['Abbreviation'].iloc[i]
                    status = results['Status'].iloc[i]

                    if status == 'Finished':
                        # Calculate time distances by times in the race
                        mask_driver = laps['Driver'] == driver
                        subset_laps_driver = laps.loc[mask_driver, ].copy()
                        laps_driver = generate_correct_laptimes(subset_laps_driver, session)

                        laps_driver = [laptime.total_seconds() for laptime in laps_driver]
                        totaltime_driver = sum(laps_driver)
                        final_diff = totaltime_driver - totaltime_fastest_driver

                        # Check pending penalties
                        mask_driver_penalties = df_pending_penalties['Driver']==driver
                        if sum(mask_driver_penalties) == 0:
                            final_diff = pass_to_finaltime(final_diff)
                            time_distance.append(final_diff)
                        else:
                            # Pending penalties
                            adding_time_for_penalty = sum(df_pending_penalties.loc[mask_driver_penalties,'QuantityPenalty'].to_list())
                            final_diff += adding_time_for_penalty
                            final_diff = pass_to_finaltime(final_diff)
                            time_distance.append(final_diff)
                    else:
                        time_distance.append(DICT_STATUS[status])

                # 5) Create df
                # 5.1 Traducir equipos a nombre normal
                list_teams = [DICT_TEAMS[team] for team in list_teams]

                # 5.2 Cambiar los apellidos a mayusculas
                list_final_pilotos =[]
                for piloto in list_pilotos:
                    nombre = piloto.split(" ")[0]
                    apellido = piloto.split(" ")[1].upper()
                    list_final_pilotos.append(nombre + " " + apellido)

                # Cambiar posicion a entero
                list_position = [int(posicion) for posicion in list_position]

                # Obtener lista de cambios de posicio
                list_cambios = []
                for i, drv in enumerate(list_abbreviation):
                    grid_posicion = results.loc[results['Abbreviation']==drv, 'GridPosition'].iloc[0]
                    final_posicion = results.loc[results['Abbreviation']==drv, 'Position'].iloc[0]
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
                    'posicion': list_position,
                    'cambios': list_cambios,
                    'nacionalidad': list_nacionalidad,
                    'Driver': list_abbreviation,
                    'piloto': list_final_pilotos,
                    'equipo': list_teams,
                    'tiempo': time_distance,
                    'puntos': list_points
                }
                df_results = pd.DataFrame(d)

                # Ordenar
                df_results.sort_values(by='posicion', ascending=True, inplace=True)
                
                # Show
                if session.session_info['Type'] == "Race":
                    # Penalizaciones
                    print(f"\nPenalizaciones cometidas en carrera:")
                    print(tabulate(df_penalties, headers='keys', showindex=False, tablefmt='grid'))
                    print(f"\nPenalizaciones pendientes de servir:")
                    print(tabulate(df_pending_penalties, headers='keys', showindex=False, tablefmt='grid'))

                    # Resultados
                    print(f"\nResultados de la {session.session_info['Type']}")
                    print(tabulate(df_results, headers='keys', showindex=False, tablefmt='grid'))
                else:
                    # Resultados
                    print(f"\nResultados de la {session.session_info['Type']}")
                    print(tabulate(df_results, headers='keys', showindex=False, tablefmt='grid'))

                
                return df_results
        except Exception as e:
            raise e
        
    # Ordenar, tabulate and return 
    # df_results.sort_values(by='Position', ascending=True, inplace=True)
    print(f"\nResultados de la {session.session_info['Type']}")
    print(tabulate(df_results, headers='keys', showindex=False, tablefmt='grid'))
    return df_results

