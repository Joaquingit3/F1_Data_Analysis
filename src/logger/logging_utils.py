import logging
import pandas as pd
from io import StringIO

def config_logger():
    log_capture = StringIO()
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s \t%(levelname)s \t%(message)s',
        handlers=[logging.StreamHandler(log_capture)]
    )
    return log_capture


def parse_logger_to_df(captured_logs):
    try:
        # Extract lines
        log_lines = captured_logs.strip().split("\n")
        # print(log_lines)
        # print("\n")
        list_modules = []
        list_type = []
        list_msg = []

        # Loop lines to parse the info
        for log_line in log_lines:
            log_line_splited = log_line.split("\t")
            # print(log_line_splited)
            try:
                if log_line_splited[0].strip() == "fastf1.api":
                    list_modules.append(log_line_splited[0].strip())
                    list_type.append(log_line_splited[1].strip())
                    list_msg.append(log_line_splited[2])
                elif log_line_splited[0].strip() == "fastf1.ergast":
                    list_modules.append(log_line_splited[0].strip())
                    list_type.append(log_line_splited[1].strip())
                    list_msg.append(log_line_splited[2])
                elif log_line_splited[0] == "Traceback (most recent call last):":
                    # print("Entra3\n")
                    pass
                elif log_line_splited[0][:len("IndexError:")] == "IndexError:":
                    # print("Entra4\n")
                    pass
                else:
                    # print("Entra5\n")
                    list_modules.append(log_line_splited[0].split(".")[2].strip())
                    list_type.append(log_line_splited[1].strip())
                    list_msg.append(log_line_splited[2])
            except IndexError as e:
                print(f"IndexError: {log_line_splited[0]}")

        # Convert to a df
        # print(f"\nlist_modules={list_modules} - {len(list_modules)}\nlist_type={list_type} - {len(list_type)}\nlist_msg={list_msg} - {len(list_msg)}")
        d = {'modules': list_modules, 'type': list_type, 'message': list_msg}
        df_logs = pd.DataFrame(d)
        return df_logs
    except Exception as e:
        raise e



def find_drivers_noLap_warning(df_logs):
    # Masks
    key = "No lap data for driver"
    mask_warning = df_logs["type"] == "WARNING"
    mask_no_lap_warning = df_logs["message"].str.startswith(key)

    # Filtrar df
    filtered_df = df_logs.loc[mask_warning & mask_no_lap_warning, ]

    # Ver si hay casos
    list_drivers = []
    if len(filtered_df) > 0:
        # Hay pilotos sin vuelta
        for i in range(filtered_df.shape[0]):
            msg = filtered_df.iloc[i]["message"]
            driver = msg.split(key)[1].strip()
            list_drivers.append(int(driver))

    return list_drivers

