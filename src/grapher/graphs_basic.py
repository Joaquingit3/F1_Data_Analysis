import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_results_table(df, column, session, save=True):
    # Plot config
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=700)
    ax.axis('off')

    # Graph table 
    table_data = df.values.tolist()
    column_labels = df.columns.tolist()
    row_colors = ['#2E2E2E', '#1E1C1C']
    table = ax.table(cellText=table_data, colLabels=column_labels,
                      cellLoc='center', loc='left',
                      colColours=["#1E1C1C"] * len(column_labels),  # Fondo de la cabecera
                      cellColours=[[row_colors[i % len(row_colors)] for _ in range(len(column_labels))]
                                   for i in range(len(table_data))],  # Fondo alternado de las celdas
                      #cellColours=[["#1E1C1C"] * len(column_labels)] * len(table_data),  # Fondo de las celdas
                      bbox=[0.1, 0.1, 0.8, 0.8])  # Posición y tamaño de la tabla [xmin, ymin, width, height]
    # Estilo de la tabla
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # save
    if save is True:
        eventname = session.event.EventName.split(" ")[0]
        path = f"{session.event.year}_{eventname}"
        path = os.path.join(os.getcwd(), "src", "outputs", path)

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, session.name)
        if not os.path.exists(path):
            os.makedirs(path)

        plot_filename = f"{column.strip()}Table"
        plot_filename_png = plot_filename + ".png"
        plot_filename_pdf = plot_filename + ".pdf"
        plot_filename = os.path.join(path, plot_filename)
        print(plot_filename)
        plt.savefig(os.path.join(path, plot_filename_png), format='png')
        plt.savefig(os.path.join(path, plot_filename_pdf), format='pdf')
    else:
        plt.show()