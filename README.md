# F1 Data Analysis

This repository contains scripts for analyzing Formula 1 data, including qualifying and race data, and generating various visualizations and statistics.

## Summary

This project provides tools to analyze Formula 1 data, such as telemetry, lap times, and driver performance. It leverages the `fastf1` library to access and process historical F1 data. Users can generate detailed plots comparing drivers, analyze track dominance, and visualize performance metrics across different sessions and seasons. This project is useful for F1 enthusiasts, data analysts, or anyone interested in gaining deeper insights into F1 performance.

## Tech Stack

* `fastf1`: 3.8.3
* `jupyter`
* `numpy`
* `matplotlib`
* `pandas`
* `seaborn`
* `pillow`
* `scikit-learn`
* `pdf2image`
* `boto3`
* `pytz`
* `tabulate`
* `pycountry`
* `statsmodels`
* `fastapi`
* `rich`
* `python-dotenv`

## Quick Start

### Installation

1. Clone the repository:
 ```bash
 git clone https://github.com/Joaquingit3/F1_Data_Analysis.git
 cd F1_Data_Analysis
 ```
2. Install dependencies:
 ```bash
 pip install -r src/requirements.txt
 ```

### Running the analysis

The primary analysis script is `src/main.py`. To run it, you will need to provide arguments for the year, grand prix, session, and drivers you want to analyze.

Example:
```bash
python src/main.py --year 2023 --grand-prix "Bahrain" --session Q --driver1 "VER" --driver2 "LEC"
```

For race analysis, use `src/main_race.py`:
```bash
python src/main_race.py --year 2023 --grand-prix "Bahrain" --session R --driver1 "VER" --driver2 "LEC"
```

Refer to the `src/__init__.py` file for available modes and options.

## Project Structure

```
.
├── src/
│ ├── __init__.py
│ ├── common/
│ ├── creators/
│ ├── driver/
│ ├── grapher/
│ ├── logger/
│ ├── s3_manager/
│ ├── utils/
│ ├── main.py
│ ├── main_extractor.py
│ ├── main_openf1.py
│ ├── main_race.py
│ ├── main_testing.py
│ ├── main_win_stats.py
│ └── requirements.txt
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

## Key Concepts

* **FastF1 Session Loading**: The core of the data retrieval happens in `src/main.py` and `src/main_race.py` using `fastf1.get_session()` and subsequent `.load()` methods. This process fetches session data, including laps and telemetry.
* **Telemetry and Lap Data Processing**: Functions in `src/utils/functions.py` and `src/grapher/graphs.py` are responsible for processing raw telemetry and lap data. This includes calculating deltas between drivers (`calculate_deltas_twodrivers`), formatting lap times (`format_time`), and extracting specific metrics like sector times.
* **Graph Generation**: The `src/grapher/` directory contains modules for creating various plots. For example, `src/grapher/graphs.py` includes functions like `graph_telemetry` for visualizing speed and throttle data, and `create_minisectors_graph` for analyzing track dominance.
* **Driver and Team Data**: Constants and helper functions related to driver and team information, such as colors and names, are managed in `src/common/const.py` and `src/driver/__init__.py`.
* **Command-Line Interface**: The `src/main.py` script uses `argparse` to handle command-line arguments for specifying analysis parameters like year, grand prix, session, and drivers. The available options and modes are defined in `src/__init__.py`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.