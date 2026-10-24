# EcoFlow Campus

EcoFlow Campus is a campus resource dashboard backed by the hourly campus
streams and waste-event datasets.

## Run locally

From `circular_grid`, install the Python dependencies and put the original
source workbooks in `circular_grid/data/raw` using the filenames configured in
`circular_grid/config.py`. To use another directory, set
`CIRCULAR_GRID_UPLOAD_DIR` before running the data audit.

```powershell
cd circular_grid
python -m pip install -r requirements.txt
$env:CIRCULAR_GRID_UPLOAD_DIR = "$PWD\data\raw"
make data
make train
make test
```

The `data` target converts and audits the supplied workbooks. It does not
generate replacement source data. `make demo-data` is available only for
explicitly synthetic waste-event demo data; do not use those generated events
as measured results.

In a second terminal, start the API and frontend:

```powershell
cd circular_grid
make run
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The API reads
`data/waste_events_train.csv` for waste allocation and
`data/waste_events_test.csv` for the classifier feed. The test file is
unlabeled, so its predictions must not be presented as accuracy measurements.
`make run` binds the API to localhost by default.

The frontend production build can be checked with `npm run build` from
`circular_grid/frontend`.

The History page provides campus-wide filters for building, date/time range,
dataset, and metric. It covers hourly readings, daily summaries, calendar and
weather records, labelled and unlabelled waste events, and manual meter logs.
Values and units come from the selected source dataset; hourly imputed readings
are identified and missing values remain blank. Building details also include
a historical-readings view for individual hourly resource streams. Static
building metadata has no timestamped history to filter.