# Circular Grid

Circular Grid is a campus resource dashboard backed by the hourly campus
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