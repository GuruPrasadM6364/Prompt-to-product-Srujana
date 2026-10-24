# Circular Grid

Circular Grid is a campus resource dashboard backed by the hourly campus
streams and waste-event datasets.

## Run locally

From `circular_grid`:

```powershell
python -m pip install -r requirements.txt
python scripts/generate_waste_events.py
python -m uvicorn src.api.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The API reads
`data/waste_events_train.csv` for waste allocation and
`data/waste_events_test.csv` for the classifier feed. The test file is
unlabeled, so its predictions are intentionally presented as predictions, not
accuracy measurements.

The frontend production build can be checked with `npm run build` from
`circular_grid/frontend`.