import { useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../api/client';
import type { CampusHistoryResponse, HistoryDataset } from '../types';

const DATASETS: { id: HistoryDataset; label: string }[] = [
  { id: 'hourly', label: 'Hourly resource readings' },
  { id: 'daily_summary', label: 'Daily campus summary' },
  { id: 'calendar', label: 'Calendar and weather' },
  { id: 'waste_train', label: 'Waste events (labelled)' },
  { id: 'waste_test', label: 'Waste events (unlabelled)' },
  { id: 'manual_logs', label: 'Manual meter logs' },
];

const METRICS: Record<HistoryDataset, { id: string; label: string }[]> = {
  hourly: [
    { id: 'electricity_kwh', label: 'Electricity (kWh)' },
    { id: 'water_litres', label: 'Water (L)' },
    { id: 'solar_generation_kwh', label: 'Solar generation (kWh)' },
    { id: 'footfall', label: 'Footfall (people)' },
  ],
  daily_summary: [
    { id: 'occupancy_avg', label: 'Average occupancy (people)' },
    { id: 'electricity_kwh', label: 'Electricity (kWh)' },
    { id: 'water_litres', label: 'Water (L)' },
    { id: 'solar_generation_kwh', label: 'Solar generation (kWh)' },
    { id: 'wet_organic_kg', label: 'Wet organic waste (kg)' },
    { id: 'dry_recyclable_kg', label: 'Dry recyclable waste (kg)' },
    { id: 'ewaste_kg', label: 'E-waste (kg)' },
    { id: 'hazardous_kg', label: 'Hazardous waste (kg)' },
    { id: 'composted_kg', label: 'Composted (kg)' },
    { id: 'recycled_kg', label: 'Recycled (kg)' },
    { id: 'contamination_pct', label: 'Contamination (%)' },
    { id: 'landfill_kg', label: 'Landfill (kg)' },
  ],
  calendar: [
    { id: 'avg_temp_c', label: 'Average temperature (°C)' },
    { id: 'rainfall_mm', label: 'Rainfall (mm)' },
    { id: 'humidity_pct', label: 'Humidity (%)' },
  ],
  waste_train: [{ id: 'weight_g', label: 'Event weight (g)' }],
  waste_test: [{ id: 'weight_g', label: 'Event weight (g)' }],
  manual_logs: [{ id: 'value', label: 'Logged value (original unit)' }],
};

const BUILDINGS = [
  ['B01', 'Academic Block A'],
  ['B02', 'Academic Block B'],
  ['B03', 'Central Library'],
  ['B04', 'Engineering Labs & Innovation Centre'],
  ['B05', 'Boys Hostel'],
  ['B06', 'Girls Hostel'],
  ['B07', 'Food Court'],
  ['B08', 'Admin Block'],
] as const;
const WASTE_CATEGORIES = [
  'food_organic',
  'plastic',
  'paper_cardboard',
  'metal',
  'ewaste',
  'glass',
];
const MANUAL_STREAMS = ['electricity', 'water', 'waste'];
const TABLE_LIMIT = 200;

function formatDateTime(value: string | null): string {
  if (!value) return 'Unavailable';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function toDateTimeLocal(value: string | null): string {
  return value ? value.slice(0, 16) : '';
}

function formatTick(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit' });
}

export default function HistoricalData() {
  const [dataset, setDataset] = useState<HistoryDataset>('hourly');
  const [metric, setMetric] = useState('electricity_kwh');
  const [buildingId, setBuildingId] = useState('all');
  const [category, setCategory] = useState('');
  const [manualStream, setManualStream] = useState('');
  const [dateStart, setDateStart] = useState('');
  const [dateEnd, setDateEnd] = useState('');
  const [appliedRange, setAppliedRange] = useState<{ start: string; end: string } | null>(null);
  const [data, setData] = useState<CampusHistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        setError(null);
        const result = await api.getCampusHistory(
          dataset,
          metric,
          buildingId,
          appliedRange?.start,
          appliedRange?.end,
          category || undefined,
          manualStream || undefined
        );
        if (cancelled) return;
        setData(result);
        if (!appliedRange) {
          setDateStart(toDateTimeLocal(result.start));
          setDateEnd(toDateTimeLocal(result.end));
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Failed to load historical data');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [dataset, metric, buildingId, appliedRange, category, manualStream]);

  function changeDataset(value: HistoryDataset) {
    setDataset(value);
    setMetric(METRICS[value][0].id);
    if (value === 'calendar') setBuildingId('all');
    setCategory('');
    setManualStream('');
    setAppliedRange(null);
  }

  function applyRange(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!dateStart || !dateEnd) {
      setError('Choose both a start and end date/time.');
      return;
    }
    if (dateStart > dateEnd) {
      setError('Start date/time must be before the end date/time.');
      return;
    }
    setAppliedRange({ start: dateStart, end: dateEnd });
  }

  const numericPoints = useMemo(
    () => (data?.points ?? []).filter((point) => point.value !== null),
    [data]
  );
  const canAggregate = dataset !== 'manual_logs';
  const chartData = useMemo(() => {
    if (!canAggregate) return [];
    const grouped = new Map<string, { sum: number; count: number }>();
    for (const point of numericPoints) {
      if (point.value === null) continue;
      const current = grouped.get(point.ts) ?? { sum: 0, count: 0 };
      current.sum += point.value;
      current.count += 1;
      grouped.set(point.ts, current);
    }
    return Array.from(grouped, ([ts, values]) => ({
      ts,
      value: values.sum / values.count,
    }));
  }, [numericPoints, canAggregate]);
  const average = canAggregate && numericPoints.length
    ? numericPoints.reduce((sum, point) => sum + (point.value ?? 0), 0) / numericPoints.length
    : null;
  const visiblePoints = data?.points.slice(0, TABLE_LIMIT) ?? [];
  const metricLabel = METRICS[dataset].find((item) => item.id === metric)?.label ?? metric;

  return (
    <section>
      <div className="page-header">
        <h1>Historical data</h1>
        <p>Filter and inspect timestamped records across resource, weather, and waste datasets.</p>
      </div>

      <div className="card history-card">
        <form className="history-controls" onSubmit={applyRange}>
          <div className="history-control">
            <label htmlFor="history-dataset">Dataset</label>
            <select
              id="history-dataset"
              className="select-control"
              value={dataset}
              onChange={(event) => changeDataset(event.target.value as HistoryDataset)}
            >
              {DATASETS.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
            </select>
          </div>
          <div className="history-control">
            <label htmlFor="history-metric">Metric</label>
            <select
              id="history-metric"
              className="select-control"
              value={metric}
              onChange={(event) => { setMetric(event.target.value); setAppliedRange(null); }}
            >
              {METRICS[dataset].map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
            </select>
          </div>
          {dataset !== 'calendar' && <div className="history-control">
            <label htmlFor="history-building">Building</label>
            <select
              id="history-building"
              className="select-control"
              value={buildingId}
              onChange={(event) => { setBuildingId(event.target.value); setAppliedRange(null); }}
            >
              <option value="all">All buildings</option>
              {BUILDINGS.map(([id, name]) => <option key={id} value={id}>{name}</option>)}
            </select>
          </div>}
          {dataset === 'waste_train' && (
            <div className="history-control">
              <label htmlFor="history-category">Waste category</label>
              <select
                id="history-category"
                className="select-control"
                value={category}
                onChange={(event) => { setCategory(event.target.value); setAppliedRange(null); }}
              >
                <option value="">All categories</option>
                {WASTE_CATEGORIES.map((item) => <option key={item} value={item}>{item.replace('_', ' ')}</option>)}
              </select>
            </div>
          )}
          {dataset === 'manual_logs' && (
            <div className="history-control">
              <label htmlFor="history-manual-stream">Manual log stream</label>
              <select
                id="history-manual-stream"
                className="select-control"
                value={manualStream}
                onChange={(event) => { setManualStream(event.target.value); setAppliedRange(null); }}
              >
                <option value="">All streams</option>
                {MANUAL_STREAMS.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </div>
          )}
          <div className="history-control">
            <label htmlFor="history-from">From date/time</label>
            <input
              id="history-from"
              type="datetime-local"
              value={dateStart}
              max={toDateTimeLocal(data?.available_end ?? null)}
              onChange={(event) => setDateStart(event.target.value)}
              required
            />
          </div>
          <div className="history-control">
            <label htmlFor="history-to">To date/time</label>
            <input
              id="history-to"
              type="datetime-local"
              min={toDateTimeLocal(data?.available_start ?? null)}
              max={toDateTimeLocal(data?.available_end ?? null)}
              onChange={(event) => setDateEnd(event.target.value)}
              value={dateEnd}
              required
            />
          </div>
          <button className="history-submit" type="submit">Apply filters</button>
        </form>

        {data?.available_start && data.available_end && (
          <p className="history-summary">
            Available data: {formatDateTime(data.available_start)} – {formatDateTime(data.available_end)}
          </p>
        )}
        {loading && <div className="loading-state"><div className="spinner" /><span>Loading history...</span></div>}
        {error && <div className="error-state">{error}</div>}
        {!loading && !error && data && data.points.length === 0 && (
          <div className="empty-state">No records match these filters.</div>
        )}
        {!loading && !error && data && data.points.length > 0 && (
          <>
            <div className="history-summary">
              {data.points.length.toLocaleString()} records · {metricLabel}
              {average !== null && <> · mean {average.toLocaleString(undefined, { maximumFractionDigits: 2 })} {data.points.find((point) => point.unit)?.unit}</>}
              {' '}· values are taken from the selected source dataset.
            </div>
            {!canAggregate && (
              <p className="history-summary">
                Manual logs retain their original units; inspect each recorded value and unit in the table.
              </p>
            )}
            {chartData.length > 0 && (
              <ResponsiveContainer width="100%" height={320}>
                <ComposedChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis
                    dataKey="ts"
                    tickFormatter={formatTick}
                    tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }} tickLine={false} axisLine={false} />
                  <Tooltip
                    labelFormatter={formatDateTime}
                    formatter={(value) => [Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 }), metricLabel]}
                    contentStyle={{
                      fontSize: '0.8125rem',
                      border: '1px solid var(--color-border)',
                      background: 'var(--color-surface)',
                      color: 'var(--color-text)',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    name={metricLabel}
                    stroke="#1a6e3c"
                    strokeWidth={2}
                    dot={chartData.length < 80}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            )}
            <p className="history-summary">
              Chart groups records with identical timestamps and shows their mean. The table lists individual records.
            </p>
            <div className="history-table-wrap">
              <table className="data-table">
                <thead>
                  <tr><th>Date and time</th><th>Building</th><th>Value</th><th>Category / event / stream</th><th>Source</th><th>Estimated</th></tr>
                </thead>
                <tbody>
                  {visiblePoints.map((point, index) => (
                    <tr key={`${point.ts}-${point.building_id}-${index}`}>
                      <td>{formatDateTime(point.ts)}</td>
                      <td>{point.building_name}</td>
                      <td>{point.value === null ? 'No reading' : `${point.value.toLocaleString(undefined, { maximumFractionDigits: 2 })} ${point.unit}`}</td>
                      <td>{point.category ?? '—'}</td>
                      <td>{point.source}</td>
                      <td>{point.imputed ? 'Yes' : 'No'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {data.points.length > visiblePoints.length && (
                <p className="history-summary">
                  Showing the first {TABLE_LIMIT} records; narrow the time range to see fewer rows.
                </p>
              )}
            </div>
          </>
        )}
      </div>
    </section>
  );
}
