import { useState, useEffect, useCallback } from 'react';
import type { FormEvent } from 'react';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts';
import { api } from '../api/client';
import type {
  BuildingForecastResponse,
  BuildingHistoryResponse,
  ForecastStream,
  ForecastHorizon,
} from '../types';

interface BuildingDetailProps {
  buildingId: string;
  onBack: () => void;
}

const STREAM_LABELS: Record<ForecastStream, string> = {
  electricity_kwh: 'Electricity (kWh)',
  water_litres: 'Water (litres)',
  footfall: 'Footfall (people)',
};

const HORIZON_LABELS: Record<ForecastHorizon, string> = {
  '6h': '6 hours',
  '12h': '12 hours',
};

type HistoryStream = ForecastStream | 'solar_generation_kwh';

const HISTORY_STREAM_LABELS: Record<HistoryStream, string> = {
  ...STREAM_LABELS,
  solar_generation_kwh: 'Solar generation (kWh)',
};

function fmtTime(ts: string): string {
  try {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return ts;
  }
}

function fmtDateTime(ts: string): string {
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return ts || 'Unavailable';
  return date.toLocaleString([], {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function toDateTimeLocal(ts: string | null): string {
  return ts ? ts.slice(0, 16) : '';
}

export default function BuildingDetail({ buildingId, onBack }: BuildingDetailProps) {
  const [data, setData] = useState<BuildingForecastResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stream, setStream] = useState<ForecastStream>('electricity_kwh');
  const [horizon, setHorizon] = useState<ForecastHorizon>('6h');
  const [historyStream, setHistoryStream] = useState<HistoryStream>('electricity_kwh');
  const [historyStart, setHistoryStart] = useState('');
  const [historyEnd, setHistoryEnd] = useState('');
  const [historyRange, setHistoryRange] = useState<{ start: string; end: string } | null>(null);
  const [historyData, setHistoryData] = useState<BuildingHistoryResponse | null>(null);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await api.getBuildingForecast(buildingId, stream, horizon);
      setData(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load forecast');
    } finally {
      setLoading(false);
    }
  }, [buildingId, stream, horizon]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    let cancelled = false;

    async function loadHistory() {
      try {
        setHistoryLoading(true);
        setHistoryError(null);
        const result = await api.getBuildingHistory(
          buildingId,
          historyStream,
          historyRange?.start,
          historyRange?.end
        );
        if (cancelled) return;
        setHistoryData(result);
        if (!historyRange) {
          setHistoryStart(toDateTimeLocal(result.start));
          setHistoryEnd(toDateTimeLocal(result.end));
        }
      } catch (e) {
        if (!cancelled) {
          setHistoryError(e instanceof Error ? e.message : 'Failed to load historical data');
        }
      } finally {
        if (!cancelled) setHistoryLoading(false);
      }
    }

    loadHistory();
    return () => {
      cancelled = true;
    };
  }, [buildingId, historyStream, historyRange]);

  function applyHistoryRange(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!historyStart || !historyEnd) {
      setHistoryError('Choose both a start and end date and time.');
      return;
    }
    if (historyStart > historyEnd) {
      setHistoryError('Start date and time must be before the end.');
      return;
    }
    setHistoryRange({ start: historyStart, end: historyEnd });
  }

  const numericHistory = historyData?.points
    .map((point) => point.value)
    .filter((value): value is number => value !== null) ?? [];
  const visibleHistoryRows = historyData?.points.slice(0, 100) ?? [];

  return (
    <div>
      {/* Back button */}
      <button className="btn-back" onClick={onBack} type="button">
        <span aria-hidden="true">&#8592;</span> All Buildings
      </button>

      <div className="page-header">
        <h1>{data ? data.building_name : `Building ${buildingId}`}</h1>
        <p>Review measured history and compare it with the available forecast.</p>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-12 mb-16" style={{ flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <label htmlFor="stream-select" style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', fontWeight: 600 }}>
            Stream
          </label>
          <select
            id="stream-select"
            className="select-control"
            value={stream}
            onChange={(e) => setStream(e.target.value as ForecastStream)}
          >
            {(Object.keys(STREAM_LABELS) as ForecastStream[]).map((s) => (
              <option key={s} value={s}>
                {STREAM_LABELS[s]}
              </option>
            ))}
          </select>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <label htmlFor="horizon-select" style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', fontWeight: 600 }}>
            Horizon
          </label>
          <select
            id="horizon-select"
            className="select-control"
            value={horizon}
            onChange={(e) => setHorizon(e.target.value as ForecastHorizon)}
          >
            {(Object.keys(HORIZON_LABELS) as ForecastHorizon[]).map((h) => (
              <option key={h} value={h}>
                {HORIZON_LABELS[h]}
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading && (
        <div className="loading-state">
          <div className="spinner" />
          <span>Loading forecast...</span>
        </div>
      )}

      {error && (
        <div className="error-state">
          <span>Could not load forecast.</span>
          <span style={{ fontSize: '0.8125rem' }}>{error}</span>
        </div>
      )}

      {data && !loading && (
        <div className="card">
          <div className="section-header">
            <h2>
              {STREAM_LABELS[stream]} - {HORIZON_LABELS[horizon]} horizon
            </h2>
          </div>

          {data.series.length === 0 ? (
            <div className="empty-state">
              <span>No forecast data available for this selection.</span>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={380}>
              <ComposedChart
                data={data.series}
                margin={{ top: 8, right: 16, bottom: 8, left: 8 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis
                  dataKey="ts"
                  tickFormatter={fmtTime}
                  tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }}
                  tickLine={false}
                  axisLine={false}
                  width={56}
                />
                <Tooltip
                  formatter={(val: number, name: string) => {
                    const labels: Record<string, string> = {
                      actual: 'Actual',
                      forecast: 'Forecast',
                      lower: 'Lower bound',
                      upper: 'Upper bound',
                    };
                    return [val !== null ? val.toFixed(1) : 'N/A', labels[name] ?? name];
                  }}
                  labelFormatter={fmtTime}
                  contentStyle={{
                    fontSize: '0.8125rem',
                    border: '1px solid var(--color-border)',
                    background: 'var(--color-surface)',
                    color: 'var(--color-text)',
                  }}
                />
                <Legend
                  wrapperStyle={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)' }}
                />

                {/* Uncertainty band */}
                <Area
                  type="monotone"
                  dataKey="upper"
                  stroke="none"
                  fill="#1a6e3c"
                  fillOpacity={0.08}
                  legendType="none"
                  isAnimationActive={false}
                  name="upper"
                />
                <Area
                  type="monotone"
                  dataKey="lower"
                  stroke="none"
                  fill="#ffffff"
                  fillOpacity={1}
                  legendType="none"
                  isAnimationActive={false}
                  name="lower"
                />

                {/* Forecast line */}
                <Line
                  type="monotone"
                  dataKey="forecast"
                  stroke="#1a6e3c"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                  strokeDasharray="5 3"
                  name="forecast"
                />

                {/* Actual line */}
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#343a40"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                  connectNulls={false}
                  name="actual"
                />

                {/* Calendar event markers */}
                {data.calendar_events.map((ev) => (
                  <ReferenceLine
                    key={ev.ts}
                    x={ev.ts}
                    stroke="var(--color-status-stale)"
                    strokeDasharray="4 2"
                    label={{
                      value: ev.label,
                      position: 'top',
                      fontSize: 10,
                      fill: 'var(--color-status-stale)',
                    }}
                  />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
          )}

          <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '8px' }}>
            Shaded area shows 90% prediction interval. Dashed line is model forecast; solid line is measured actual.
          </p>
        </div>
      )}

      <section className="card history-card">
        <div className="section-header">
          <h2>Historical readings</h2>
          {historyData?.available_start && historyData.available_end && (
            <span className="text-muted">
              Available {fmtDateTime(historyData.available_start)} to{' '}
              {fmtDateTime(historyData.available_end)}
            </span>
          )}
        </div>

        <form className="history-controls" onSubmit={applyHistoryRange}>
          <div className="history-control">
            <label htmlFor="history-stream">Resource</label>
            <select
              id="history-stream"
              className="select-control"
              value={historyStream}
              onChange={(event) => setHistoryStream(event.target.value as HistoryStream)}
            >
              {(Object.keys(HISTORY_STREAM_LABELS) as HistoryStream[]).map((item) => (
                <option key={item} value={item}>{HISTORY_STREAM_LABELS[item]}</option>
              ))}
            </select>
          </div>
          <div className="history-control">
            <label htmlFor="history-start">From date and time</label>
            <input
              id="history-start"
              type="datetime-local"
              value={historyStart}
              max={toDateTimeLocal(historyData?.available_end ?? null)}
              onChange={(event) => setHistoryStart(event.target.value)}
              required
            />
          </div>
          <div className="history-control">
            <label htmlFor="history-end">To date and time</label>
            <input
              id="history-end"
              type="datetime-local"
              value={historyEnd}
              min={toDateTimeLocal(historyData?.available_start ?? null)}
              max={toDateTimeLocal(historyData?.available_end ?? null)}
              onChange={(event) => setHistoryEnd(event.target.value)}
              required
            />
          </div>
          <button className="history-submit" type="submit">Show history</button>
        </form>

        {historyLoading && (
          <div className="loading-state">
            <div className="spinner" />
            <span>Loading historical readings...</span>
          </div>
        )}
        {historyError && <div className="error-state">{historyError}</div>}
        {!historyLoading && !historyError && historyData && historyData.points.length === 0 && (
          <div className="empty-state">
            <span>No readings were recorded for this building and time range.</span>
          </div>
        )}
        {!historyLoading && !historyError && historyData && historyData.points.length > 0 && (
          <>
            <p className="history-summary">
              {historyData.points.length} hourly records
              {numericHistory.length > 0 && (
                <> · mean {HISTORY_STREAM_LABELS[historyStream].split(' (')[0].toLowerCase()} value{' '}
                  {(
                    numericHistory.reduce((sum, value) => sum + value, 0) / numericHistory.length
                  ).toLocaleString(undefined, { maximumFractionDigits: 2 })}
                </>
              )}
              {' '}· values are from the hourly dataset; missing readings remain blank.
            </p>
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart data={historyData.points} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis
                  dataKey="ts"
                  tickFormatter={fmtTime}
                  tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  labelFormatter={fmtDateTime}
                  formatter={(value) => [
                    value == null ? 'No reading' : Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 }),
                    HISTORY_STREAM_LABELS[historyStream],
                  ]}
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
                  name={HISTORY_STREAM_LABELS[historyStream]}
                  stroke="#1a6e3c"
                  strokeWidth={2}
                  dot={false}
                  connectNulls={false}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>

            <div className="history-table-wrap">
              <table className="data-table">
                <thead>
                  <tr><th>Date and time</th><th>Reading</th><th>Source</th><th>Estimated</th></tr>
                </thead>
                <tbody>
                  {visibleHistoryRows.map((point) => (
                    <tr key={point.ts}>
                      <td>{fmtDateTime(point.ts)}</td>
                      <td>{point.value === null ? 'No reading' : point.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                      <td>{point.source}</td>
                      <td>{point.imputed ? 'Yes' : 'No'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {historyData.points.length > visibleHistoryRows.length && (
                <p className="history-summary">
                  Showing the first 100 records. Narrow the date/time range to inspect individual readings.
                </p>
              )}
            </div>
          </>
        )}
      </section>
    </div>
  );
}
