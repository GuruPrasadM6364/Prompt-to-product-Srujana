import { useState, useEffect, useCallback } from 'react';
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
import type { BuildingForecastResponse, ForecastStream, ForecastHorizon } from '../types';

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

function fmtTime(ts: string): string {
  try {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return ts;
  }
}

export default function BuildingDetail({ buildingId, onBack }: BuildingDetailProps) {
  const [data, setData] = useState<BuildingForecastResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stream, setStream] = useState<ForecastStream>('electricity_kwh');
  const [horizon, setHorizon] = useState<ForecastHorizon>('6h');

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

  return (
    <div>
      {/* Back button */}
      <button className="btn-back" onClick={onBack} type="button">
        <span aria-hidden="true">&#8592;</span> All Buildings
      </button>

      <div className="page-header">
        <h1>{data ? data.building_name : `Building ${buildingId}`}</h1>
        <p>Actual vs forecast with uncertainty band. Calendar events shown as vertical markers.</p>
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
    </div>
  );
}
