import { useState, useEffect } from 'react';
import BuildingCard from '../components/BuildingCard';
import { api } from '../api/client';
import type { CampusOverviewResponse } from '../types';

interface CampusOverviewProps {
  onSelectBuilding: (id: string) => void;
}

export default function CampusOverview({ onSelectBuilding }: CampusOverviewProps) {
  const [data, setData] = useState<CampusOverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        setError(null);
        const result = await api.getCampusOverview();
        if (!cancelled) setData(result);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load campus data');
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();

    // Auto-refresh every 30s
    const interval = setInterval(load, 30_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const online = data?.buildings.filter((b) => b.status === 'online').length ?? 0;
  const total = data?.buildings.length ?? 0;

  return (
    <div>
      <div className="page-header">
        <h1>Campus Overview</h1>
        <p>
          Live resource consumption across all campus buildings.
          {data && (
            <>
              {' '}
              <span className="text-muted">
                {online}/{total} buildings online &middot; updated{' '}
                {new Date(data.updated_at).toLocaleTimeString()}
              </span>
            </>
          )}
        </p>
      </div>

      {loading && !data && (
        <div className="loading-state">
          <div className="spinner" />
          <span>Loading buildings...</span>
        </div>
      )}

      {error && (
        <div className="error-state">
          <span>Could not load campus data.</span>
          <span style={{ fontSize: '0.8125rem' }}>{error}</span>
        </div>
      )}

      {data && (
        <>
          {/* Summary strip */}
          <div className="grid-4 mb-24">
            <div className="card">
              <div className="card__title">Total buildings</div>
              <div className="card__value">{total}</div>
            </div>
            <div className="card">
              <div className="card__title">Online</div>
              <div className="card__value" style={{ color: 'var(--color-status-online)' }}>
                {online}
              </div>
            </div>
            <div className="card">
              <div className="card__title">Total electricity (now)</div>
              <div className="card__value">
                {data.buildings
                  .reduce((s, b) => s + b.electricity_kwh, 0)
                  .toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                <span className="card__sub" style={{ fontWeight: 400, fontSize: '1rem' }}>
                  {' '}kWh
                </span>
              </div>
            </div>
            <div className="card">
              <div className="card__title">Total water (now)</div>
              <div className="card__value">
                {data.buildings
                  .reduce((s, b) => s + b.water_litres, 0)
                  .toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                <span className="card__sub" style={{ fontWeight: 400, fontSize: '1rem' }}>
                  {' '}L
                </span>
              </div>
            </div>
          </div>

          {data.buildings.length === 0 ? (
            <div className="empty-state">
              <span>No buildings found.</span>
            </div>
          ) : (
            <div className="grid-auto">
              {data.buildings.map((b) => (
                <BuildingCard key={b.id} building={b} onClick={onSelectBuilding} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
