import StatusDot from './StatusDot';
import EstimatedBadge from './EstimatedBadge';
import SparkLine from './SparkLine';
import type { BuildingSummary } from '../types';

interface BuildingCardProps {
  building: BuildingSummary;
  onClick: (id: string) => void;
}

function fmt(n: number, digits = 0): string {
  return n.toLocaleString('en-IN', { maximumFractionDigits: digits });
}

export default function BuildingCard({ building, onClick }: BuildingCardProps) {
  const {
    id,
    name,
    status,
    electricity_kwh,
    electricity_estimated,
    water_litres,
    water_estimated,
    footfall,
    footfall_estimated,
    sparkline,
  } = building;

  return (
    <div
      className="card card--interactive"
      role="button"
      tabIndex={0}
      aria-label={`View details for ${name}`}
      onClick={() => onClick(id)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick(id);
        }
      }}
    >
      {/* Header */}
      <div className="flex items-center gap-8 mb-8">
        <StatusDot status={status} />
        <span
          className="font-semibold"
          style={{ flex: 1, fontSize: '0.9375rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
        >
          {name}
        </span>
        <span
          className="badge"
          style={{
            background:
              status === 'online' ? 'var(--color-surface-2)' : undefined,
            color: 'var(--color-text-muted)',
            fontSize: '0.6875rem',
          }}
        >
          {status}
        </span>
      </div>

      {/* Metrics */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '8px',
          marginBottom: '12px',
        }}
      >
        {/* Electricity */}
        <div>
          <div className="card__title">Electricity</div>
          <div
            style={{
              display: 'flex',
              alignItems: 'baseline',
              gap: '4px',
              flexWrap: 'wrap',
            }}
          >
            <span className="card__value" style={{ fontSize: '1.125rem' }}>
              {fmt(electricity_kwh, 1)}
            </span>
            <span className="card__sub">kWh</span>
          </div>
          {electricity_estimated && <EstimatedBadge />}
        </div>

        {/* Water */}
        <div>
          <div className="card__title">Water</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px', flexWrap: 'wrap' }}>
            <span className="card__value" style={{ fontSize: '1.125rem' }}>
              {fmt(water_litres)}
            </span>
            <span className="card__sub">L</span>
          </div>
          {water_estimated && <EstimatedBadge />}
        </div>

        {/* Footfall */}
        <div>
          <div className="card__title">Footfall</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px', flexWrap: 'wrap' }}>
            <span className="card__value" style={{ fontSize: '1.125rem' }}>
              {fmt(footfall)}
            </span>
            <span className="card__sub">ppl</span>
          </div>
          {footfall_estimated && <EstimatedBadge />}
        </div>
      </div>

      {/* Sparkline: last 24h electricity */}
      <div>
        <div
          style={{
            fontSize: '0.6875rem',
            color: 'var(--color-text-muted)',
            marginBottom: '2px',
          }}
        >
          Electricity - last 24h (kWh)
        </div>
        <SparkLine data={sparkline} />
      </div>
    </div>
  );
}
