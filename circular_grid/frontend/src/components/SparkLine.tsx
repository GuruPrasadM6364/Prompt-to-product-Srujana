import { LineChart, Line, ResponsiveContainer, Tooltip } from 'recharts';
import type { SparkPoint } from '../types';

interface SparkLineProps {
  data: SparkPoint[];
  color?: string;
  height?: number;
}

export default function SparkLine({
  data,
  color = 'var(--color-primary)',
  height = 36,
}: SparkLineProps) {
  if (!data || data.length === 0) {
    return (
      <div
        style={{ height, display: 'flex', alignItems: 'center' }}
        className="text-muted"
        aria-label="No sparkline data"
      >
        <span style={{ fontSize: '0.75rem' }}>No data</span>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
        <Line
          type="monotone"
          dataKey="value"
          stroke={color}
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
        <Tooltip
          formatter={(val: number) => [val.toFixed(1), 'kWh']}
          labelFormatter={(label: string) => {
            try {
              return new Date(label).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            } catch {
              return label;
            }
          }}
          contentStyle={{
            fontSize: '0.75rem',
            padding: '4px 8px',
            border: '1px solid var(--color-border)',
            background: 'var(--color-surface)',
            color: 'var(--color-text)',
          }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
