import type { BuildingStatus } from '../types';

interface StatusDotProps {
  status: BuildingStatus;
  size?: number;
}

const COLOR: Record<BuildingStatus, string> = {
  online: 'var(--color-status-online)',
  stale: 'var(--color-status-stale)',
  offline: 'var(--color-status-offline)',
};

const LABEL: Record<BuildingStatus, string> = {
  online: 'Online',
  stale: 'Stale',
  offline: 'Offline',
};

export default function StatusDot({ status, size = 10 }: StatusDotProps) {
  return (
    <span
      role="img"
      aria-label={LABEL[status]}
      title={LABEL[status]}
      style={{
        display: 'inline-block',
        width: size,
        height: size,
        borderRadius: '50%',
        backgroundColor: COLOR[status],
        flexShrink: 0,
      }}
    />
  );
}
