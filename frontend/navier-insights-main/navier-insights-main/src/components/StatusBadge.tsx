interface StatusBadgeProps {
  status: 'SAFE' | 'WARNING' | 'CRITICAL';
  size?: 'sm' | 'md';
}

export default function StatusBadge({ status, size = 'sm' }: StatusBadgeProps) {
  const styles = {
    SAFE: 'status-safe',
    WARNING: 'status-warning',
    CRITICAL: 'status-critical',
  };
  const px = size === 'sm' ? 'px-2.5 py-0.5 text-xs' : 'px-3 py-1 text-sm';
  return (
    <span className={`inline-flex items-center font-semibold rounded-full border ${styles[status]} ${px}`}>
      {status}
    </span>
  );
}
