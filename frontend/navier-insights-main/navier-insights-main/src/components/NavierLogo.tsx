import { motion } from 'framer-motion';

export default function NavierLogo({ size = 'md' }: { size?: 'sm' | 'md' }) {
  const textSize = size === 'sm' ? 'text-lg' : 'text-xl';
  return (
    <div className="flex items-center gap-2">
      <svg width={size === 'sm' ? 24 : 28} height={size === 'sm' ? 24 : 28} viewBox="0 0 28 28" fill="none">
        <rect x="2" y="12" width="24" height="4" rx="2" fill="url(#logo-grad)" />
        <rect x="6" y="8" width="4" height="12" rx="2" fill="url(#logo-grad)" />
        <rect x="18" y="8" width="4" height="12" rx="2" fill="url(#logo-grad)" />
        <circle cx="14" cy="14" r="2" fill="url(#logo-grad)" opacity="0.8" />
        <defs>
          <linearGradient id="logo-grad" x1="0" y1="0" x2="28" y2="28">
            <stop stopColor="#0A4FD4" />
            <stop offset="1" stopColor="#00C9C8" />
          </linearGradient>
        </defs>
      </svg>
      <span className={`font-heading font-bold ${textSize} gradient-text`}>NAVIER</span>
    </div>
  );
}
