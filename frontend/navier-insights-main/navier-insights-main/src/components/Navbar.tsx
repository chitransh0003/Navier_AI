import { useState, useRef, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Bell, User, ChevronDown } from 'lucide-react';
import NavierLogo from './NavierLogo';

const ALERTS = [
  { text: 'WARNING on SEG-04', time: '2 mins ago' },
  { text: 'SCS dropped to 71% on RKL-03', time: '14 mins ago' },
  { text: 'Batch change detected MS→HSD', time: '1 hr ago' },
];

const NAV_LINKS = [
  { path: '/dashboard', label: 'Dashboard' },
  { path: '/simulate', label: 'Simulate' },
  { path: '/analytics', label: 'Analytics' },
];

export default function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const [showNotifs, setShowNotifs] = useState(false);
  const [showUser, setShowUser] = useState(false);
  const notifRef = useRef<HTMLDivElement>(null);
  const userRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (notifRef.current && !notifRef.current.contains(e.target as Node)) setShowNotifs(false);
      if (userRef.current && !userRef.current.contains(e.target as Node)) setShowUser(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-16 bg-card/90 backdrop-blur-md shadow-sm border-b border-border flex items-center px-6">
      <Link to="/dashboard"><NavierLogo size="sm" /></Link>

      <div className="flex items-center gap-6 ml-10">
        {NAV_LINKS.map(link => (
          <Link
            key={link.path}
            to={link.path}
            className={`text-sm font-medium transition-colors pb-1 border-b-2 ${
              location.pathname === link.path
                ? 'text-primary border-primary'
                : 'text-muted-foreground border-transparent hover:text-foreground'
            }`}
          >
            {link.label}
          </Link>
        ))}
      </div>

      <div className="ml-auto flex items-center gap-3">
        <motion.button
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
          onClick={() => navigate('/simulate')}
          className="gradient-bg text-primary-foreground font-semibold text-sm rounded-full px-5 py-2"
        >
          Run Simulation
        </motion.button>

        <div ref={notifRef} className="relative">
          <button onClick={() => { setShowNotifs(!showNotifs); setShowUser(false); }} className="relative p-2 rounded-full hover:bg-muted transition-colors">
            <Bell size={20} className="text-muted-foreground" />
            <span className="absolute -top-0.5 -right-0.5 w-4 h-4 rounded-full bg-destructive text-destructive-foreground text-[10px] flex items-center justify-center font-bold">3</span>
          </button>
          <AnimatePresence>
            {showNotifs && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95, y: -5 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: -5 }}
                className="absolute top-full right-0 mt-2 w-80 bg-card rounded-xl shadow-xl border border-border overflow-hidden"
              >
                <div className="px-4 py-3 border-b border-border font-heading font-semibold text-sm">Notifications</div>
                {ALERTS.map((a, i) => (
                  <div key={i} className="px-4 py-3 hover:bg-muted/50 transition-colors border-b border-border last:border-0">
                    <p className="text-sm font-medium text-foreground">{a.text}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">{a.time}</p>
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div ref={userRef} className="relative">
          <button onClick={() => { setShowUser(!showUser); setShowNotifs(false); }} className="w-9 h-9 rounded-full gradient-bg flex items-center justify-center text-primary-foreground text-xs font-bold">
            OP
          </button>
          <AnimatePresence>
            {showUser && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95, y: -5 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: -5 }}
                className="absolute top-full right-0 mt-2 w-44 bg-card rounded-xl shadow-xl border border-border overflow-hidden"
              >
                {['Profile', 'Settings', 'Sign Out'].map(item => (
                  <button key={item} className="w-full text-left px-4 py-2.5 text-sm hover:bg-muted/50 transition-colors text-foreground">
                    {item}
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </nav>
  );
}
