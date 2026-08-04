import { Link, useLocation } from 'react-router-dom';

function NavBar() {
  const location = useLocation();
  const links = [
    { path: '/', label: 'Home' },
    { path: '/detect', label: 'Detect Image' },
    { path: '/video', label: 'Detect Video' },
    { path: '/history', label: 'History' },
    { path: '/analytics', label: 'Analytics' },
  ];

  return (
    <nav className="border-b border-slate-800 bg-slate-950/90">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4 lg:px-8">
        <Link to="/" className="text-lg font-semibold tracking-wide text-white">
          LaneVision AI
        </Link>
        <div className="flex gap-4 text-sm text-slate-300">
          {links.map((link) => (
            <Link
              key={link.path}
              to={link.path}
              className={`rounded-full px-3 py-2 transition ${location.pathname === link.path ? 'bg-slate-800 text-sky-400' : 'hover:bg-slate-800 hover:text-white'}`}
            >
              {link.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}

export default NavBar;
