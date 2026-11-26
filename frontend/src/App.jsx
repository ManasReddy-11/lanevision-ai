import { lazy, Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';
import NavBar from './components/NavBar';
import Home from './pages/Home';
import Detect from './pages/Detect';
import History from './pages/History';
import Video from './pages/Video';

const Analytics = lazy(() => import('./pages/Analytics'));

function App() {
  return (
    <>
      <NavBar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/detect" element={<Detect />} />
        <Route path="/video" element={<Video />} />
        <Route path="/history" element={<History />} />
        <Route path="/analytics" element={<Suspense fallback={<p role="status" className="p-6 text-slate-300">Loading analytics…</p>}><Analytics /></Suspense>} />
      </Routes>
    </>
  );
}

export default App;
