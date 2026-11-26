import { useEffect, useState } from 'react';
import { Activity, Cpu, Sparkles } from 'lucide-react';
import { Link } from 'react-router-dom';
import { checkHealth } from '../services/api';

function Home() {
  const [status, setStatus] = useState('Checking...');

  useEffect(() => {
    const loadHealth = async () => {
      try {
        await checkHealth();
        setStatus('API Online');
      } catch (error) {
        setStatus('API Offline');
      }
    };

    loadHealth();
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <main className="mx-auto flex max-w-6xl flex-col px-6 py-16 lg:px-8">
        <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-8 shadow-2xl shadow-black/30">
          <div className="flex items-center gap-3 text-sm font-medium uppercase tracking-[0.3em] text-sky-400">
            <Activity size={18} />
            LaneVision AI
          </div>
          <h1 className="mt-6 text-4xl font-semibold text-white sm:text-5xl">
            Analyze lanes in recorded road footage
          </h1>
          <p className="mt-4 max-w-2xl text-lg text-slate-300">
            A modular platform for detecting, analyzing and visualizing road lanes using classical computer vision and deep learning.
          </p>

          <div className="mt-8 flex flex-wrap items-center gap-4">
            <Link to="/video" className="rounded-full bg-sky-500 px-5 py-2 font-semibold text-slate-950 hover:bg-sky-400">Analyze a video</Link>
            <Link to="/detect" className="rounded-full border border-slate-700 px-5 py-2 font-medium text-white hover:border-sky-400">Analyze an image</Link>
            <div className="rounded-full border border-slate-700 bg-slate-800 px-4 py-2 text-sm text-slate-200">
              API Status: <span className="ml-2 font-semibold text-sky-400">{status}</span>
            </div>
          </div>
        </section>

        <section className="mt-10 grid gap-6 md:grid-cols-3">
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-6">
            <div className="flex items-center gap-3 text-sky-400">
              <Activity size={18} />
              <h2 className="text-lg font-semibold">Video analysis</h2>
            </div>
            <p className="mt-3 text-sm text-slate-400">
              Compare the original and annotated footage and review frame-by-frame detection statistics.
            </p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-6">
            <div className="flex items-center gap-3 text-sky-400">
              <Cpu size={18} />
              <h2 className="text-lg font-semibold">OpenCV Processing</h2>
            </div>
            <p className="mt-3 text-sm text-slate-400">
              Choose between the classic and advanced lane detectors for uploaded footage.
            </p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-6">
            <div className="flex items-center gap-3 text-sky-400">
              <Sparkles size={18} />
              <h2 className="text-lg font-semibold">AI lane segmentation</h2>
            </div>
            <p className="mt-3 text-sm text-slate-400">
              Use the AI detector when a trained model is installed on the server.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}

export default Home;
