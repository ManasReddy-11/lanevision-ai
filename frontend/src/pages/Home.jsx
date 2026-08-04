import { useEffect, useState } from 'react';
import { Activity, Cpu, Sparkles } from 'lucide-react';
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
            Real-Time Intelligent Lane Detection &amp; Road Analysis
          </h1>
          <p className="mt-4 max-w-2xl text-lg text-slate-300">
            A modular platform for detecting, analyzing and visualizing road lanes using classical computer vision and deep learning.
          </p>

          <div className="mt-8 flex flex-wrap items-center gap-4">
            <div className="rounded-full border border-slate-700 bg-slate-800 px-4 py-2 text-sm text-slate-200">
              API Status: <span className="ml-2 font-semibold text-sky-400">{status}</span>
            </div>
          </div>
        </section>

        <section className="mt-10 grid gap-6 md:grid-cols-3">
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-6">
            <div className="flex items-center gap-3 text-sky-400">
              <Activity size={18} />
              <h2 className="text-lg font-semibold">Real-Time Detection</h2>
            </div>
            <p className="mt-3 text-sm text-slate-400">
              Designed for fast processing workflows and clear visual feedback.
            </p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-6">
            <div className="flex items-center gap-3 text-sky-400">
              <Cpu size={18} />
              <h2 className="text-lg font-semibold">OpenCV Processing</h2>
            </div>
            <p className="mt-3 text-sm text-slate-400">
              A robust classical computer vision foundation is being built first.
            </p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-6">
            <div className="flex items-center gap-3 text-sky-400">
              <Sparkles size={18} />
              <h2 className="text-lg font-semibold">AI-Ready Architecture</h2>
            </div>
            <p className="mt-3 text-sm text-slate-400">
              The system is structured to evolve into deep learning without disrupting the experience.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}

export default Home;
