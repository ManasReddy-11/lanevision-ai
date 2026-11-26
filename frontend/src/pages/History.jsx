import { useEffect, useMemo, useState } from 'react';
import { Activity, Clock3, ScanLine } from 'lucide-react';
import { getAnalytics, getDetectionHistory, mediaUrl } from '../services/api';

function History() {
  const [runs, setRuns] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [history, analytics] = await Promise.all([
          getDetectionHistory(),
          getAnalytics(),
        ]);
        setRuns(history.items);
        setAnalytics(analytics);
      } catch (err) {
        setError(err?.message || 'Unable to load history');
      }
    };
    load();
  }, []);

  const summaryCards = useMemo(() => [
    { label: 'Total Runs', value: analytics?.total_runs ?? 0 },
    { label: 'Successful', value: analytics?.successful_runs ?? 0 },
    { label: 'Failed', value: analytics?.failed_runs ?? 0 },
    { label: 'Avg Confidence', value: analytics ? `${(analytics.average_confidence * 100).toFixed(1)}%` : '0.0%' },
  ], [analytics]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <main className="mx-auto max-w-7xl px-6 py-10 lg:px-8">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-8 shadow-2xl shadow-black/30">
          <div className="flex items-center gap-2 text-sm font-medium uppercase tracking-[0.3em] text-sky-400">
            <Activity size={16} />
            Detection History
          </div>
          <h1 className="mt-3 text-3xl font-semibold text-white">Review recent detection runs and analytics</h1>
          <p className="mt-3 max-w-2xl text-slate-400">Every successful or failed image and video request is stored locally so you can inspect recent runs and spot trends.</p>
        </div>

        <div className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {summaryCards.map((card) => (
            <div key={card.label} className="rounded-2xl border border-slate-800 bg-slate-900/80 p-4">
              <p className="text-sm text-slate-400">{card.label}</p>
              <p className="mt-2 text-2xl font-semibold text-white">{card.value}</p>
            </div>
          ))}
        </div>

        {error ? <div className="mt-6 rounded-2xl border border-amber-700/40 bg-amber-900/20 p-4 text-sm text-amber-200">{error}</div> : null}

        <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/80 p-6">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-white">Recent Runs</h2>
            <div className="flex items-center gap-2 text-sm text-slate-400"><Clock3 size={16} /> Latest activity</div>
          </div>
          <div className="mt-6 space-y-3">
            {runs.length === 0 ? <p className="text-sm text-slate-400">No runs recorded yet.</p> : runs.map((run) => (
              <div key={run.id} className="flex flex-col gap-3 rounded-2xl border border-slate-800 bg-slate-950/60 p-4 md:flex-row md:items-center md:justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <ScanLine size={16} className="text-sky-400" />
                    <span className="text-sm font-semibold text-white">{run.filename}</span>
                    <span className={`rounded-full px-2 py-1 text-xs ${run.status === 'success' ? 'bg-emerald-500/15 text-emerald-300' : 'bg-rose-500/15 text-rose-300'}`}>{run.status}</span>
                  </div>
                  <p className="mt-2 text-sm text-slate-400">{run.kind} • {run.algorithm} • {run.created_at}</p>
                </div>
                <div className="text-sm text-slate-400">
                  <div>Confidence: {run.confidence == null ? 'N/A' : `${(run.confidence * 100).toFixed(1)}%`}</div>
                  {run.result_url && <a className="text-sky-400" href={mediaUrl(run.result_url)} target="_blank" rel="noreferrer">Open result</a>}
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}

export default History;
