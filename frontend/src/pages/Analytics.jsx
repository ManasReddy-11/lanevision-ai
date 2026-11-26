import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Activity, BarChart3, Clock3, LineChart, ScanLine, Sparkles, TrendingUp } from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { getAnalytics, getDetectionDetail } from '../services/api';

const chartColors = ['#38bdf8', '#8b5cf6'];

function asNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function formatPercent(value) {
  const numericValue = asNumber(value);
  if (!Number.isFinite(numericValue)) {
    return '0.0%';
  }
  const percent = numericValue * 100;
  return `${percent.toFixed(1)}%`;
}

function formatProcessingTime(value) {
  const numericValue = asNumber(value);
  if (!Number.isFinite(numericValue) || numericValue <= 0) {
    return 'No data';
  }
  if (numericValue >= 1000) {
    return `${(numericValue / 1000).toFixed(2)} s`;
  }
  return `${numericValue.toFixed(2)} ms`;
}

function formatMetric(value, suffix = '') {
  const numericValue = asNumber(value);
  if (!Number.isFinite(numericValue) || numericValue <= 0) {
    return 'No data';
  }
  return `${numericValue.toFixed(2)}${suffix}`.trim();
}

function buildActivityData(timeline = []) {
  return timeline.map((entry) => ({
    date: entry.date,
    analyses: asNumber(entry.count),
  }));
}

function buildAlgorithmData(perAlgorithm = {}) {
  return [
    { name: 'Classic', value: asNumber(perAlgorithm.classic?.count) },
    { name: 'Advanced', value: asNumber(perAlgorithm.advanced?.count) },
  ];
}

function buildMediaData(analytics) {
  return [
    { name: 'Images', value: asNumber(analytics?.total_images) },
    { name: 'Videos', value: asNumber(analytics?.total_videos) },
  ];
}

function buildPerformanceData(perAlgorithm = {}) {
  return [
    { name: 'Classic', averageProcessingTimeMs: asNumber(perAlgorithm.classic?.average_processing_time_ms) },
    { name: 'Advanced', averageProcessingTimeMs: asNumber(perAlgorithm.advanced?.average_processing_time_ms) },
  ];
}

function Analytics() {
  const [analytics, setAnalytics] = useState(null);
  const [recentDetections, setRecentDetections] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const loadAnalytics = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getAnalytics();
      setAnalytics(data);

      const recentItems = Array.isArray(data?.recent_detections) ? data.recent_detections.slice(0, 5) : [];
      if (recentItems.length === 0) {
        setRecentDetections([]);
        return;
      }

      const detailResults = await Promise.all(
        recentItems.map(async (item) => {
          try {
            return await getDetectionDetail(item.id);
          } catch {
            return null;
          }
        })
      );

      const enriched = recentItems.map((item, index) => ({
        ...item,
        ...(detailResults[index] || {}),
      }));
      setRecentDetections(enriched);
    } catch {
      setError('Unable to load analytics.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadAnalytics();
  }, []);

  const totalAnalyses = useMemo(() => asNumber(analytics?.total_detections ?? analytics?.total_runs), [analytics]);
  const imageCount = useMemo(() => asNumber(analytics?.total_images), [analytics]);
  const videoCount = useMemo(() => asNumber(analytics?.total_videos), [analytics]);
  const averageConfidence = useMemo(() => asNumber(analytics?.average_confidence), [analytics]);
  const averageProcessingTime = useMemo(() => asNumber(analytics?.average_processing_time_ms), [analytics]);
  const advancedCount = useMemo(() => asNumber(analytics?.per_algorithm?.advanced?.count ?? analytics?.advanced_detections), [analytics]);
  const advancedUsagePercent = totalAnalyses > 0 ? (advancedCount / totalAnalyses) * 100 : 0;

  const activityData = useMemo(() => buildActivityData(analytics?.activity_timeline), [analytics]);
  const algorithmData = useMemo(() => buildAlgorithmData(analytics?.per_algorithm), [analytics]);
  const mediaData = useMemo(() => buildMediaData(analytics), [analytics]);
  const performanceData = useMemo(() => buildPerformanceData(analytics?.per_algorithm), [analytics]);

  const cards = [
    { label: 'Total Analyses', value: totalAnalyses.toString(), accent: 'text-sky-300' },
    { label: 'Images Processed', value: imageCount.toString(), accent: 'text-emerald-300' },
    { label: 'Videos Processed', value: videoCount.toString(), accent: 'text-violet-300' },
    { label: 'Average Confidence', value: formatPercent(averageConfidence), accent: 'text-amber-300' },
    { label: 'Average Processing Time', value: formatProcessingTime(averageProcessingTime), accent: 'text-cyan-300' },
    { label: 'Advanced Mode Usage', value: `${advancedCount} / ${totalAnalyses} (${advancedUsagePercent.toFixed(0)}%)`, accent: 'text-fuchsia-300' },
  ];

  const showEmptyState = !isLoading && !error && totalAnalyses === 0;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <main className="mx-auto max-w-7xl px-6 py-10 lg:px-8">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-8 shadow-2xl shadow-black/30">
          <div className="flex flex-wrap items-center gap-2 text-sm font-medium uppercase tracking-[0.3em] text-sky-400">
            <Activity size={16} />
            LaneVision Analytics
          </div>
          <h1 className="mt-3 text-3xl font-semibold text-white">Performance and detection insights from your processed images and videos.</h1>
          <p className="mt-3 max-w-3xl text-slate-400">
            Review persisted detection trends, compare algorithm performance, and monitor how your recent analyses are shaping up.
          </p>
        </div>

        {isLoading ? (
          <div className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {Array.from({ length: 6 }).map((_, index) => (
              <div key={index} className="h-32 animate-pulse rounded-2xl border border-slate-800 bg-slate-900/70" />
            ))}
          </div>
        ) : null}

        {error ? (
          <div className="mt-8 rounded-2xl border border-amber-700/40 bg-amber-900/20 p-6 text-amber-100">
            <h2 className="text-lg font-semibold">Unable to load analytics.</h2>
            <p className="mt-2 text-sm text-amber-200">The analytics API could not be reached. Please retry once the backend is available.</p>
            <button
              type="button"
              onClick={() => loadAnalytics()}
              className="mt-4 rounded-full border border-amber-600 bg-amber-500/10 px-4 py-2 text-sm font-medium text-amber-200 transition hover:bg-amber-500/20"
            >
              Retry
            </button>
          </div>
        ) : null}

        {!isLoading && !error ? (
          <>
            {showEmptyState ? (
              <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/80 p-8 text-center">
                <h2 className="text-xl font-semibold text-white">No analytics available yet.</h2>
                <p className="mt-3 text-slate-400">Process an image or video to generate detection analytics and start tracking your performance.</p>
                <div className="mt-6 flex flex-wrap justify-center gap-3">
                  <Link to="/detect" className="rounded-full bg-sky-500 px-4 py-2 text-sm font-medium text-slate-950 transition hover:bg-sky-400">
                    Detect Image
                  </Link>
                  <Link to="/video" className="rounded-full border border-slate-700 px-4 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-800">
                    Detect Video
                  </Link>
                </div>
              </div>
            ) : (
              <>
                <div className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                  {cards.map((card) => (
                    <div key={card.label} className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
                      <p className="text-sm text-slate-400">{card.label}</p>
                      <p className={`mt-3 text-2xl font-semibold ${card.accent}`}>{card.value}</p>
                    </div>
                  ))}
                </div>

                <div className="mt-8 grid gap-6 xl:grid-cols-2">
                  <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6">
                    <div className="flex items-center gap-2 text-slate-200">
                      <LineChart size={18} className="text-sky-400" />
                      <h2 className="text-xl font-semibold">Detection Activity</h2>
                    </div>
                    <p className="mt-2 text-sm text-slate-400">Daily analysis volume from your persisted detections.</p>
                    <div className="mt-6 h-72">
                      <ResponsiveContainer width="100%" height="100%">
                        <RechartsLineChart data={activityData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                          <XAxis dataKey="date" stroke="#94a3b8" tickLine={false} axisLine={false} />
                          <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} />
                          <Tooltip />
                          <Line type="monotone" dataKey="analyses" stroke="#38bdf8" strokeWidth={2} dot={{ r: 3 }} />
                        </RechartsLineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6">
                    <div className="flex items-center gap-2 text-slate-200">
                      <BarChart3 size={18} className="text-violet-400" />
                      <h2 className="text-xl font-semibold">Algorithm Usage</h2>
                    </div>
                    <p className="mt-2 text-sm text-slate-400">Classic versus advanced detections across your saved runs.</p>
                    <div className="mt-6 h-72">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={algorithmData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                          <XAxis dataKey="name" stroke="#94a3b8" tickLine={false} axisLine={false} />
                          <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} />
                          <Tooltip />
                          <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                            {algorithmData.map((entry, index) => (
                              <Cell key={`${entry.name}-${index}`} fill={chartColors[index % chartColors.length]} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                <div className="mt-8 grid gap-6 xl:grid-cols-2">
                  <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6">
                    <div className="flex items-center gap-2 text-slate-200">
                      <ScanLine size={18} className="text-emerald-400" />
                      <h2 className="text-xl font-semibold">Media Types</h2>
                    </div>
                    <p className="mt-2 text-sm text-slate-400">How your persisted detections are split between images and videos.</p>
                    <div className="mt-6 h-72">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie data={mediaData} dataKey="value" nameKey="name" outerRadius={90} label>
                            {mediaData.map((entry, index) => (
                              <Cell key={`${entry.name}-${index}`} fill={chartColors[index % chartColors.length]} />
                            ))}
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6">
                    <div className="flex items-center gap-2 text-slate-200">
                      <TrendingUp size={18} className="text-amber-400" />
                      <h2 className="text-xl font-semibold">Average Processing Time</h2>
                    </div>
                    <p className="mt-2 text-sm text-slate-400">Lower processing time means faster processing for the same workflow.</p>
                    <div className="mt-6 h-72">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={performanceData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                          <XAxis dataKey="name" stroke="#94a3b8" tickLine={false} axisLine={false} />
                          <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} />
                          <Tooltip formatter={(value) => [formatProcessingTime(value), 'Average Processing Time']} />
                          <Bar dataKey="averageProcessingTimeMs" radius={[8, 8, 0, 0]} fill="#8b5cf6" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                {analytics?.average_processing_fps !== undefined || analytics?.average_detection_success_rate !== undefined || analytics?.average_both_lane_detection_rate !== undefined ? (
                  <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/80 p-6">
                    <div className="flex items-center gap-2 text-slate-200">
                      <Sparkles size={18} className="text-cyan-400" />
                      <h2 className="text-xl font-semibold">Video Performance</h2>
                    </div>
                    <div className="mt-6 grid gap-4 md:grid-cols-3">
                      {analytics?.average_processing_fps !== undefined ? (
                        <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                          <p className="text-sm text-slate-400">Average Processing FPS</p>
                          <p className="mt-2 text-xl font-semibold text-cyan-300">{formatMetric(analytics.average_processing_fps)}</p>
                        </div>
                      ) : null}
                      {analytics?.average_detection_success_rate !== undefined ? (
                        <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                          <p className="text-sm text-slate-400">Detection Success Rate</p>
                          <p className="mt-2 text-xl font-semibold text-emerald-300">{formatPercent(analytics.average_detection_success_rate)}</p>
                        </div>
                      ) : null}
                      {analytics?.average_both_lane_detection_rate !== undefined ? (
                        <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                          <p className="text-sm text-slate-400">Both Lanes Detected</p>
                          <p className="mt-2 text-xl font-semibold text-violet-300">{formatPercent(analytics.average_both_lane_detection_rate)}</p>
                        </div>
                      ) : null}
                    </div>
                  </div>
                ) : null}

                {recentDetections.length > 0 ? (
                  <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/80 p-6">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <h2 className="text-xl font-semibold text-white">Recent Analyses</h2>
                        <p className="mt-2 text-sm text-slate-400">The latest persisted detections from your history feed.</p>
                      </div>
                      <Link to="/history" className="text-sm font-medium text-sky-400 transition hover:text-sky-300">
                        View history
                      </Link>
                    </div>
                    <div className="mt-6 space-y-3">
                      {recentDetections.map((item) => (
                        <div key={item.id} className="flex flex-col gap-3 rounded-2xl border border-slate-800 bg-slate-950/70 p-4 md:flex-row md:items-center md:justify-between">
                          <div>
                            <div className="flex items-center gap-2">
                              <Clock3 size={15} className="text-sky-400" />
                              <span className="text-sm font-semibold text-white">{item.original_filename || item.filename || 'Detection'}</span>
                              <span className="rounded-full bg-slate-800 px-2 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-300">{item.media_type || 'unknown'}</span>
                            </div>
                            <p className="mt-2 text-sm text-slate-400">
                              {item.algorithm || 'classic'} • {item.status || 'unknown'} • {item.created_at || 'unknown'}
                            </p>
                          </div>
                          <div className="text-sm text-slate-400">
                            <div>Confidence: {item.confidence !== undefined ? formatPercent(item.confidence) : 'No data'}</div>
                            <div>Processing Time: {item.processing_time_ms !== undefined ? formatProcessingTime(item.processing_time_ms) : 'No data'}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}

                <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/70 p-6 text-sm text-slate-400">
                  <p>
                    Confidence values are heuristic scores produced by each detection algorithm and should not be interpreted as calibrated probabilities.
                  </p>
                </div>
              </>
            )}
          </>
        ) : null}
      </main>
    </div>
  );
}

export default Analytics;
