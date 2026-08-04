import { useMemo, useState } from 'react';
import { AlertTriangle, Download, Info, ScanLine } from 'lucide-react';
import VideoUploader from '../components/VideoUploader';
import { detectVideo } from '../services/api';

function Video() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [algorithm, setAlgorithm] = useState('classic');

  const handleFileChange = ({ file, error: nextError }) => {
    setSelectedFile(file);
    setError(nextError || null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(file ? URL.createObjectURL(file) : null);
    if (!file) {
      setResult(null);
    }
  };

  const analyzeVideo = async () => {
    if (!selectedFile) {
      setError('Please select a video first.');
      return;
    }

    setError(null);
    setIsProcessing(true);
    try {
      const response = await detectVideo(selectedFile, algorithm);
      setResult(response.data);
    } catch (err) {
      const detail = err?.response?.data?.detail || err?.message || 'Unable to process the video.';
      setError(detail);
    } finally {
      setIsProcessing(false);
    }
  };

  const resetState = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setError(null);
    setResult(null);
    setPreviewUrl(null);
  };

  const apiBaseUrl = useMemo(() => import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000', []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <main className="mx-auto max-w-7xl px-6 py-10 lg:px-8">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-8 shadow-2xl shadow-black/30">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <div className="flex items-center gap-2 text-sm font-medium uppercase tracking-[0.3em] text-sky-400">
                <ScanLine size={16} />
                Video Detection Workspace
              </div>
              <h1 className="mt-3 text-3xl font-semibold text-white">Upload a road video and analyze lane boundaries frame by frame</h1>
              <p className="mt-3 max-w-2xl text-slate-400">The backend processes every frame with the classic OpenCV lane detector and applies temporal smoothing to reduce flicker.</p>
            </div>
            <div className="rounded-full border border-slate-700 bg-slate-800 px-4 py-2 text-sm text-slate-300">{algorithm === 'advanced' ? 'Advanced algorithm • curved-lane tracking' : 'Classic algorithm • temporal smoothing'}</div>
          </div>
        </div>

        <div className="mt-8 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <VideoUploader file={selectedFile} onFileChange={handleFileChange} onAnalyze={analyzeVideo} isProcessing={isProcessing} error={error} />
          <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6">
            <div className="mb-4">
              <label className="text-sm font-medium text-slate-300">
                Detection Algorithm
                <select value={algorithm} onChange={(event) => setAlgorithm(event.target.value)} className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white">
                  <option value="classic">Classic OpenCV</option>
                  <option value="advanced">Advanced Lane Detection</option>
                </select>
              </label>
            </div>
            <h2 className="text-xl font-semibold text-white">Temporal Tracking</h2>
            <p className="mt-2 text-sm text-slate-400">Lane detections are smoothed between adjacent frames to reduce visual jitter. Temporary missing detections may persist briefly for continuity.</p>
            <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-950/60 p-4 text-sm text-slate-300">
              <div className="flex items-center gap-2 font-semibold text-white"><Info size={16} className="text-sky-400" />Source FPS vs processing FPS</div>
              <p className="mt-2">Source FPS describes the uploaded video playback rate. Processing FPS reflects how quickly LaneVision AI analyzes frames computationally.</p>
            </div>
          </div>
        </div>

        {isProcessing ? (
          <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/80 p-8 text-center">
            <h3 className="text-xl font-semibold text-white">Processing video...</h3>
            <p className="mt-2 text-sm text-slate-400">LaneVision AI is analyzing the video frame by frame. This may take longer than the source video's playback duration on smaller hardware.</p>
          </div>
        ) : null}

        {result ? (
          <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/80 p-6">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <h2 className="text-2xl font-semibold text-white">Video Analysis Result</h2>
                <p className="mt-1 text-sm text-slate-400">The processed video and core metrics are shown below.</p>
              </div>
              <div className="flex gap-3">
                <button type="button" onClick={resetState} className="rounded-full border border-slate-700 bg-slate-800 px-4 py-2 text-sm font-medium text-slate-100 transition hover:border-sky-500 hover:text-sky-400">Analyze Another Video</button>
                <a href={`${apiBaseUrl}${result.result.url}`} download="lanevision-video-result.mp4" className="inline-flex items-center gap-2 rounded-full bg-sky-500 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-sky-400">
                  <Download size={16} />
                  Download Processed Video
                </a>
              </div>
            </div>

            <div className="mt-8 grid gap-6 lg:grid-cols-2">
              <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                <h3 className="text-lg font-semibold text-white">Original Video</h3>
                <div className="mt-4 overflow-hidden rounded-xl border border-slate-800">
                  {previewUrl ? <video src={previewUrl} controls className="h-80 w-full object-contain" /> : null}
                </div>
              </div>
              <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                <h3 className="text-lg font-semibold text-white">Lane Detection Result</h3>
                <div className="mt-4 overflow-hidden rounded-xl border border-slate-800">
                  <video src={`${apiBaseUrl}${result.result.url}`} controls className="h-80 w-full object-contain" />
                </div>
              </div>
            </div>

            <div className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4"><p className="text-sm text-slate-400">Frames Processed</p><p className="mt-2 text-2xl font-semibold text-white">{result.metrics.frames_processed}</p></div>
              <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4"><p className="text-sm text-slate-400">Source FPS</p><p className="mt-2 text-2xl font-semibold text-white">{result.metrics.source_fps?.toFixed(1) ?? 'N/A'}</p></div>
              <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4"><p className="text-sm text-slate-400">Processing FPS</p><p className="mt-2 text-2xl font-semibold text-white">{result.metrics.average_processing_fps?.toFixed(1) ?? 'N/A'}</p></div>
              <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4"><p className="text-sm text-slate-400">Average Confidence</p><p className="mt-2 text-2xl font-semibold text-white">{result.metrics.average_confidence != null ? `${(result.metrics.average_confidence * 100).toFixed(1)}%` : 'N/A'}</p></div>
              <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4"><p className="text-sm text-slate-400">Detection Success Rate</p><p className="mt-2 text-2xl font-semibold text-white">{result.metrics.detection_success_rate != null ? `${(result.metrics.detection_success_rate * 100).toFixed(1)}%` : 'N/A'}</p></div>
              <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4"><p className="text-sm text-slate-400">Average Vehicle Offset</p><p className="mt-2 text-2xl font-semibold text-white">{result.metrics.average_vehicle_offset_pixels != null ? `${result.metrics.average_vehicle_offset_pixels.toFixed(1)} px` : 'N/A'}</p></div>
            </div>

            <div className="mt-8 flex flex-wrap gap-4">
              <div className="rounded-2xl border border-slate-800 bg-slate-950/60 px-4 py-3 text-sm text-slate-300"><span className="font-semibold text-white">Both Lanes</span>: {result.metrics.both_lane_detection_rate != null ? `${(result.metrics.both_lane_detection_rate * 100).toFixed(1)}%` : 'N/A'}</div>
              <div className="rounded-2xl border border-slate-800 bg-slate-950/60 px-4 py-3 text-sm text-slate-300"><span className="font-semibold text-white">Avg Processing Time</span>: {result.metrics.average_processing_time_ms != null ? `${result.metrics.average_processing_time_ms.toFixed(1)} ms` : 'N/A'}</div>
              {result.warnings?.length ? <div className="rounded-2xl border border-amber-700/40 bg-amber-900/20 p-4 text-sm text-amber-200"><div className="flex items-center gap-2 font-semibold"><AlertTriangle size={16} />Warnings</div><ul className="mt-2 list-disc space-y-1 pl-5">{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></div> : null}
            </div>
          </div>
        ) : null}
      </main>
    </div>
  );
}

export default Video;
