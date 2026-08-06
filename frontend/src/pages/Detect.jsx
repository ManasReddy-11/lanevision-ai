import { useMemo, useState } from 'react';
import { AlertTriangle, Download, Info, ScanLine } from 'lucide-react';
import ImageUploader from '../components/ImageUploader';
import { detectImage } from '../services/api';

function Detect() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [algorithm, setAlgorithm] = useState('classic');

  const handleFileChange = ({ file, error: nextError }) => {
    setSelectedFile(file);
    setError(nextError || null);
    if (!file) {
      setResult(null);
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile) {
      setError('Please select an image first.');
      return;
    }

    setError(null);
    setIsProcessing(true);
    try {
      const response = await detectImage(selectedFile, algorithm);
      setResult(response.data);
    } catch (err) {
      const detail = err?.response?.data?.detail || err?.message || 'Unable to process the image.';
      setError(detail);
    } finally {
      setIsProcessing(false);
    }
  };

  const confidencePercent = useMemo(() => {
    if (!result?.metrics?.confidence) return null;
    return `${(result.metrics.confidence * 100).toFixed(1)}%`;
  }, [result]);

  const aiMetadataItems = useMemo(() => {
    if (algorithm !== 'ai' || !result?.ai_metadata) {
      return [];
    }

    const entries = [
      ['Model', result.ai_metadata.model_name],
      ['Architecture', result.ai_metadata.architecture],
      ['Device', result.ai_metadata.device],
      ['Checkpoint loaded', result.ai_metadata.checkpoint_loaded != null ? (result.ai_metadata.checkpoint_loaded ? 'Yes' : 'No') : null],
      ['Input size', Array.isArray(result.ai_metadata.input_size) ? `${result.ai_metadata.input_size[0]} × ${result.ai_metadata.input_size[1]}` : null],
      ['Inference time', result.ai_metadata.inference_time_ms != null ? `${result.ai_metadata.inference_time_ms.toFixed(1)} ms` : null],
      ['Confidence', result.ai_metadata.confidence != null ? `${(result.ai_metadata.confidence * 100).toFixed(1)}%` : null],
    ];

    return entries.filter(([, value]) => value != null && value !== '');
  }, [algorithm, result]);

  const resetState = () => {
    setSelectedFile(null);
    setError(null);
    setResult(null);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <main className="mx-auto max-w-7xl px-6 py-10 lg:px-8">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-8 shadow-2xl shadow-black/30">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <div className="flex items-center gap-2 text-sm font-medium uppercase tracking-[0.3em] text-sky-400">
                <ScanLine size={16} />
                Lane Detection Workspace
              </div>
              <h1 className="mt-3 text-3xl font-semibold text-white">Upload a road image and analyze lane boundaries</h1>
              <p className="mt-3 max-w-2xl text-slate-400">
                Compare classic OpenCV lane detection with the new AI-assisted segmentation path and inspect lane center and vehicle offset from a single image.
              </p>
            </div>
            <div className="rounded-full border border-slate-700 bg-slate-800 px-4 py-2 text-sm text-slate-300">
              {algorithm === 'advanced' ? 'Advanced algorithm • curved-lane fitting' : algorithm === 'ai' ? 'AI segmentation • neural mask inference' : 'Classic algorithm • local processing'}
            </div>
          </div>
        </div>

        <div className="mt-8 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <ImageUploader
            file={selectedFile}
            onFileChange={handleFileChange}
            onAnalyze={analyzeImage}
            isProcessing={isProcessing}
            error={error}
          />

          <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-slate-300">
                Detection Algorithm
                <select value={algorithm} onChange={(event) => setAlgorithm(event.target.value)} className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white">
                  <option value="classic">Classic OpenCV</option>
                  <option value="advanced">Advanced Lane Detection</option>
                  <option value="ai">AI Segmentation</option>
                </select>
              </label>
              <div>
                <h2 className="text-xl font-semibold text-white">Processing Pipeline</h2>
                <p className="mt-1 text-sm text-slate-400">Educational summary of the selected algorithm.</p>
              </div>
              <div className="rounded-full border border-slate-700 bg-slate-800 px-3 py-1 text-xs uppercase tracking-[0.25em] text-slate-300">
                {algorithm === 'ai' ? 'Neural AI' : algorithm === 'advanced' ? 'Curved-Lane CV' : 'Classical CV'}
              </div>
            </div>
            <ol className="mt-6 space-y-3 text-sm text-slate-300">
              {(algorithm === 'ai'
                ? ["Semantic Segmentation", "Mask Postprocessing", "Lane Extraction", "Lane Overlay", "Confidence Estimation"]
                : ["White/Yellow Color Filtering", "Gaussian Blur", "Canny Edge Detection", "Region of Interest", "Hough Line Transform", "Lane Classification", "Weighted Lane Fitting", "Lane Overlay"]
              ).map((step) => (
                <li key={step} className="flex items-center gap-3 rounded-xl border border-slate-800 bg-slate-950/60 px-3 py-2">
                  <span className="flex h-6 w-6 items-center justify-center rounded-full bg-sky-500/15 text-xs font-semibold text-sky-400">{step[0]}</span>
                  <span>{step}</span>
                </li>
              ))}
            </ol>
          </div>
        </div>

        {isProcessing ? (
          <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/80 p-8 text-center">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full border border-sky-500/30 bg-sky-500/10 text-sky-400">
              <ScanLine size={20} />
            </div>
            <h3 className="mt-4 text-xl font-semibold text-white">Analyzing road geometry...</h3>
            <p className="mt-2 text-sm text-slate-400">The backend is validating the image and running the selected lane detector pipeline.</p>
          </div>
        ) : null}

        {result ? (
          <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/80 p-6">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <h2 className="text-2xl font-semibold text-white">Lane Detection Result</h2>
                <p className="mt-1 text-sm text-slate-400">The processed road image and key metrics are shown below.</p>
              </div>
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={resetState}
                  className="rounded-full border border-slate-700 bg-slate-800 px-4 py-2 text-sm font-medium text-slate-100 transition hover:border-sky-500 hover:text-sky-400"
                >
                  Analyze Another Image
                </button>
                <a
                  href={`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}${result.result.url}`}
                  download="lanevision-result.jpg"
                  className="inline-flex items-center gap-2 rounded-full bg-sky-500 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-sky-400"
                >
                  <Download size={16} />
                  Download Result
                </a>
              </div>
            </div>

            <div className="mt-8 grid gap-6 lg:grid-cols-2">
              <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                <h3 className="text-lg font-semibold text-white">Original</h3>
                <div className="mt-4 overflow-hidden rounded-xl border border-slate-800">
                  <img src={`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}${result.original.url}`} alt="Original uploaded road" className="h-80 w-full object-contain" />
                </div>
              </div>
              <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                <h3 className="text-lg font-semibold text-white">Lane Detection</h3>
                <div className="mt-4 overflow-hidden rounded-xl border border-slate-800">
                  <img src={`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}${result.result.url}`} alt="Processed lane detection" className="h-80 w-full object-contain" />
                </div>
              </div>
            </div>

            <div className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-5">
              <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
                <p className="text-sm text-slate-400">Detection Confidence</p>
                <p className="mt-2 text-2xl font-semibold text-white">{confidencePercent ?? 'N/A'}</p>
              </div>
              <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
                <p className="text-sm text-slate-400">Processing Time</p>
                <p className="mt-2 text-2xl font-semibold text-white">{result.metrics.processing_time_ms != null ? `${result.metrics.processing_time_ms.toFixed(1)} ms` : 'N/A'}</p>
              </div>
              <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
                <p className="text-sm text-slate-400">Segments Detected</p>
                <p className="mt-2 text-2xl font-semibold text-white">{result.metrics.detected_segments ?? 'N/A'}</p>
              </div>
              <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
                <p className="text-sm text-slate-400">Vehicle Offset</p>
                <p className="mt-2 text-2xl font-semibold text-white">{result.metrics.vehicle_offset_pixels != null ? `${result.metrics.vehicle_offset_pixels.toFixed(1)} px` : 'N/A'}</p>
              </div>
              <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
                <p className="text-sm text-slate-400">Lane Center</p>
                <p className="mt-2 text-2xl font-semibold text-white">{result.metrics.lane_center_x != null ? result.metrics.lane_center_x.toFixed(1) : 'N/A'}</p>
              </div>
            </div>

            <div className="mt-8 flex flex-wrap gap-4">
              <div className="rounded-2xl border border-slate-800 bg-slate-950/60 px-4 py-3 text-sm text-slate-300">
                <span className="font-semibold text-white">Left Lane</span>: {result.lanes.left_detected ? 'Detected' : 'Not Detected'}
              </div>
              <div className="rounded-2xl border border-slate-800 bg-slate-950/60 px-4 py-3 text-sm text-slate-300">
                <span className="font-semibold text-white">Right Lane</span>: {result.lanes.right_detected ? 'Detected' : 'Not Detected'}
              </div>
              <div className="flex items-center gap-2 rounded-2xl border border-slate-800 bg-slate-950/60 px-4 py-3 text-sm text-slate-300">
                <Info size={16} className="text-sky-400" />
                {algorithm === 'ai' ? 'AI results are based on segmentation confidence and may fall back to a graceful unavailable-state if no checkpoint is configured.' : 'Confidence is a heuristic score derived from lane geometry and segment consistency.'}
              </div>
            </div>

            {algorithm === 'ai' ? (
              <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
                <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.25em] text-sky-400">
                  <ScanLine size={16} />
                  AI Model / Neural Inference
                </div>
                {aiMetadataItems.length ? (
                  <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                    {aiMetadataItems.map(([label, value]) => (
                      <div key={label} className="rounded-2xl border border-slate-800 bg-slate-900/70 p-3">
                        <p className="text-sm text-slate-400">{label}</p>
                        <p className="mt-1 font-semibold text-white">{value}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="mt-4 rounded-2xl border border-amber-700/40 bg-amber-900/20 p-4 text-sm text-amber-200">
                    AI model checkpoint is not configured. Neural inference is unavailable.
                  </div>
                )}
              </div>
            ) : null}

            {result.warnings?.length ? (
              <div className="mt-6 rounded-2xl border border-amber-700/40 bg-amber-900/20 p-4 text-sm text-amber-200">
                <div className="flex items-center gap-2 font-semibold">
                  <AlertTriangle size={16} />
                  Warnings
                </div>
                <ul className="mt-2 list-disc space-y-1 pl-5">
                  {result.warnings.map((warning) => <li key={warning}>{warning}</li>)}
                </ul>
              </div>
            ) : null}
          </div>
        ) : null}
      </main>
    </div>
  );
}

export default Detect;
