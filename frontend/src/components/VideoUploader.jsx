import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Film, Upload, X } from 'lucide-react';

function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB'];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function VideoUploader({ file, onFileChange, onAnalyze, isProcessing, error, maxSizeMb = 100 }) {
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const fileInputRef = useRef(null);

  const acceptedExtensions = useMemo(() => '.mp4, .mov, .avi', []);
  const acceptedTypes = useMemo(() => ['video/mp4', 'video/quicktime', 'video/x-msvideo'], []);

  const revokePreview = useCallback(() => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
  }, [previewUrl]);

  useEffect(() => {
    return () => revokePreview();
  }, [revokePreview]);

  useEffect(() => {
    if (!file) {
      revokePreview();
      setMetadata(null);
      return;
    }

    const nextPreviewUrl = URL.createObjectURL(file);
    setPreviewUrl((currentUrl) => {
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      return nextPreviewUrl;
    });

    const video = document.createElement('video');
    video.preload = 'metadata';
    video.src = nextPreviewUrl;
    video.onloadedmetadata = () => {
      const duration = Number.isFinite(video.duration) ? video.duration : null;
      setMetadata({ duration });
    };

    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [file]);

  const handleFiles = useCallback((selectedFiles) => {
    const selected = selectedFiles?.[0];
    if (!selected) return;

    const extension = selected.name.split('.').pop()?.toLowerCase();
    const extensionOk = ['mp4', 'mov', 'avi'].includes(extension || '');
    if (!extensionOk) {
      onFileChange({ error: 'Unsupported video format. Use MP4, MOV, or AVI.' });
      return;
    }

    if (!acceptedTypes.includes(selected.type) && selected.type) {
      onFileChange({ error: 'Unsupported video MIME type.' });
      return;
    }

    if (selected.size > maxSizeMb * 1024 * 1024) {
      onFileChange({ error: `File exceeds the ${maxSizeMb}MB limit.` });
      return;
    }

    onFileChange({ file: selected, error: null });
  }, [acceptedTypes, maxSizeMb, onFileChange]);

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    handleFiles(event.dataTransfer.files);
  };

  const handleSelectFile = () => fileInputRef.current?.click();

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6">
      <div
        onDrop={handleDrop}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        className={`rounded-2xl border-2 border-dashed p-6 text-center transition ${isDragging ? 'border-sky-500 bg-slate-800/80' : 'border-slate-700 bg-slate-950/50'}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={acceptedExtensions}
          className="hidden"
          onChange={(event) => handleFiles(event.target.files)}
        />
        <div className="flex flex-col items-center justify-center gap-3">
          <div className="rounded-full bg-slate-800 p-3 text-sky-400">
            <Upload size={20} />
          </div>
          <div>
            <p className="text-lg font-semibold text-white">Drag & drop a road video</p>
            <p className="mt-1 text-sm text-slate-400">MP4, MOV, or AVI up to {maxSizeMb}MB</p>
          </div>
          <button type="button" onClick={handleSelectFile} className="rounded-full border border-slate-700 bg-slate-800 px-4 py-2 text-sm font-medium text-slate-100 transition hover:border-sky-500 hover:text-sky-400">
            Browse files
          </button>
        </div>
      </div>

      {error ? <p className="mt-4 text-sm text-rose-400">{error}</p> : null}

      {file ? (
        <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="font-semibold text-white">Selected Video</p>
              <p className="mt-1 text-sm text-slate-400">{file.name}</p>
              <p className="text-sm text-slate-500">{formatBytes(file.size)}</p>
              {metadata?.duration ? <p className="text-sm text-slate-500">Duration: {metadata.duration.toFixed(2)}s</p> : null}
            </div>
            <button type="button" onClick={() => onFileChange({ file: null, error: null })} className="rounded-full border border-slate-700 p-2 text-slate-300 transition hover:border-rose-500 hover:text-rose-400" aria-label="Remove selected video">
              <X size={16} />
            </button>
          </div>

          {previewUrl ? (
            <div className="mt-4 overflow-hidden rounded-xl border border-slate-800">
              <video src={previewUrl} controls className="h-64 w-full object-cover" />
            </div>
          ) : null}

          <button type="button" onClick={onAnalyze} disabled={isProcessing} className="mt-4 inline-flex items-center gap-2 rounded-full bg-sky-500 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-60">
            <Film size={16} />
            {isProcessing ? 'Processing...' : 'Analyze Video'}
          </button>
        </div>
      ) : null}
    </div>
  );
}

export default VideoUploader;
