import React, { useState, useRef } from 'react';
import { MidiFileItem, PieceInfo, ConnectionStatus } from '../hooks/usePianoStream';

interface ControlsProps {
  connectionStatus: ConnectionStatus;
  files: MidiFileItem[];
  selectedFile: string;
  pieceInfo: PieceInfo | null;
  isPlaying: boolean;
  melodyOnly: boolean;
  playbackSpeed: number;
  onSelectFile: (path: string) => void;
  onToggleMelody: (val: boolean) => void;
  onPlay: () => void;
  onStop: () => void;
  onChangeSpeed: (speed: number) => void;
  onUpload: (file: File) => Promise<any>;
  onReconnect: () => void;
}

export const Controls: React.FC<ControlsProps> = ({
  connectionStatus,
  files,
  selectedFile,
  pieceInfo,
  isPlaying,
  melodyOnly,
  playbackSpeed,
  onSelectFile,
  onToggleMelody,
  onPlay,
  onStop,
  onChangeSpeed,
  onUpload,
  onReconnect,
}) => {
  const [uploadState, setUploadState] = useState<'idle' | 'dragover' | 'uploading' | 'error' | 'success'>('idle');
  const [uploadError, setUploadError] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (uploadState !== 'uploading') {
      setUploadState('dragover');
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    if (uploadState === 'dragover') {
      setUploadState('idle');
    }
  };

  const processFile = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.mid') && !file.name.toLowerCase().endsWith('.midi')) {
      setUploadState('error');
      setUploadError('Only .mid or .midi score files are accepted.');
      setTimeout(() => setUploadState('idle'), 3500);
      return;
    }

    setUploadState('uploading');
    setUploadError('');
    try {
      await onUpload(file);
      setUploadState('success');
      setTimeout(() => setUploadState('idle'), 2500);
    } catch (err: any) {
      setUploadState('error');
      setUploadError(err.message || 'Upload failed');
      setTimeout(() => setUploadState('idle'), 4000);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      await processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      await processFile(e.target.files[0]);
      e.target.value = '';
    }
  };

  return (
    <div style={panelContainerStyle}>
      {/* Upper Control Strip: File Select, Play/Stop, Speed */}
      <div style={controlStripStyle}>
        {/* Piece Selection */}
        <div style={groupStyle}>
          <label style={labelStyle} htmlFor="piece-select">Repertoire score</label>
          <select
            id="piece-select"
            value={selectedFile}
            onChange={(e) => onSelectFile(e.target.value)}
            disabled={isPlaying}
            style={selectStyle}
          >
            {files.map((f) => (
              <option key={f.path} value={f.path}>
                [{f.folder}] {f.name}
              </option>
            ))}
          </select>
        </div>

        {/* Melody Only Toggle */}
        <div style={checkboxGroupStyle}>
          <label style={checkboxLabelStyle}>
            <input
              type="checkbox"
              checked={melodyOnly}
              onChange={(e) => onToggleMelody(e.target.checked)}
              disabled={isPlaying}
              style={checkboxInputStyle}
            />
            <span>Extract melody only</span>
          </label>
          {pieceInfo?.dropped_notes ? (
            <span style={hintBadgeStyle}>
              {melodyOnly ? `Active (${pieceInfo.dropped_notes} poly notes pruned)` : `${pieceInfo.dropped_notes} poly notes`}
            </span>
          ) : null}
        </div>

        {/* Playback Speed */}
        <div style={groupStyle}>
          <label style={labelStyle} htmlFor="speed-select">Playback tempo speed</label>
          <div style={speedButtonGroup}>
            {[0.5, 0.75, 1.0, 1.25, 1.5].map((spd) => (
              <button
                key={spd}
                type="button"
                onClick={() => onChangeSpeed(spd)}
                style={{
                  ...speedButtonStyle,
                  backgroundColor: playbackSpeed === spd ? 'var(--accent-brass)' : 'var(--surface-bay)',
                  color: playbackSpeed === spd ? '#121316' : 'var(--text-primary)',
                  fontWeight: playbackSpeed === spd ? 700 : 500,
                }}
              >
                {spd}×
              </button>
            ))}
          </div>
        </div>

        {/* Play / Stop Action */}
        <div style={actionButtonGroup}>
          {!isPlaying ? (
            <button
              type="button"
              onClick={onPlay}
              disabled={connectionStatus !== 'connected' || !selectedFile}
              style={playButtonStyle}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z" />
              </svg>
              <span>Play RL Agent</span>
            </button>
          ) : (
            <button type="button" onClick={onStop} style={stopButtonStyle}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M6 6h12v12H6z" />
              </svg>
              <span>Stop Stream</span>
            </button>
          )}
        </div>
      </div>

      {/* Piece Info & Drag-Drop Uploader Strip */}
      <div style={secondaryStripStyle}>
        {/* Piece Telemetry Summary */}
        <div style={pieceSummaryBayStyle}>
          <div style={summaryHeader}>
            <span style={summaryTitle}>Active Score Analysis</span>
            {pieceInfo?.name && <span style={summaryFileName}>{pieceInfo.name}</span>}
          </div>
          <div style={summaryDetailsGrid}>
            <div>
              <span style={dimLabel}>Notes:</span>{' '}
              <strong className="tabular-num">
                {melodyOnly ? pieceInfo?.melody_info?.total_notes ?? '—' : pieceInfo?.info?.total_notes ?? '—'}
              </strong>
            </div>
            <div>
              <span style={dimLabel}>Tempo:</span>{' '}
              <strong className="tabular-num">{pieceInfo?.info?.tempo_bpm ?? '—'} BPM</strong>
            </div>
            <div>
              <span style={dimLabel}>Beats:</span>{' '}
              <strong className="tabular-num">
                {pieceInfo?.info?.total_beats ? pieceInfo.info.total_beats.toFixed(1) : '—'}
              </strong>
            </div>
          </div>
        </div>

        {/* Drag & Drop Upload Bay */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => uploadState !== 'uploading' && fileInputRef.current?.click()}
          style={{
            ...uploadDropzoneStyle,
            borderColor:
              uploadState === 'dragover'
                ? 'var(--accent-brass)'
                : uploadState === 'error'
                ? 'var(--status-strike-wrong)'
                : uploadState === 'success'
                ? 'var(--status-strike-hit)'
                : 'var(--border-subtle)',
            backgroundColor:
              uploadState === 'dragover'
                ? 'rgba(200, 155, 83, 0.08)'
                : 'var(--surface-bay)',
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".mid,.midi"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          <div style={dropzoneContentStyle}>
            {uploadState === 'uploading' ? (
              <span style={uploadStatusText}>Uploading score file...</span>
            ) : uploadState === 'error' ? (
              <span style={{ ...uploadStatusText, color: 'var(--status-strike-wrong)' }}>
                {uploadError || 'Upload failed'}
              </span>
            ) : uploadState === 'success' ? (
              <span style={{ ...uploadStatusText, color: 'var(--status-strike-hit)' }}>
                MIDI parsed successfully
              </span>
            ) : uploadState === 'dragover' ? (
              <span style={{ ...uploadStatusText, color: 'var(--accent-brass)' }}>
                Release to upload MIDI
              </span>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                <span>Drop MIDI file to upload or browse</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Connection State Warning Bar if disconnected */}
      {connectionStatus !== 'connected' && (
        <div style={connectionBannerStyle}>
          <span>
            {connectionStatus === 'reconnecting'
              ? 'Attempting to reconnect WebSocket to http://localhost:8000 via proxy...'
              : 'Backend connection lost. Ensure the Python server is running on port 8000.'}
          </span>
          <button type="button" onClick={onReconnect} style={reconnectButtonStyle}>
            Retry Connection
          </button>
        </div>
      )}
    </div>
  );
};

/* Styles */
const panelContainerStyle: React.CSSProperties = {
  width: '100%',
  maxWidth: '1100px',
  margin: '0 auto',
  display: 'flex',
  flexDirection: 'column',
  gap: '12px',
};

const controlStripStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  flexWrap: 'wrap',
  gap: '14px',
  background: 'var(--surface-chassis)',
  border: '1px solid var(--border-subtle)',
  borderRadius: '6px',
  padding: '12px 16px',
};

const groupStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '6px',
};

const labelStyle: React.CSSProperties = {
  fontSize: '11px',
  fontWeight: 600,
  color: 'var(--text-muted)',
};

const selectStyle: React.CSSProperties = {
  background: 'var(--surface-bay)',
  border: '1px solid var(--border-subtle)',
  color: 'var(--text-primary)',
  borderRadius: '4px',
  padding: '6px 10px',
  fontSize: '13px',
  minWidth: '220px',
  outline: 'none',
};

const checkboxGroupStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '6px',
};

const checkboxLabelStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '8px',
  fontSize: '13px',
  cursor: 'pointer',
  userSelect: 'none',
  color: 'var(--text-primary)',
};

const checkboxInputStyle: React.CSSProperties = {
  accentColor: 'var(--accent-brass)',
  cursor: 'pointer',
};

const hintBadgeStyle: React.CSSProperties = {
  fontSize: '10px',
  color: 'var(--text-muted)',
  fontFamily: 'JetBrains Mono, monospace',
};

const speedButtonGroup: React.CSSProperties = {
  display: 'flex',
  gap: '4px',
};

const speedButtonStyle: React.CSSProperties = {
  border: '1px solid var(--border-subtle)',
  borderRadius: '4px',
  padding: '5px 8px',
  fontSize: '11px',
};

const actionButtonGroup: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
};

const playButtonStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '8px',
  background: 'var(--status-strike-hit)',
  color: '#ffffff',
  fontWeight: 600,
  fontSize: '13px',
  padding: '8px 18px',
  borderRadius: '4px',
};

const stopButtonStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '8px',
  background: 'var(--status-strike-wrong)',
  color: '#ffffff',
  fontWeight: 600,
  fontSize: '13px',
  padding: '8px 18px',
  borderRadius: '4px',
};

const secondaryStripStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '1fr 1fr',
  gap: '12px',
};

const pieceSummaryBayStyle: React.CSSProperties = {
  background: 'var(--surface-chassis)',
  border: '1px solid var(--border-subtle)',
  borderRadius: '6px',
  padding: '12px 16px',
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'center',
  gap: '8px',
};

const summaryHeader: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
};

const summaryTitle: React.CSSProperties = {
  fontSize: '11px',
  fontWeight: 600,
  color: 'var(--text-muted)',
};

const summaryFileName: React.CSSProperties = {
  fontSize: '12px',
  fontFamily: 'JetBrains Mono, monospace',
  color: 'var(--accent-brass)',
};

const summaryDetailsGrid: React.CSSProperties = {
  display: 'flex',
  gap: '24px',
  fontSize: '12px',
};

const dimLabel: React.CSSProperties = {
  color: 'var(--text-dim)',
};

const uploadDropzoneStyle: React.CSSProperties = {
  border: '1px dashed var(--border-subtle)',
  borderRadius: '6px',
  padding: '12px 16px',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  transition: 'border-color 0.15s ease, background-color 0.15s ease',
};

const dropzoneContentStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  fontSize: '12px',
  color: 'var(--text-muted)',
};

const uploadStatusText: React.CSSProperties = {
  fontWeight: 500,
};

const connectionBannerStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: '10px 14px',
  background: 'rgba(194, 84, 66, 0.15)',
  border: '1px solid rgba(194, 84, 66, 0.35)',
  borderRadius: '5px',
  color: 'var(--status-strike-wrong)',
  fontSize: '12px',
};

const reconnectButtonStyle: React.CSSProperties = {
  padding: '4px 10px',
  background: 'var(--surface-bay)',
  border: '1px solid var(--border-subtle)',
  color: 'var(--text-primary)',
  borderRadius: '3px',
  fontSize: '11px',
  fontWeight: 600,
};
