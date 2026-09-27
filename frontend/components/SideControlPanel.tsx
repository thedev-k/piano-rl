import React, { useRef } from 'react';
import { MidiFileItem, ConnectionStatus } from '../hooks/usePianoStream';

interface SideControlPanelProps {
  connectionStatus: ConnectionStatus;
  files: MidiFileItem[];
  selectedFile: string;
  isPlaying: boolean;
  tempo: number;
  playbackSpeed: number;
  audioUnlocked: boolean;
  onSelectFile: (path: string) => void;
  onPlay: (path?: string) => void;
  onStop: () => void;
  onChangeSpeed: (speed: number) => void;
  onUpload: (file: File) => Promise<any>;
  onUnlockAudio: () => void;
}

export const SideControlPanel: React.FC<SideControlPanelProps> = ({
  connectionStatus,
  files,
  selectedFile,
  isPlaying,
  tempo,
  playbackSpeed,
  audioUnlocked,
  onSelectFile,
  onPlay,
  onStop,
  onChangeSpeed,
  onUpload,
  onUnlockAudio,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const selectedFileName = selectedFile
    ? selectedFile.split(/[/\\]/).pop()?.replace(/\.midi?$/i, '').replace(/_/g, ' ')
    : 'No piece selected';

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      try {
        const res = await onUpload(file);
        if (res.file?.path) {
          onSelectFile(res.file.path);
          onPlay(res.file.path);
        }
      } catch (err) {
        console.error('Upload error:', err);
      } finally {
        e.target.value = '';
      }
    }
  };

  return (
    <aside style={panelContainerStyle} aria-label="Recital Program Controls">
      {/* Hidden file input for upload */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".mid,.midi"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      {/* Program Card Header */}
      <div style={cardHeaderStyle}>
        <div style={headerTitleRow}>
          <span style={programEyebrowStyle}>Recital Program</span>
          <span
            title={`Backend: ${connectionStatus}`}
            style={{
              ...statusDotStyle,
              backgroundColor:
                connectionStatus === 'connected'
                  ? '#4e9f76'
                  : connectionStatus === 'reconnecting'
                  ? '#d9a544'
                  : '#c25442',
            }}
          />
        </div>
        <div style={goldDividerStyle} />
      </div>

      {/* 1. File Chooser & Upload */}
      <div style={sectionGroupStyle}>
        <label style={fieldLabelStyle} htmlFor="piece-selector">
          Select score
        </label>
        <select
          id="piece-selector"
          value={selectedFile}
          onChange={(e) => {
            onSelectFile(e.target.value);
            onPlay(e.target.value);
          }}
          style={selectInputStyle}
        >
          {files.map((f) => (
            <option key={f.path} value={f.path}>
              [{f.folder}] {f.name.replace(/\.midi?$/i, '')}
            </option>
          ))}
        </select>

        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          style={uploadButtonStyle}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <span>Upload custom MIDI</span>
        </button>
      </div>

      {/* 2. Track Name (Now playing) with elegant serif typography */}
      <div style={nowPlayingBoxStyle}>
        <span style={nowPlayingLabelStyle}>Now playing</span>
        <h2 style={trackTitleSerifStyle} title={selectedFileName}>
          {selectedFileName}
        </h2>
        <div style={tempoRowStyle}>
          <span style={tempoTextStyle}>Tempo: {Math.round(tempo)} BPM</span>
        </div>
      </div>

      {/* 3. Play / Pause Control */}
      <div style={sectionGroupStyle}>
        {!isPlaying ? (
          <button
            type="button"
            onClick={() => onPlay()}
            style={playPrimaryButtonStyle}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
            <span>Play recital</span>
          </button>
        ) : (
          <button
            type="button"
            onClick={onStop}
            style={stopPrimaryButtonStyle}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M6 6h12v12H6z" />
            </svg>
            <span>Pause recital</span>
          </button>
        )}
      </div>

      {/* 4. Playback Speed Control */}
      <div style={sectionGroupStyle}>
        <span style={fieldLabelStyle}>Playback speed</span>
        <div style={speedButtonGroupStyle}>
          {[0.5, 0.75, 1.0, 1.25, 1.5].map((spd) => (
            <button
              key={spd}
              type="button"
              onClick={() => onChangeSpeed(spd)}
              style={{
                ...speedButtonStyle,
                backgroundColor: playbackSpeed === spd ? '#d9a544' : 'rgba(255, 255, 255, 0.06)',
                color: playbackSpeed === spd ? '#141214' : '#d6d3d0',
                fontWeight: playbackSpeed === spd ? 700 : 500,
              }}
            >
              {spd}×
            </button>
          ))}
        </div>
      </div>

      {/* Audio Engine Enable toggle if needed */}
      {!audioUnlocked && (
        <button
          type="button"
          onClick={onUnlockAudio}
          style={audioUnlockCalloutStyle}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z" />
          </svg>
          <span>Enable piano audio</span>
        </button>
      )}
    </aside>
  );
};

/* Styles: Program Card Stock on Concert Stage */
const panelContainerStyle: React.CSSProperties = {
  position: 'absolute',
  right: '36px',
  top: '50%',
  transform: 'translateY(-50%)',
  width: '290px',
  background: 'rgba(22, 18, 20, 0.94)', // Deep warm charcoal-brown
  backdropFilter: 'blur(16px)',
  WebkitBackdropFilter: 'blur(16px)',
  border: '1px solid rgba(217, 165, 68, 0.42)', // Gold hairline edge
  borderRadius: '8px',
  padding: '24px 22px',
  boxShadow: '0 20px 48px rgba(0, 0, 0, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.08)',
  display: 'flex',
  flexDirection: 'column',
  gap: '20px',
  zIndex: 100,
  color: '#e8e5e1',
  pointerEvents: 'auto',
};

const cardHeaderStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '8px',
};

const headerTitleRow: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
};

const programEyebrowStyle: React.CSSProperties = {
  fontSize: '13px',
  fontWeight: 600,
  color: '#d9a544', // Warm stage gold
  letterSpacing: '0.04em',
  fontFamily: 'serif',
};

const statusDotStyle: React.CSSProperties = {
  width: '7px',
  height: '7px',
  borderRadius: '50%',
  display: 'inline-block',
};

const goldDividerStyle: React.CSSProperties = {
  width: '100%',
  height: '1px',
  background: 'linear-gradient(90deg, rgba(217, 165, 68, 0.6) 0%, rgba(217, 165, 68, 0.1) 100%)',
};

const sectionGroupStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '7px',
};

const fieldLabelStyle: React.CSSProperties = {
  fontSize: '11px',
  color: '#a39c94',
  fontWeight: 500,
};

const selectInputStyle: React.CSSProperties = {
  background: '#120f12',
  border: '1px solid rgba(217, 165, 68, 0.25)',
  color: '#f0ece6',
  borderRadius: '4px',
  padding: '8px 10px',
  fontSize: '12px',
  outline: 'none',
  width: '100%',
};

const uploadButtonStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '7px',
  background: 'rgba(255, 255, 255, 0.05)',
  border: '1px dashed rgba(217, 165, 68, 0.35)',
  color: '#d9a544',
  borderRadius: '4px',
  padding: '7px 12px',
  fontSize: '11px',
  fontWeight: 500,
  cursor: 'pointer',
};

const nowPlayingBoxStyle: React.CSSProperties = {
  background: 'rgba(10, 8, 10, 0.65)',
  borderLeft: '2px solid #d9a544',
  padding: '10px 14px',
  display: 'flex',
  flexDirection: 'column',
  gap: '4px',
};

const nowPlayingLabelStyle: React.CSSProperties = {
  fontSize: '11px',
  color: '#9e968c',
  fontStyle: 'italic',
};

const trackTitleSerifStyle: React.CSSProperties = {
  fontSize: '17px',
  fontWeight: 500,
  fontFamily: 'Georgia, "Playfair Display", "Times New Roman", serif',
  color: '#fcfaf7',
  lineHeight: 1.3,
  textTransform: 'capitalize',
  margin: 0,
};

const tempoRowStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  fontSize: '11px',
  color: '#b0a89d',
  fontFamily: 'JetBrains Mono, monospace',
  marginTop: '2px',
};

const tempoTextStyle: React.CSSProperties = {
  color: '#d9a544',
};

const playPrimaryButtonStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '8px',
  background: '#d9a544',
  color: '#141113',
  fontWeight: 600,
  fontSize: '13px',
  padding: '10px 16px',
  borderRadius: '4px',
  cursor: 'pointer',
  border: 'none',
  boxShadow: '0 4px 14px rgba(217, 165, 68, 0.3)',
};

const stopPrimaryButtonStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '8px',
  background: '#8c2424',
  color: '#ffffff',
  fontWeight: 600,
  fontSize: '13px',
  padding: '10px 16px',
  borderRadius: '4px',
  cursor: 'pointer',
  border: 'none',
};

const speedButtonGroupStyle: React.CSSProperties = {
  display: 'flex',
  gap: '4px',
  width: '100%',
};

const speedButtonStyle: React.CSSProperties = {
  flex: 1,
  padding: '6px 0',
  borderRadius: '3px',
  fontSize: '11px',
  cursor: 'pointer',
  border: '1px solid rgba(255, 255, 255, 0.08)',
  textAlign: 'center',
};

const audioUnlockCalloutStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '8px',
  padding: '8px',
  background: 'rgba(217, 165, 68, 0.12)',
  border: '1px solid rgba(217, 165, 68, 0.3)',
  color: '#d9a544',
  borderRadius: '4px',
  fontSize: '11px',
  cursor: 'pointer',
};
