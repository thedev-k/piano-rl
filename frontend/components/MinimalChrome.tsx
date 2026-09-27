import React, { useRef, useState } from 'react';
import { MidiFileItem, ConnectionStatus } from '../hooks/usePianoStream';

interface MinimalChromeProps {
  connectionStatus: ConnectionStatus;
  files: MidiFileItem[];
  selectedFile: string;
  isPlaying: boolean;
  tempo: number;
  audioUnlocked: boolean;
  onSelectFile: (path: string) => void;
  onPlay: (path?: string) => void;
  onStop: () => void;
  onUpload: (file: File) => Promise<any>;
  onUnlockAudio: () => void;
}

export const MinimalChrome: React.FC<MinimalChromeProps> = ({
  connectionStatus,
  files,
  selectedFile,
  isPlaying,
  tempo,
  audioUnlocked,
  onSelectFile,
  onPlay,
  onStop,
  onUpload,
  onUnlockAudio,
}) => {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const selectedFileName = selectedFile
    ? selectedFile.split(/[/\\]/).pop()?.replace(/\.midi?$/i, '')
    : null;

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setIsUploading(true);
      try {
        const res = await onUpload(file);
        if (res.file?.path) {
          onSelectFile(res.file.path);
          onPlay(res.file.path);
        }
      } catch (err) {
        console.error('Upload error:', err);
      } finally {
        setIsUploading(false);
        e.target.value = '';
      }
    }
  };

  const handleSelectPreloaded = (path: string) => {
    onSelectFile(path);
    setDropdownOpen(false);
    onPlay(path);
  };

  return (
    <div style={chromeOverlayStyle}>
      {/* Top-Center Minimal Pill Control matching reference sketch */}
      <div style={pillContainerStyle}>
        <input
          ref={fileInputRef}
          type="file"
          accept=".mid,.midi"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />

        {/* Deep Blue Rounded Pill Button */}
        <div style={pillStyle}>
          {/* Status micro-dot */}
          <span
            title={`Backend: ${connectionStatus}`}
            style={{
              ...statusDotStyle,
              backgroundColor:
                connectionStatus === 'connected'
                  ? '#4e9f76'
                  : connectionStatus === 'reconnecting'
                  ? '#c89b53'
                  : '#c25442',
            }}
          />

          {/* Primary Action Button (Matches sketch: Choose Midi File) */}
          {!selectedFileName ? (
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              style={pillMainActionStyle}
            >
              {isUploading ? 'Uploading Score...' : 'Choose Midi File'}
            </button>
          ) : (
            <div style={nowPlayingCluster}>
              <button
                type="button"
                onClick={() => setDropdownOpen(!dropdownOpen)}
                style={nowPlayingTitleStyle}
                title="Click to switch piece"
              >
                {selectedFileName}
              </button>

              {/* Play / Stop Stream Toggle */}
              <button
                type="button"
                onClick={isPlaying ? onStop : () => onPlay()}
                style={playToggleButtonStyle}
                title={isPlaying ? 'Stop playback' : 'Play RL agent'}
              >
                {isPlaying ? (
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M6 6h12v12H6z" />
                  </svg>
                ) : (
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                )}
              </button>

              {/* Tempo indicator */}
              <span style={tempoBadgeStyle}>{Math.round(tempo)} BPM</span>
            </div>
          )}

          {/* Dropdown Caret for Library */}
          <button
            type="button"
            onClick={() => setDropdownOpen(!dropdownOpen)}
            style={caretButtonStyle}
            title="Browse repertoire files"
          >
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M6 9l6 6 6-6" />
            </svg>
          </button>

          {/* Audio Engine Enable button if context needs unlock */}
          {!audioUnlocked && (
            <button
              type="button"
              onClick={onUnlockAudio}
              style={audioUnlockButtonStyle}
              title="Click to enable sound"
            >
              <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor">
                <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z" />
              </svg>
            </button>
          )}
        </div>

        {/* Repertoire Dropdown Menu */}
        {dropdownOpen && (
          <div style={dropdownMenuStyle}>
            <div style={dropdownHeaderStyle}>
              <span>Repertoire Library</span>
              <button
                type="button"
                onClick={() => {
                  fileInputRef.current?.click();
                  setDropdownOpen(false);
                }}
                style={uploadLinkStyle}
              >
                + Choose from disk
              </button>
            </div>

            <div style={dropdownListStyle}>
              {files.map((f) => (
                <div
                  key={f.path}
                  onClick={() => handleSelectPreloaded(f.path)}
                  style={{
                    ...dropdownItemStyle,
                    backgroundColor: selectedFile === f.path ? 'rgba(200, 155, 83, 0.16)' : 'transparent',
                    color: selectedFile === f.path ? '#c89b53' : '#ededed',
                  }}
                >
                  <span style={folderBadgeStyle}>[{f.folder}]</span>
                  <span style={itemFilenameStyle}>{f.name}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

/* Styles */
const chromeOverlayStyle: React.CSSProperties = {
  position: 'absolute',
  top: '22px',
  left: 0,
  right: 0,
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  pointerEvents: 'none',
  zIndex: 100,
};

const pillContainerStyle: React.CSSProperties = {
  position: 'relative',
  pointerEvents: 'auto',
};

const pillStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '12px',
  padding: '10px 24px',
  background: '#011242', // Deep navy blue matching sketch
  border: '1px solid rgba(255, 255, 255, 0.18)',
  borderRadius: '9999px',
  boxShadow: '0 12px 32px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.2)',
};

const statusDotStyle: React.CSSProperties = {
  width: '6px',
  height: '6px',
  borderRadius: '50%',
  display: 'inline-block',
};

const pillMainActionStyle: React.CSSProperties = {
  background: 'none',
  border: 'none',
  color: '#ffffff',
  fontSize: '15px',
  fontWeight: 400,
  letterSpacing: '0.01em',
  cursor: 'pointer',
  padding: '0 4px',
  fontFamily: 'serif',
};

const nowPlayingCluster: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '10px',
};

const nowPlayingTitleStyle: React.CSSProperties = {
  background: 'none',
  border: 'none',
  color: '#ffffff',
  fontSize: '14px',
  fontFamily: 'serif',
  fontWeight: 400,
  maxWidth: '220px',
  whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  cursor: 'pointer',
  padding: 0,
  textAlign: 'left',
};

const playToggleButtonStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: '24px',
  height: '24px',
  borderRadius: '50%',
  background: 'rgba(255, 255, 255, 0.15)',
  color: '#ffffff',
  cursor: 'pointer',
};

const tempoBadgeStyle: React.CSSProperties = {
  fontSize: '11px',
  fontFamily: 'JetBrains Mono, monospace',
  color: 'rgba(255, 255, 255, 0.6)',
};

const caretButtonStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  background: 'none',
  border: 'none',
  color: 'rgba(255, 255, 255, 0.7)',
  cursor: 'pointer',
  padding: '4px',
};

const audioUnlockButtonStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: '22px',
  height: '22px',
  borderRadius: '50%',
  background: 'rgba(200, 155, 83, 0.25)',
  color: '#c89b53',
  cursor: 'pointer',
};

const dropdownMenuStyle: React.CSSProperties = {
  position: 'absolute',
  top: 'calc(100% + 10px)',
  left: '50%',
  transform: 'translateX(-50%)',
  width: '320px',
  maxHeight: '380px',
  background: 'rgba(8, 14, 34, 0.96)',
  backdropFilter: 'blur(20px)',
  WebkitBackdropFilter: 'blur(20px)',
  border: '1px solid rgba(255, 255, 255, 0.15)',
  borderRadius: '12px',
  padding: '12px',
  display: 'flex',
  flexDirection: 'column',
  gap: '8px',
  boxShadow: '0 16px 40px rgba(0, 0, 0, 0.7)',
  zIndex: 200,
};

const dropdownHeaderStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  fontSize: '11px',
  color: 'rgba(255, 255, 255, 0.5)',
  paddingBottom: '8px',
  borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
};

const uploadLinkStyle: React.CSSProperties = {
  background: 'none',
  border: 'none',
  color: '#c89b53',
  fontSize: '11px',
  fontWeight: 600,
  cursor: 'pointer',
};

const dropdownListStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '4px',
  overflowY: 'auto',
  maxHeight: '290px',
};

const dropdownItemStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  padding: '8px 10px',
  borderRadius: '6px',
  fontSize: '12px',
  cursor: 'pointer',
  transition: 'background-color 0.15s ease',
};

const folderBadgeStyle: React.CSSProperties = {
  fontSize: '10px',
  color: 'rgba(255, 255, 255, 0.4)',
  fontFamily: 'JetBrains Mono, monospace',
};

const itemFilenameStyle: React.CSSProperties = {
  whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
};
