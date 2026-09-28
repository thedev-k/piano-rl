import React, { useState, useRef, useEffect, useCallback } from 'react';
import { MidiFileItem } from '../hooks/usePianoStream';
import gsap from 'gsap';

interface DynamicIslandProps {
  isPlaying: boolean;
  selectedFile: string;
  files: MidiFileItem[];
  tempo: number;
  playbackSpeed: number;
  onPlay: (path?: string) => void;
  onStop: () => void;
  onChangeSpeed: (speed: number) => void;
  onSelectFile: (path: string) => void;
}

// Animated EQ bars for when music is playing
const EQBars: React.FC<{ active: boolean }> = ({ active }) => {
  const bars = [0.4, 0.7, 0.55, 0.85, 0.45, 0.65, 0.5];
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 14, marginRight: 6 }}>
      {bars.map((h, i) => (
        <div
          key={i}
          style={{
            width: 2.5,
            height: active ? `${h * 14}px` : 3,
            background: '#c49a3e',
            borderRadius: 2,
            transition: active
              ? `height ${0.4 + i * 0.07}s ease-in-out ${i * 0.05}s`
              : 'height 0.3s ease-out',
            animation: active ? `eq-bounce-${i % 3} ${0.7 + i * 0.1}s ease-in-out infinite alternate` : 'none',
          }}
        />
      ))}
    </div>
  );
};

export const DynamicIsland: React.FC<DynamicIslandProps> = ({
  isPlaying,
  selectedFile,
  files,
  tempo,
  playbackSpeed,
  onPlay,
  onStop,
  onChangeSpeed,
  onSelectFile,
}) => {
  const [expanded, setExpanded] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const collapsedRef = useRef<HTMLDivElement>(null);
  const expandedRef = useRef<HTMLDivElement>(null);
  const tl = useRef<gsap.core.Timeline | null>(null);

  const selectedFileObj = files.find((f) => f.path === selectedFile);
  const displayName = selectedFileObj ? selectedFileObj.name.replace(/\.(mid|midi)$/i, '') : 'Choose a song';

  const expand = useCallback(() => {
    if (expanded) return;
    setExpanded(true);
    if (tl.current) tl.current.kill();

    // Fade out collapsed content instantly
    gsap.set(collapsedRef.current, { opacity: 0, pointerEvents: 'none' });

    tl.current = gsap.timeline();
    // Spring-like expansion from collapsed pill → squircle
    tl.current
      .to(containerRef.current, {
        width: 360,
        height: 260,
        borderRadius: 36,
        duration: 0.55,
        ease: 'back.out(1.2)',
      })
      .to(expandedRef.current, {
        opacity: 1,
        y: 0,
        duration: 0.25,
        ease: 'power2.out',
        pointerEvents: 'auto',
      }, '-=0.1');
  }, [expanded]);

  const collapse = useCallback(() => {
    if (!expanded) return;
    if (tl.current) tl.current.kill();

    gsap.set(expandedRef.current, { pointerEvents: 'none' });

    tl.current = gsap.timeline({
      onComplete: () => setExpanded(false),
    });
    tl.current
      .to(expandedRef.current, {
        opacity: 0,
        y: -8,
        duration: 0.15,
        ease: 'power2.in',
      })
      .to(containerRef.current, {
        width: isPlaying ? 220 : 160,
        height: 44,
        borderRadius: 22,
        duration: 0.4,
        ease: 'back.out(1.5)',
      }, '-=0.05')
      .to(collapsedRef.current, {
        opacity: 1,
        duration: 0.2,
        ease: 'power2.out',
        pointerEvents: 'auto',
      }, '-=0.15');
  }, [expanded, isPlaying]);

  // Animate width when isPlaying changes while collapsed
  useEffect(() => {
    if (!expanded) {
      gsap.to(containerRef.current, {
        width: isPlaying ? 220 : 160,
        duration: 0.35,
        ease: 'back.out(1.5)',
      });
    }
  }, [isPlaying, expanded]);

  // Init expanded content as invisible/offset
  useEffect(() => {
    gsap.set(expandedRef.current, { opacity: 0, y: -8, pointerEvents: 'none' });
  }, []);

  return (
    <>
      {/* EQ keyframe animations injected once */}
      <style>{`
        @keyframes eq-bounce-0 { from { transform: scaleY(0.5); } to { transform: scaleY(1); } }
        @keyframes eq-bounce-1 { from { transform: scaleY(0.3); } to { transform: scaleY(1); } }
        @keyframes eq-bounce-2 { from { transform: scaleY(0.6); } to { transform: scaleY(0.9); } }
        @media (prefers-reduced-motion: reduce) {
          .dynamic-island * { animation: none !important; transition: opacity 200ms ease !important; }
        }
      `}</style>

      <div style={islandWrapperStyle}>
        <div
          ref={containerRef}
          className="dynamic-island"
          style={islandStyle}
          onMouseLeave={collapse}
        >
          {/* COLLAPSED STATE */}
          <div
            ref={collapsedRef}
            style={collapsedContentStyle}
            onClick={expand}
          >
            <EQBars active={isPlaying} />
            <span style={collapsedTitleStyle}>{displayName}</span>

            {/* Play/Pause button — responds immediately on pointer-down */}
            <button
              style={pillBtnStyle}
              onClick={(e) => {
                e.stopPropagation();
                isPlaying ? onStop() : onPlay();
              }}
              onPointerDown={(e) => {
                e.stopPropagation();
                gsap.to(e.currentTarget, { scale: 0.88, duration: 0.08, ease: 'power2.out' });
              }}
              onPointerUp={(e) => {
                e.stopPropagation();
                gsap.to(e.currentTarget, { scale: 1, duration: 0.25, ease: 'back.out(2)' });
              }}
              onPointerLeave={(e) => {
                gsap.to(e.currentTarget, { scale: 1, duration: 0.2, ease: 'power2.out' });
              }}
            >
              {isPlaying ? (
                // Pause icon
                <svg width="10" height="11" viewBox="0 0 10 11" fill="none">
                  <rect x="0" y="0" width="3.5" height="11" rx="1.5" fill="currentColor" />
                  <rect x="6.5" y="0" width="3.5" height="11" rx="1.5" fill="currentColor" />
                </svg>
              ) : (
                // Play icon
                <svg width="11" height="13" viewBox="0 0 11 13" fill="none">
                  <path d="M0 0.5L11 6.5L0 12.5V0.5Z" fill="currentColor" />
                </svg>
              )}
            </button>
          </div>

          {/* EXPANDED STATE — squircle song picker */}
          <div ref={expandedRef} style={expandedContentStyle}>
            <div style={expandedHeaderStyle}>
              <div style={expandedTitleStyle}>{displayName}</div>
              <div style={bpmBadgeStyle}>{Math.round(tempo)} BPM</div>
            </div>

            {/* Song list */}
            <div style={songListStyle}>
              {files.map((f) => {
                const name = f.name.replace(/\.(mid|midi)$/i, '');
                const isSelected = f.path === selectedFile;
                return (
                  <button
                    key={f.path}
                    style={{
                      ...songItemStyle,
                      background: isSelected ? 'rgba(196,154,62,0.15)' : 'transparent',
                      color: isSelected ? '#c49a3e' : '#e8e8e8',
                      borderLeft: isSelected ? '2px solid #c49a3e' : '2px solid transparent',
                    }}
                    onPointerDown={(e) => {
                      gsap.to(e.currentTarget, { scale: 0.96, duration: 0.08 });
                    }}
                    onPointerUp={(e) => {
                      gsap.to(e.currentTarget, { scale: 1, duration: 0.25, ease: 'back.out(2)' });
                      onSelectFile(f.path);
                    }}
                    onPointerLeave={(e) => {
                      gsap.to(e.currentTarget, { scale: 1, duration: 0.2 });
                    }}
                  >
                    <span style={songNameStyle}>{name}</span>
                    <span style={folderStyle}>{f.folder}</span>
                  </button>
                );
              })}
            </div>

            {/* Bottom controls */}
            <div style={bottomRowStyle}>
              {/* Speed */}
              <div style={speedRowStyle}>
                <span style={speedLabelStyle}>{playbackSpeed.toFixed(1)}×</span>
                <input
                  type="range"
                  min="0.5"
                  max="2"
                  step="0.1"
                  value={playbackSpeed}
                  onChange={(e) => onChangeSpeed(parseFloat(e.target.value))}
                  style={sliderStyle}
                />
              </div>

              {/* Play/Stop */}
              <button
                style={expandedPlayBtnStyle}
                onClick={(e) => {
                  e.stopPropagation();
                  isPlaying ? onStop() : onPlay();
                  collapse();
                }}
                onPointerDown={(e) => {
                  e.stopPropagation();
                  gsap.to(e.currentTarget, { scale: 0.9, duration: 0.08 });
                }}
                onPointerUp={(e) => {
                  e.stopPropagation();
                  gsap.to(e.currentTarget, { scale: 1, duration: 0.3, ease: 'back.out(2)' });
                }}
                onPointerLeave={(e) => {
                  gsap.to(e.currentTarget, { scale: 1, duration: 0.2 });
                }}
              >
                {isPlaying ? 'Stop' : 'Play'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

/* ─────────────── Styles ─────────────── */

const islandWrapperStyle: React.CSSProperties = {
  position: 'absolute',
  top: 16,
  left: 0,
  width: '100%',
  display: 'flex',
  justifyContent: 'center',
  zIndex: 200,
  pointerEvents: 'none',
};

const islandStyle: React.CSSProperties = {
  position: 'relative',
  background: 'rgba(8, 8, 10, 0.92)',
  backdropFilter: 'blur(24px) saturate(180%)',
  WebkitBackdropFilter: 'blur(24px) saturate(180%)',
  color: '#fff',
  borderRadius: 22,
  width: 160,
  height: 44,
  overflow: 'hidden',
  cursor: 'pointer',
  boxShadow: '0 8px 32px rgba(0,0,0,0.6), inset 0 0.5px 0 rgba(255,255,255,0.08)',
  border: '0.5px solid rgba(255,255,255,0.1)',
  pointerEvents: 'auto',
  willChange: 'width, height, border-radius',
};

const collapsedContentStyle: React.CSSProperties = {
  position: 'absolute',
  inset: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '0 8px 0 14px',
  gap: 8,
};

const collapsedTitleStyle: React.CSSProperties = {
  flex: 1,
  fontSize: 13,
  fontWeight: 500,
  letterSpacing: '-0.01em',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
  fontFamily: 'system-ui, -apple-system, sans-serif',
};

const pillBtnStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: 30,
  height: 30,
  borderRadius: 15,
  background: 'rgba(255,255,255,0.08)',
  border: 'none',
  color: '#fff',
  cursor: 'pointer',
  flexShrink: 0,
  willChange: 'transform',
};

const expandedContentStyle: React.CSSProperties = {
  position: 'absolute',
  inset: 0,
  display: 'flex',
  flexDirection: 'column',
  padding: '20px 20px 16px',
  gap: 12,
  boxSizing: 'border-box',
};

const expandedHeaderStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'baseline',
};

const expandedTitleStyle: React.CSSProperties = {
  fontSize: 16,
  fontWeight: 600,
  letterSpacing: '-0.02em',
  color: '#fff',
  fontFamily: 'system-ui, -apple-system, sans-serif',
  flex: 1,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
};

const bpmBadgeStyle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 500,
  color: '#c49a3e',
  fontFamily: 'system-ui, -apple-system, sans-serif',
  letterSpacing: '0.02em',
  flexShrink: 0,
  marginLeft: 8,
};

const songListStyle: React.CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  display: 'flex',
  flexDirection: 'column',
  gap: 2,
  scrollbarWidth: 'none',
};

const songItemStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '7px 10px',
  borderRadius: 8,
  border: 'none',
  cursor: 'pointer',
  width: '100%',
  textAlign: 'left',
  transition: 'background 0.15s ease',
  willChange: 'transform',
};

const songNameStyle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 500,
  fontFamily: 'system-ui, -apple-system, sans-serif',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
  flex: 1,
};

const folderStyle: React.CSSProperties = {
  fontSize: 10,
  color: '#555',
  fontFamily: 'system-ui, -apple-system, sans-serif',
  marginLeft: 6,
  flexShrink: 0,
};

const bottomRowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 12,
  borderTop: '0.5px solid rgba(255,255,255,0.08)',
  paddingTop: 12,
};

const speedRowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  flex: 1,
};

const speedLabelStyle: React.CSSProperties = {
  fontSize: 11,
  color: '#888',
  fontFamily: 'system-ui, -apple-system, sans-serif',
  minWidth: 28,
  textAlign: 'right',
};

const sliderStyle: React.CSSProperties = {
  flex: 1,
  accentColor: '#c49a3e',
  height: 3,
};

const expandedPlayBtnStyle: React.CSSProperties = {
  padding: '8px 18px',
  borderRadius: 20,
  background: '#c49a3e',
  color: '#000',
  border: 'none',
  cursor: 'pointer',
  fontSize: 13,
  fontWeight: 700,
  letterSpacing: '-0.01em',
  fontFamily: 'system-ui, -apple-system, sans-serif',
  flexShrink: 0,
  willChange: 'transform',
};
