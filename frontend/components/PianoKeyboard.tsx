import React, { useEffect, useMemo } from 'react';
import { useGsapKeyAnimation, KeyVisualState } from '../hooks/useGsapKeyAnimation';

interface PianoKeyboardProps {
  activePitches: number[];
  results: Record<string, 'exact' | 'off_by_one' | 'wrong'>;
  pedalDown: boolean;
  onKeyClick?: (pitch: number) => void;
}

interface KeyData {
  pitch: number;
  noteName: string;
  isBlack: boolean;
  octave: number;
  whiteIndex: number; // Index among white keys (0 to 26)
  isMiddleC?: boolean;
}

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export const PianoKeyboard: React.FC<PianoKeyboardProps> = ({
  activePitches,
  results,
  pedalDown,
  onKeyClick,
}) => {
  const { registerKey, triggerKeyStrike } = useGsapKeyAnimation();

  // Generate 47 keys from C2 (36) to A#5 (82)
  const { whiteKeys, blackKeys, allKeys } = useMemo(() => {
    const whites: KeyData[] = [];
    const blacks: KeyData[] = [];
    const all: KeyData[] = [];

    let currentWhiteIndex = 0;

    for (let pitch = 36; pitch <= 82; pitch++) {
      const noteIndex = pitch % 12;
      const noteName = NOTE_NAMES[noteIndex];
      const octave = Math.floor(pitch / 12) - 1;
      const isBlack = noteName.includes('#');

      const keyData: KeyData = {
        pitch,
        noteName,
        octave,
        isBlack,
        whiteIndex: currentWhiteIndex,
        isMiddleC: pitch === 60, // C4
      };

      all.push(keyData);

      if (isBlack) {
        blacks.push(keyData);
      } else {
        whites.push(keyData);
        currentWhiteIndex++;
      }
    }

    return { whiteKeys: whites, blackKeys: blacks, allKeys: all };
  }, []);

  // Determine state for a pitch
  const getKeyVisualState = (pitch: number): KeyVisualState => {
    const pitchStr = pitch.toString();
    if (results[pitchStr]) {
      return results[pitchStr];
    }
    if (activePitches.includes(pitch)) {
      return 'active';
    }
    return 'idle';
  };

  // Trigger GSAP strike animations whenever activePitches or results update
  useEffect(() => {
    for (const key of allKeys) {
      const state = getKeyVisualState(key.pitch);
      if (state !== 'idle') {
        triggerKeyStrike(key.pitch, state, key.isBlack);
      } else {
        triggerKeyStrike(key.pitch, 'idle', key.isBlack);
      }
    }
  }, [activePitches, results, allKeys, triggerKeyStrike]);

  const totalWhites = whiteKeys.length; // 27

  return (
    <div style={containerStyle}>
      {/* Upper Acoustic Rail & Pedal Status */}
      <div style={railHeaderStyle}>
        <div style={railBrandStyle}>
          <span style={railBrandDot} />
          <span style={railBrandTitle}>ACOUSTIC STAGE · 47 KEYS (C2 – A#5)</span>
        </div>
        <div style={railStatusCluster}>
          <span style={pedalDown ? pedalActiveStyle : pedalIdleStyle}>
            {pedalDown ? '● SUSTAIN PEDAL DOWN' : '○ PEDAL UP'}
          </span>
        </div>
      </div>

      {/* Keyboard Bed */}
      <div style={keyboardBedStyle}>
        {/* White Keys Layer */}
        <div style={whiteKeysRowStyle}>
          {whiteKeys.map((k) => {
            const state = getKeyVisualState(k.pitch);
            const isHit = state === 'exact';
            const isOff = state === 'off_by_one';
            const isWrong = state === 'wrong';
            const isActive = state === 'active';

            return (
              <div
                key={k.pitch}
                ref={(el) => registerKey(k.pitch, el)}
                onClick={() => onKeyClick?.(k.pitch)}
                data-pitch={k.pitch}
                style={{
                  ...whiteKeyBaseStyle,
                  width: `${100 / totalWhites}%`,
                  backgroundColor: isHit
                    ? 'var(--status-strike-hit)'
                    : isOff
                    ? 'var(--status-strike-off)'
                    : isWrong
                    ? 'var(--status-strike-wrong)'
                    : isActive
                    ? 'var(--status-active-ai)'
                    : 'var(--key-ivory)',
                  boxShadow: isHit
                    ? '0 6px 14px var(--status-strike-hit-glow), inset 0 -3px 0 rgba(0,0,0,0.2)'
                    : isWrong
                    ? '0 6px 14px var(--status-strike-wrong-glow), inset 0 -3px 0 rgba(0,0,0,0.2)'
                    : 'inset 0 -4px 0 var(--key-ivory-shadow)',
                  color: isHit || isOff || isWrong || isActive ? '#ffffff' : 'var(--text-dim)',
                }}
              >
                {/* Octave & Middle C Markers */}
                {k.noteName === 'C' && (
                  <span style={octaveLabelStyle}>
                    C{k.octave}
                    {k.isMiddleC && <span style={middleCDotStyle} title="Middle C (MIDI 60)" />}
                  </span>
                )}
              </div>
            );
          })}
        </div>

        {/* Black Keys Layer (Positioned with true piano geometry relative to white keys) */}
        {blackKeys.map((k) => {
          const state = getKeyVisualState(k.pitch);
          const isHit = state === 'exact';
          const isOff = state === 'off_by_one';
          const isWrong = state === 'wrong';
          const isActive = state === 'active';

          // Black key width is ~62% of a white key width
          // Left position aligns at the boundary of the white key it follows
          const whiteWidthPercent = 100 / totalWhites;
          const blackWidthPercent = whiteWidthPercent * 0.62;
          const leftPercent = k.whiteIndex * whiteWidthPercent - blackWidthPercent / 2;

          return (
            <div
              key={k.pitch}
              ref={(el) => registerKey(k.pitch, el)}
              onClick={() => onKeyClick?.(k.pitch)}
              data-pitch={k.pitch}
              style={{
                ...blackKeyBaseStyle,
                width: `${blackWidthPercent}%`,
                left: `${leftPercent}%`,
                backgroundColor: isHit
                  ? 'var(--status-strike-hit)'
                  : isOff
                  ? 'var(--status-strike-off)'
                  : isWrong
                  ? 'var(--status-strike-wrong)'
                  : isActive
                  ? 'var(--status-active-ai)'
                  : 'var(--key-ebony)',
                boxShadow: isHit
                  ? '0 6px 12px var(--status-strike-hit-glow)'
                  : isWrong
                  ? '0 6px 12px var(--status-strike-wrong-glow)'
                  : '0 3px 6px rgba(0,0,0,0.6), inset 0 1px 0 var(--key-ebony-rim)',
                borderBottom: isHit || isWrong ? '2px solid rgba(255,255,255,0.4)' : '3px solid #0d0e11',
              }}
            >
              <div style={blackKeyFeltCap} />
            </div>
          );
        })}
      </div>
    </div>
  );
};

/* Styles */
const containerStyle: React.CSSProperties = {
  width: '100%',
  maxWidth: '1100px',
  margin: '0 auto',
  background: 'var(--surface-well)',
  border: '1px solid var(--border-subtle)',
  borderRadius: '6px',
  padding: '12px 14px 14px 14px',
  boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.04), 0 12px 32px rgba(0,0,0,0.4)',
};

const railHeaderStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  paddingBottom: '10px',
  borderBottom: '1px solid var(--border-subtle)',
  marginBottom: '10px',
};

const railBrandStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
};

const railBrandDot: React.CSSProperties = {
  width: '6px',
  height: '6px',
  borderRadius: '50%',
  backgroundColor: 'var(--accent-brass)',
};

const railBrandTitle: React.CSSProperties = {
  fontSize: '11px',
  fontWeight: 600,
  letterSpacing: '0.06em',
  color: 'var(--text-muted)',
};

const railStatusCluster: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '12px',
};

const pedalIdleStyle: React.CSSProperties = {
  fontSize: '11px',
  fontWeight: 500,
  color: 'var(--text-dim)',
  padding: '2px 8px',
  borderRadius: '3px',
  background: 'var(--surface-chassis)',
};

const pedalActiveStyle: React.CSSProperties = {
  fontSize: '11px',
  fontWeight: 600,
  color: 'var(--accent-brass)',
  padding: '2px 8px',
  borderRadius: '3px',
  background: 'rgba(200, 155, 83, 0.15)',
  border: '1px solid rgba(200, 155, 83, 0.3)',
};

const keyboardBedStyle: React.CSSProperties = {
  position: 'relative',
  width: '100%',
  height: '190px',
  background: '#090a0c',
  border: '1px solid #1a1c22',
  borderRadius: '3px',
  overflow: 'hidden',
  userSelect: 'none',
};

const whiteKeysRowStyle: React.CSSProperties = {
  display: 'flex',
  width: '100%',
  height: '100%',
};

const whiteKeyBaseStyle: React.CSSProperties = {
  position: 'relative',
  height: '100%',
  borderRight: '1px solid #202228',
  borderBottomLeftRadius: '3px',
  borderBottomRightRadius: '3px',
  cursor: 'pointer',
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'flex-end',
  alignItems: 'center',
  paddingBottom: '8px',
  transition: 'background-color 0.12s ease-out',
  boxSizing: 'border-box',
};

const blackKeyBaseStyle: React.CSSProperties = {
  position: 'absolute',
  top: 0,
  height: '62%',
  zIndex: 10,
  borderBottomLeftRadius: '2px',
  borderBottomRightRadius: '2px',
  cursor: 'pointer',
  transition: 'background-color 0.12s ease-out',
  boxSizing: 'border-box',
};

const blackKeyFeltCap: React.CSSProperties = {
  width: '100%',
  height: '4px',
  background: 'rgba(255,255,255,0.06)',
};

const octaveLabelStyle: React.CSSProperties = {
  fontSize: '10px',
  fontWeight: 600,
  fontFamily: 'JetBrains Mono, monospace',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  gap: '3px',
  pointerEvents: 'none',
};

const middleCDotStyle: React.CSSProperties = {
  width: '4px',
  height: '4px',
  borderRadius: '50%',
  backgroundColor: 'var(--accent-brass)',
};
