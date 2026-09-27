import React from 'react';
import { useScoreboardGsap } from '../hooks/useScoreboardGsap';
import { Metrics } from '../hooks/usePianoStream';

interface ScoreboardProps {
  metrics: Metrics;
  tempo: number;
}

export const Scoreboard: React.FC<ScoreboardProps> = ({ metrics, tempo }) => {
  const animatedTotal = useScoreboardGsap(metrics.total_notes, {
    formatFn: (v) => v.toString().padStart(4, '0'),
  });
  const animatedHits = useScoreboardGsap(metrics.hits, {
    formatFn: (v) => v.toString().padStart(4, '0'),
  });
  const animatedWrong = useScoreboardGsap(metrics.wrong_presses, {
    formatFn: (v) => v.toString().padStart(4, '0'),
  });
  const animatedMiss = useScoreboardGsap(metrics.missed_notes, {
    formatFn: (v) => v.toString().padStart(4, '0'),
  });
  const animatedBpm = useScoreboardGsap(Math.round(tempo), {
    formatFn: (v) => v.toString(),
  });

  const rawAttempted = metrics.hits + metrics.wrong_presses + metrics.missed_notes;
  const accuracyPct = rawAttempted > 0 ? Math.min(100, Math.round((metrics.hits / rawAttempted) * 1000) / 10) : 100.0;
  const animatedAccuracy = useScoreboardGsap(Math.round(accuracyPct * 10), {
    formatFn: (v) => `${(v / 10).toFixed(1)}%`,
  });

  return (
    <div style={boardContainerStyle}>
      <div style={metricsGridStyle}>
        {/* Total Notes */}
        <div style={cellStyle}>
          <span style={labelStyle}>Total notes</span>
          <span ref={animatedTotal.ref} style={valueStyle} className="tabular-num">
            {animatedTotal.formatted}
          </span>
        </div>

        {/* Hits */}
        <div style={cellStyle}>
          <span style={{ ...labelStyle, color: 'var(--status-strike-hit)' }}>Hits (Exact & Off-1)</span>
          <span
            ref={animatedHits.ref}
            style={{ ...valueStyle, color: 'var(--status-strike-hit)' }}
            className="tabular-num"
          >
            {animatedHits.formatted}
          </span>
        </div>

        {/* Wrong */}
        <div style={cellStyle}>
          <span style={{ ...labelStyle, color: 'var(--status-strike-wrong)' }}>Wrong strikes</span>
          <span
            ref={animatedWrong.ref}
            style={{ ...valueStyle, color: 'var(--status-strike-wrong)' }}
            className="tabular-num"
          >
            {animatedWrong.formatted}
          </span>
        </div>

        {/* Misses */}
        <div style={cellStyle}>
          <span style={labelStyle}>Missed notes</span>
          <span ref={animatedMiss.ref} style={valueStyle} className="tabular-num">
            {animatedMiss.formatted}
          </span>
        </div>

        {/* Accuracy */}
        <div style={cellStyle}>
          <span style={labelStyle}>Accuracy</span>
          <span ref={animatedAccuracy.ref} style={accentValueStyle} className="tabular-num">
            {animatedAccuracy.formatted}
          </span>
        </div>

        {/* BPM */}
        <div style={{ ...cellStyle, borderRight: 'none' }}>
          <span style={labelStyle}>Tempo (BPM)</span>
          <span ref={animatedBpm.ref} style={valueStyle} className="tabular-num">
            {animatedBpm.formatted}
          </span>
        </div>
      </div>
    </div>
  );
};

/* Styles */
const boardContainerStyle: React.CSSProperties = {
  width: '100%',
  maxWidth: '1100px',
  margin: '0 auto',
  background: 'var(--surface-chassis)',
  border: '1px solid var(--border-subtle)',
  borderRadius: '6px',
  overflow: 'hidden',
};

const metricsGridStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
  width: '100%',
};

const cellStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  padding: '12px 16px',
  borderRight: '1px solid var(--border-subtle)',
  borderBottom: '1px solid var(--border-subtle)',
  background: 'var(--surface-chassis)',
  minWidth: 0,
};

const labelStyle: React.CSSProperties = {
  fontSize: '11px',
  fontWeight: 500,
  color: 'var(--text-muted)',
  marginBottom: '4px',
};

const valueStyle: React.CSSProperties = {
  fontSize: '22px',
  fontWeight: 600,
  color: 'var(--text-primary)',
  letterSpacing: '-0.02em',
  display: 'inline-block',
  transformOrigin: 'left center',
};

const accentValueStyle: React.CSSProperties = {
  fontSize: '22px',
  fontWeight: 600,
  color: 'var(--accent-brass)',
  letterSpacing: '-0.02em',
  display: 'inline-block',
  transformOrigin: 'left center',
};
