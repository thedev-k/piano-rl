import { useState, useEffect, useRef, useCallback } from 'react';
import { soundEngine } from '../audio/soundEngine';

export type ConnectionStatus = 'connected' | 'reconnecting' | 'error' | 'disconnected';

export interface MidiFileItem {
  folder: string;
  name: string;
  path: string;
}

export interface Metrics {
  hits: number;
  wrong_presses: number;
  missed_notes: number;
  total_notes: number;
}

export interface StepEvent {
  step: number;
  beat: number;
  pitches: number[];
  results: Record<string, 'exact' | 'off_by_one' | 'wrong'>;
  overallResult: 'exact' | 'off_by_one' | 'wrong' | null;
  pedal: boolean;
}

export interface PieceInfo {
  path: string;
  name: string;
  info?: {
    tempo_bpm?: number;
    total_beats?: number;
    total_notes?: number;
    pitch_range?: [number, number];
  };
  melody_info?: {
    total_notes?: number;
    pitch_range?: [number, number];
  };
  dropped_notes?: number;
}

export function usePianoStream() {
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [files, setFiles] = useState<MidiFileItem[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [pieceInfo, setPieceInfo] = useState<PieceInfo | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [melodyOnly, setMelodyOnly] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [tempo, setTempo] = useState<number>(120);
  const [totalBeats, setTotalBeats] = useState<number>(0);

  const [metrics, setMetrics] = useState<Metrics>({
    hits: 0,
    wrong_presses: 0,
    missed_notes: 0,
    total_notes: 0,
  });

  const [currentStep, setCurrentStep] = useState<StepEvent>({
    step: 0,
    beat: 0,
    pitches: [],
    results: {},
    overallResult: null,
    pedal: false,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const isManuallyClosedRef = useRef<boolean>(false);

  // Fetch available files from /api/files
  const fetchFiles = useCallback(async () => {
    try {
      const res = await fetch('/api/files');
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data: MidiFileItem[] = await res.json();
      setFiles(data);
      if (data.length > 0 && !selectedFile) {
        setSelectedFile(data[0].path);
      }
      return data;
    } catch (err) {
      console.warn('Failed to fetch files from /api/files:', err);
      return [];
    }
  }, [selectedFile]);

  // Fetch piece info from /api/piece-info
  const fetchPieceInfo = useCallback(async (path: string) => {
    if (!path) return;
    try {
      const res = await fetch(`/api/piece-info?path=${encodeURIComponent(path)}`);
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data: PieceInfo = await res.json();
      setPieceInfo(data);
      if (data.info?.tempo_bpm) {
        setTempo(data.info.tempo_bpm);
      }
      if (data.info?.total_beats) {
        setTotalBeats(data.info.total_beats);
      }
    } catch (err) {
      console.warn('Failed to fetch piece info:', err);
    }
  }, []);

  // Connect to WebSocket with reconnect logic
  const connectWebSocket = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    isManuallyClosedRef.current = false;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    setConnectionStatus((prev) => (prev === 'connected' ? 'connected' : 'reconnecting'));

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnectionStatus('connected');
        reconnectAttemptsRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'init') {
            setIsPlaying(true);
            if (data.tempo) setTempo(data.tempo);
            if (data.total_beats) setTotalBeats(data.total_beats);
            setMetrics({
              hits: 0,
              wrong_presses: 0,
              missed_notes: 0,
              total_notes: data.notes?.length || 0,
            });
          } else if (data.type === 'step') {
            // Update sound
            if (Array.isArray(data.notes) && data.notes.length > 0) {
              for (const n of data.notes) {
                soundEngine.playNote(n.pitch, n.duration, n.velocity);
              }
            } else if (Array.isArray(data.pitches)) {
              for (const p of data.pitches) {
                soundEngine.playNote(p, 0.4, 0.7);
              }
            }

            soundEngine.setPedal(Boolean(data.pedal));

            setCurrentStep({
              step: data.step ?? 0,
              beat: data.beat ?? 0,
              pitches: data.pitches ?? [],
              results: data.results ?? {},
              overallResult: data.result ?? null,
              pedal: Boolean(data.pedal),
            });

            if (data.metrics) {
              setMetrics({
                hits: data.metrics.hits ?? 0,
                wrong_presses: data.metrics.wrong_presses ?? 0,
                missed_notes: data.metrics.missed_notes ?? 0,
                total_notes: data.metrics.total_notes ?? 0,
              });
            }
          } else if (data.type === 'done') {
            setIsPlaying(false);
            soundEngine.stopAll();
          } else if (data.type === 'error') {
            console.error('WebSocket playback error:', data.message);
            setIsPlaying(false);
            soundEngine.stopAll();
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onerror = () => {
        setConnectionStatus('error');
      };

      ws.onclose = () => {
        soundEngine.stopAll();
        setIsPlaying(false);
        if (!isManuallyClosedRef.current) {
          setConnectionStatus('reconnecting');
          reconnectAttemptsRef.current += 1;
          const delay = Math.min(1000 * Math.pow(1.5, reconnectAttemptsRef.current), 10000);
          reconnectTimeoutRef.current = window.setTimeout(() => {
            connectWebSocket();
          }, delay);
        } else {
          setConnectionStatus('disconnected');
        }
      };
    } catch {
      setConnectionStatus('error');
    }
  }, []);

  // Initialize connection and file list on mount
  useEffect(() => {
    fetchFiles();
    connectWebSocket();

    return () => {
      isManuallyClosedRef.current = true;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [fetchFiles, connectWebSocket]);

  // When selected file changes, update piece info
  useEffect(() => {
    if (selectedFile) {
      fetchPieceInfo(selectedFile);
    }
  }, [selectedFile, fetchPieceInfo]);

  // Command handlers
  const play = useCallback(
    async (overridePath?: string) => {
      await soundEngine.startAudioContext();
      const targetFile = overridePath || selectedFile;
      if (!targetFile) return;

      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            action: 'play',
            file: targetFile,
            melody_only: melodyOnly,
          })
        );
        setIsPlaying(true);
      } else {
        console.warn('Cannot play: WebSocket is not open');
        setConnectionStatus('reconnecting');
        connectWebSocket();
      }
    },
    [selectedFile, melodyOnly, connectWebSocket]
  );

  const stop = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'stop' }));
    }
    soundEngine.stopAll();
    setIsPlaying(false);
    setCurrentStep({
      step: 0,
      beat: 0,
      pitches: [],
      results: {},
      overallResult: null,
      pedal: false,
    });
  }, []);

  const changeSpeed = useCallback((speed: number) => {
    setPlaybackSpeed(speed);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          action: 'speed',
          value: speed,
        })
      );
    }
  }, []);

  const uploadFile = useCallback(
    async (file: File) => {
      const filename = file.name;
      const res = await fetch(`/api/upload?filename=${encodeURIComponent(filename)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/octet-stream',
        },
        body: file,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await res.json();
      await fetchFiles();
      if (data.file?.path) {
        setSelectedFile(data.file.path);
      }
      return data;
    },
    [fetchFiles]
  );

  return {
    connectionStatus,
    files,
    selectedFile,
    setSelectedFile,
    pieceInfo,
    isPlaying,
    melodyOnly,
    setMelodyOnly,
    playbackSpeed,
    tempo,
    totalBeats,
    metrics,
    currentStep,
    play,
    stop,
    changeSpeed,
    uploadFile,
    refreshFiles: fetchFiles,
    reconnect: connectWebSocket,
  };
}
