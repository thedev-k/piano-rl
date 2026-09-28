import React, { useEffect, useState } from 'react';
import { usePianoStream } from '../hooks/usePianoStream';
import { PianoScene } from '../3d/PianoScene';
import { SideControlPanel } from '../components/SideControlPanel';
import { soundEngine } from '../audio/soundEngine';

export const ConcertView: React.FC = () => {
  const {
    connectionStatus,
    files,
    selectedFile,
    setSelectedFile,
    isPlaying,
    tempo,
    playbackSpeed,
    currentStep,
    play,
    stop,
    changeSpeed,
    uploadFile,
  } = usePianoStream();

  const [audioUnlocked, setAudioUnlocked] = useState(false);

  useEffect(() => {
    if (soundEngine.audioReady) {
      setAudioUnlocked(true);
    }
  }, []);

  const handleUnlockAudio = async () => {
    const success = await soundEngine.startAudioContext();
    if (success) {
      setAudioUnlocked(true);
    }
  };

  const handleSelectFile = (path: string) => {
    setSelectedFile(path);
  };

  const handlePlay = (path?: string) => {
    handleUnlockAudio();
    play(path);
  };

  return (
    <div style={appWrapperStyle}>
      <SideControlPanel
        connectionStatus={connectionStatus}
        files={files}
        selectedFile={selectedFile}
        isPlaying={isPlaying}
        tempo={tempo}
        playbackSpeed={playbackSpeed}
        audioUnlocked={audioUnlocked}
        onSelectFile={handleSelectFile}
        onPlay={handlePlay}
        onStop={stop}
        onChangeSpeed={changeSpeed}
        onUpload={uploadFile}
        onUnlockAudio={handleUnlockAudio}
      />

      <PianoScene
        activePitches={currentStep.pitches}
        results={currentStep.results}
        pedalDown={currentStep.pedal}
        isPlaying={isPlaying}
        hasFile={Boolean(selectedFile)}
      />
    </div>
  );
};

const appWrapperStyle: React.CSSProperties = {
  position: 'relative',
  width: '100vw',
  height: '100vh',
  overflow: 'hidden',
  background: '#0a0b0e',
};
