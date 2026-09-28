import React, { useEffect } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { GrandPiano } from '../3d/GrandPiano';
import { usePianoStream } from '../hooks/usePianoStream';
import { soundEngine } from '../audio/soundEngine';
import { DynamicIsland } from '../components/DynamicIsland';

const TopDownCameraRig = () => {
  const { camera } = useThree();
  useEffect(() => {
    // Piano keybed is at y=1.6. Keys are at z=0.0. Tail extends to z=-2.45.
    // Framed so keyboard spans 75-80% of width, with full grand piano body and interior visible.
    camera.position.set(0, 4.05, 1.30);
    camera.lookAt(0, 1.55, -0.90);
  }, [camera]);
  return null;
};

export const TopDownView: React.FC = () => {
  const {
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
  } = usePianoStream();

  useEffect(() => {
    const unlock = () => soundEngine.startAudioContext();
    window.addEventListener('click', unlock, { once: true });
    return () => window.removeEventListener('click', unlock);
  }, []);

  return (
    <div style={containerStyle}>
      <DynamicIsland
        isPlaying={isPlaying}
        selectedFile={selectedFile}
        files={files}
        tempo={tempo}
        playbackSpeed={playbackSpeed}
        onPlay={(path) => {
          soundEngine.startAudioContext();
          play(path);
        }}
        onStop={stop}
        onChangeSpeed={changeSpeed}
        onSelectFile={setSelectedFile}
      />

      {/* Lighter backdrop halo BEHIND canvas so black lacquer silhouette stands out cleanly */}
      <div style={backgroundHaloStyle} />

      <Canvas
        style={{ position: 'relative', zIndex: 1 }}
        camera={{ position: [0, 4.05, 1.30], fov: 38 }}
        gl={{ alpha: true, antialias: true }}
      >
        {/* Ambient base fill */}
        <ambientLight intensity={1.8} color="#2b3040" />

        {/* Warm key light from above and slightly in front */}
        <directionalLight
          position={[0.3, 5.8, 1.4]}
          intensity={3.0}
          color="#fff5e4"
        />

        {/* Dedicated crisp front light for the keyboard surface */}
        <directionalLight
          position={[0, 4.0, 1.2]}
          intensity={2.4}
          color="#fffdf8"
        />

        {/* Cool soft rim light from back-left (tracing straight bass edge and tail) */}
        <directionalLight
          position={[-3.6, 4.2, -3.2]}
          intensity={3.4}
          color="#8ab8ea"
        />

        {/* Cool soft rim light from back-right (tracing the curved bentside silhouette) */}
        <directionalLight
          position={[3.6, 4.2, -2.8]}
          intensity={3.8}
          color="#9ec6f8"
        />

        {/* Soft rear fill to maintain definition on the tail rim */}
        <directionalLight
          position={[0, 3.2, -3.8]}
          intensity={1.5}
          color="#6a82a0"
        />

        <TopDownCameraRig />

        {/* Grand piano model in top-down mode */}
        <GrandPiano
          activePitches={currentStep.pitches}
          pedalDown={currentStep.pedal}
          glowColor="#c49a3e"
          isTopDownView={true}
        />
      </Canvas>
    </div>
  );
};

const containerStyle: React.CSSProperties = {
  position: 'relative',
  width: '100vw',
  height: '100vh',
  overflow: 'hidden',
  background: '#191b22', // deep warm charcoal / navy-grey
};

const backgroundHaloStyle: React.CSSProperties = {
  position: 'absolute',
  top: 0,
  left: 0,
  width: '100%',
  height: '100%',
  background:
    'radial-gradient(ellipse 75% 65% at 50% 46%, rgba(68, 78, 102, 0.45) 0%, rgba(32, 35, 46, 0.85) 60%, rgba(20, 21, 28, 1) 100%)',
  pointerEvents: 'none',
  zIndex: 0,
};
