import React from 'react';
import * as THREE from 'three';
import { Canvas, useFrame } from '@react-three/fiber';
import { GrandPiano } from './GrandPiano';
import { PianoBench } from './PianoBench';
import { SeatedPianist } from './SeatedPianist';
import { ConcertStage } from './ConcertStage';

interface PianoSceneProps {
  activePitches: number[];
  results: Record<string, 'exact' | 'off_by_one' | 'wrong'>;
  pedalDown: boolean;
  isPlaying: boolean;
  hasFile: boolean;
}

const CameraRig: React.FC<{ isPlaying: boolean; hasFile: boolean }> = ({ isPlaying, hasFile }) => {
  // Front-facing concert-audience view (Side Elevation Profile):
  // Camera at seated-audience eye height (~1.82m), positioned perpendicular to the piano's long axis (at -X),
  // looking at the side profile of the piano and performer.
  // The pianist and bench sit on the left, the keyboard runs edge-on, the grand piano body
  // and propped lid extend to the right in profile against the rich red velvet curtain backdrop.
  // LookAt target is nudged along +Z so piano+performer sit comfortably left-of-center,
  // leaving clear space on the right for the Recital Program side panel card.
  const basePos = new THREE.Vector3(-5.8, 3.25, -2.6);
  const baseLookAt = new THREE.Vector3(0.2, 1.80, 0.75);

  useFrame((state) => {
    const camera = state.camera;

    if (!isPlaying && !hasFile) {
      // Extremely subtle, slow idle drift only when idle with no file loaded
      const t = state.clock.getElapsedTime() * 0.14;
      const driftX = Math.sin(t) * 0.02;
      const driftY = Math.cos(t * 0.65) * 0.012;
      camera.position.set(basePos.x + driftX, basePos.y + driftY, basePos.z);
    } else {
      // Completely static during performance
      camera.position.copy(basePos);
    }

    camera.lookAt(baseLookAt);
  });

  return null;
};

const StageLighting: React.FC = () => {
  // Target 1: Centered on the keyboard, pianist hands, face, and bench
  const performerAndKeyTarget = React.useMemo(() => {
    const obj = new THREE.Object3D();
    obj.position.set(-0.12, 1.85, -0.3);
    return obj;
  }, []);

  // Target 2: Centered on the grand piano case, soundboard, and lid
  const pianoBodyTarget = React.useMemo(() => {
    const obj = new THREE.Object3D();
    obj.position.set(0.0, 1.8, 1.2);
    return obj;
  }, []);

  // Target 3: Centered on the bench and pianist's legs
  const benchTarget = React.useMemo(() => {
    const obj = new THREE.Object3D();
    obj.position.set(-0.12, 1.0, -0.78);
    return obj;
  }, []);

  return (
    <>
      <primitive object={performerAndKeyTarget} />
      <primitive object={pianoBodyTarget} />
      <primitive object={benchTarget} />

      {/* Recital Hall Ambient: Soft red bounce keeping curtain fold detail visible but darker */}
      <ambientLight color="#2c1015" intensity={0.92} />

      {/* 1. PRIMARY PERFORMER & KEYBOARD OVERHEAD SPOTLIGHT:
          Positioned overhead from the audience side (-X), directly illuminating the
          keyboard, hands, pianist face/torso, and bench with wide coverage. */}
      <spotLight
        position={[-4.2, 7.8, -0.1]}
        target={performerAndKeyTarget}
        color="#fff4dc"
        intensity={11}
        decay={0}
        distance={9.5}
        angle={0.82}
        penumbra={0.55}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-bias={-0.0001}
      />

      {/* 2. DIRECT KEYBED & HANDS SPOTLIGHT:
          Shines down onto the white/black keys and pianist's hands,
          making individual keys crisp, bright, and legible. */}
      <spotLight
        position={[-2.4, 4.8, -0.2]}
        target={performerAndKeyTarget}
        color="#fff8ed"
        intensity={10}
        decay={0}
        distance={7.5}
        angle={0.78}
        penumbra={0.45}
      />

      {/* 3. WARM FRONT AUDIENCE FILL (DISTANCE-BOUNDED SPOTLIGHT):
          Direct audience-line fill ensuring the pianist's facial features, hands,
          and suit tailoring are clearly legible and three-dimensional,
          without spraying light onto the background curtains. */}
      <spotLight
        position={[-6.2, 3.2, -0.2]}
        target={performerAndKeyTarget}
        color="#ffeacc"
        intensity={9}
        decay={0}
        distance={8.0}
        angle={0.65}
        penumbra={0.55}
      />

      {/* 4. BENCH & LOWER BODY FILL:
          Ensures the bench, pianist's trousers, and shoes are properly legible on the floor. */}
      <spotLight
        position={[-5.0, 2.0, -0.8]}
        target={benchTarget}
        color="#ffdca8"
        intensity={6}
        decay={0}
        distance={7.0}
        angle={0.65}
        penumbra={0.5}
      />

      {/* 5. PIANO BODY & LID SPOTLIGHT:
          Covers the grand piano case, propped lid, and soundboard in the same warm-gold tone,
          blending seamlessly to form a single generous concert spotlight pool. */}
      <spotLight
        position={[-3.6, 7.8, 1.4]}
        target={pianoBodyTarget}
        color="#fff0d2"
        intensity={8}
        decay={0}
        distance={9.5}
        angle={0.75}
        penumbra={0.55}
        castShadow
      />

      {/* 6. RIM / SOUNDBOARD ACCENT:
          Grazes the open lid contour and soundboard interior with golden warmth. */}
      <spotLight
        position={[2.8, 6.2, 1.6]}
        target={pianoBodyTarget}
        color="#e8be6c"
        intensity={3.8}
        decay={0}
        distance={9.0}
        angle={0.7}
        penumbra={0.6}
        castShadow
      />
    </>
  );
};

export const PianoScene: React.FC<PianoSceneProps> = ({
  activePitches,
  results,
  pedalDown,
  isPlaying,
  hasFile,
}) => {
  return (
    <div style={sceneContainerStyle}>
      <Canvas
        shadows
        camera={{
          position: [-5.8, 3.25, -2.6],
          fov: 42,
          near: 0.1,
          far: 60,
        }}
        gl={{
          antialias: true,
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1.35,
        }}
        style={{ width: '100%', height: '100%' }}
      >
        {/* Composition Camera Rig */}
        <CameraRig isPlaying={isPlaying} hasFile={hasFile} />

        {/* Concert Stage Lighting */}
        <StageLighting />

        {/* STAGE ENVIRONMENT & ACTORS */}
        <group>
          {/* Recital Hall Environment: Burgundy Velvet Curtains & Polished Wood Floor */}
          <ConcertStage />

          {/* Grand Piano (Body, 47 Keys, Open Lid, Interior, Legs, Lyre) */}
          <GrandPiano
            activePitches={activePitches}
            results={results}
            pedalDown={pedalDown}
          />

          {/* Piano Bench (Grounded directly in front of keyboard) */}
          <PianoBench />

          {/* Unified Seated Pianist Figure (Single continuous body: torso, head, suit, connected arms & hands on keys) */}
          <SeatedPianist
            activePitches={activePitches}
            results={results}
          />
        </group>
      </Canvas>
    </div>
  );
};

const sceneContainerStyle: React.CSSProperties = {
  width: '100vw',
  height: '100vh',
  position: 'relative',
  background: '#090507', // Concert hall darkness
  overflow: 'hidden',
};
