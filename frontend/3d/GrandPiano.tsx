import React, { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';

interface GrandPianoProps {
  activePitches: number[];
  results: Record<string, 'exact' | 'off_by_one' | 'wrong'>;
  pedalDown: boolean;
}

interface KeyInfo {
  pitch: number;
  noteName: string;
  isBlack: boolean;
  whiteIndex: number;
  x: number;
}

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export const GrandPiano: React.FC<GrandPianoProps> = ({
  activePitches,
  results,
  pedalDown,
}) => {
  const whiteKeyWidth = 0.082;
  const whiteKeyLength = 0.44;
  const whiteKeyHeight = 0.05;

  const blackKeyWidth = 0.046;
  const blackKeyLength = 0.28;
  const blackKeyHeight = 0.072;

  // Generate 47 keys: C2 (36) to A#5 (82)
  const { whiteKeys, blackKeys, allKeys, keyboardWidth } = useMemo(() => {
    const whites: KeyInfo[] = [];
    const blacks: KeyInfo[] = [];
    const all: KeyInfo[] = [];
    let currentWhiteIndex = 0;

    const totalWhites = 27;
    const totalWidth = totalWhites * whiteKeyWidth;
    const startX = -totalWidth / 2 + whiteKeyWidth / 2;

    for (let pitch = 36; pitch <= 82; pitch++) {
      const noteName = NOTE_NAMES[pitch % 12];
      const isBlack = noteName.includes('#');

      if (!isBlack) {
        const x = startX + currentWhiteIndex * whiteKeyWidth;
        const keyData: KeyInfo = {
          pitch,
          noteName,
          isBlack: false,
          whiteIndex: currentWhiteIndex,
          x,
        };
        whites.push(keyData);
        all.push(keyData);
        currentWhiteIndex++;
      }
    }

    for (let pitch = 36; pitch <= 82; pitch++) {
      const noteName = NOTE_NAMES[pitch % 12];
      const isBlack = noteName.includes('#');

      if (isBlack) {
        const prevWhite = whites.find((w) => w.pitch === pitch - 1);
        if (prevWhite) {
          const x = prevWhite.x + whiteKeyWidth / 2;
          const keyData: KeyInfo = {
            pitch,
            noteName,
            isBlack: true,
            whiteIndex: prevWhite.whiteIndex,
            x,
          };
          blacks.push(keyData);
          all.push(keyData);
        }
      }
    }

    return { whiteKeys: whites, blackKeys: blacks, allKeys: all, keyboardWidth: totalWidth };
  }, [whiteKeyWidth]);

  // Key mesh references for smooth physical depression
  const keyMeshRefs = useRef<{ [pitch: number]: THREE.Mesh | null }>({});

  useFrame((_, delta) => {
    const lerpSpeed = Math.min(1, delta * 24);

    allKeys.forEach((key) => {
      const mesh = keyMeshRefs.current[key.pitch];
      if (!mesh) return;

      const pitchStr = key.pitch.toString();
      const isPressed = activePitches.includes(key.pitch);
      const result = results[pitchStr];

      let targetDepression = 0;
      let targetRotationX = 0;

      if (isPressed || result) {
        if (result === 'wrong') {
          targetDepression = key.isBlack ? 0.052 : 0.06;
          targetRotationX = key.isBlack ? -0.065 : -0.07;
        } else {
          targetDepression = key.isBlack ? 0.032 : 0.038;
          targetRotationX = key.isBlack ? -0.038 : -0.044;
        }
      }

      mesh.position.y = THREE.MathUtils.lerp(
        mesh.position.y,
        (key.isBlack ? whiteKeyHeight + 0.016 : whiteKeyHeight / 2) - targetDepression,
        lerpSpeed
      );

      mesh.rotation.x = THREE.MathUtils.lerp(mesh.rotation.x, targetRotationX, lerpSpeed);
    });
  });

  // Piano Materials
  const pianoBlackGlossMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#101217'),
        roughness: 0.32,
        metalness: 0.12,
      }),
    []
  );

  const whiteKeyMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#faf8f2'),
        roughness: 0.22,
        metalness: 0.02,
      }),
    []
  );

  const blackKeyMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#16181d'),
        roughness: 0.32,
        metalness: 0.08,
      }),
    []
  );

  const goldCastIronMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#c49a3e'),
        roughness: 0.26,
        metalness: 0.82,
      }),
    []
  );

  const soundboardWoodMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#cfa05d'),
        roughness: 0.45,
        metalness: 0.05,
      }),
    []
  );

  const brassMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#dfaf4c'),
        roughness: 0.18,
        metalness: 0.92,
      }),
    []
  );

  const redFeltMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#9e1b1b'),
        roughness: 0.85,
      }),
    []
  );

  const pianoHeight = 2.1;
  const pianoBodyWidth = keyboardWidth + 0.36;
  const halfW = pianoBodyWidth / 2;

  // 2D Shape of Grand Piano Body starting at Z = 0 extending to Z = +3.0
  const grandRimShape = useMemo(() => {
    const l = 3.0;
    const shape = new THREE.Shape();
    shape.moveTo(-halfW, 0);
    shape.lineTo(-halfW, l - 0.5);
    shape.bezierCurveTo(-halfW + 0.1, l + 0.1, -halfW + 0.7, l + 0.2, 0.05, l);
    shape.bezierCurveTo(halfW - 0.35, l - 0.25, halfW - 0.45, l * 0.52, halfW - 0.04, l * 0.28);
    shape.lineTo(halfW, 0);
    shape.closePath();
    return shape;
  }, [halfW]);

  const extrudeSettingsRim = useMemo(
    () => ({
      depth: 0.68,
      bevelEnabled: true,
      bevelSegments: 2,
      steps: 1,
      bevelSize: 0.015,
      bevelThickness: 0.015,
    }),
    []
  );

  const extrudeSettingsLid = useMemo(
    () => ({
      depth: 0.036,
      bevelEnabled: true,
      bevelSegments: 2,
      steps: 1,
      bevelSize: 0.012,
      bevelThickness: 0.012,
    }),
    []
  );

  return (
    <group position={[0, pianoHeight, 0]}>
      {/* 1. KEYBOARD ASSEMBLY (Keys sit from Z = -0.28 to Z = 0.16) */}
      <group position={[0, 0, 0]}>
        {/* Red Felt damper rail behind key stems */}
        <mesh position={[0, whiteKeyHeight + 0.005, 0.16]} material={redFeltMaterial}>
          <boxGeometry args={[keyboardWidth + 0.04, 0.016, 0.04]} />
        </mesh>

        {/* White Keys */}
        {whiteKeys.map((k) => (
          <mesh
            key={k.pitch}
            ref={(el) => (keyMeshRefs.current[k.pitch] = el)}
            position={[k.x, whiteKeyHeight / 2, -0.06]}
            material={whiteKeyMaterial}
            castShadow
            receiveShadow
          >
            <boxGeometry args={[whiteKeyWidth * 0.96, whiteKeyHeight, whiteKeyLength]} />
          </mesh>
        ))}

        {/* Black Keys */}
        {blackKeys.map((k) => (
          <mesh
            key={k.pitch}
            ref={(el) => (keyMeshRefs.current[k.pitch] = el)}
            position={[k.x, whiteKeyHeight + 0.016, 0.02]}
            material={blackKeyMaterial}
            castShadow
            receiveShadow
          >
            <boxGeometry args={[blackKeyWidth * 0.94, blackKeyHeight, blackKeyLength]} />
          </mesh>
        ))}

        {/* Keybed casing lip under keys */}
        <mesh position={[0, -0.04, -0.08]} material={pianoBlackGlossMaterial}>
          <boxGeometry args={[pianoBodyWidth, 0.08, whiteKeyLength + 0.12]} />
        </mesh>
        {/* Left Cheek (contoured below white keys to expose keyboard in side profile) */}
        <mesh position={[-keyboardWidth / 2 - 0.08, -0.015, -0.08]} material={pianoBlackGlossMaterial}>
          <boxGeometry args={[0.12, 0.03, whiteKeyLength + 0.04]} />
        </mesh>
        <mesh position={[-keyboardWidth / 2 - 0.08, 0.08, 0.11]} material={pianoBlackGlossMaterial}>
          <boxGeometry args={[0.12, 0.18, 0.10]} />
        </mesh>
        {/* Right Cheek */}
        <mesh position={[keyboardWidth / 2 + 0.09, 0.08, -0.06]} material={pianoBlackGlossMaterial}>
          <boxGeometry args={[0.16, 0.18, whiteKeyLength + 0.08]} />
        </mesh>
      </group>

      {/* 2. GRAND PIANO BODY & SOUNDBOARD */}
      <group position={[0, 0.08, 0.18]}>
        {/* Rim Case Wall Extruded in +Z */}
        <group rotation={[Math.PI / 2, 0, 0]} position={[0, 0.68, 0]}>
          <mesh material={pianoBlackGlossMaterial} castShadow receiveShadow>
            <extrudeGeometry args={[grandRimShape, extrudeSettingsRim]} />
          </mesh>
        </group>

        {/* Soundboard Floor strictly inside rim */}
        <group rotation={[Math.PI / 2, 0, 0]} position={[0, 0.06, 0]}>
          <mesh material={soundboardWoodMaterial}>
            <shapeGeometry args={[grandRimShape]} />
          </mesh>
        </group>

        {/* Cast Iron Harp / Plate inside soundboard */}
        <group position={[0, 0.12, 0.1]}>
          <mesh position={[-0.32, 0, 1.4]} rotation={[0, 0.12, 0]} material={goldCastIronMaterial}>
            <boxGeometry args={[0.075, 0.045, 2.3]} />
          </mesh>
          <mesh position={[0.26, 0, 1.3]} rotation={[0, -0.22, 0]} material={goldCastIronMaterial}>
            <boxGeometry args={[0.075, 0.045, 2.0]} />
          </mesh>
          <mesh position={[-0.04, 0, 0.45]} material={goldCastIronMaterial}>
            <boxGeometry args={[pianoBodyWidth * 0.72, 0.05, 0.14]} />
          </mesh>
          <mesh position={[-0.05, 0.02, 1.4]} rotation={[-Math.PI / 2, 0, 0]}>
            <planeGeometry args={[1.5, 2.2]} />
            <meshBasicMaterial color="#d4d9e2" wireframe opacity={0.35} transparent />
          </mesh>
        </group>

        {/* Fallboard dividing keyboard from soundboard */}
        <mesh position={[0, 0.16, 0.02]} material={pianoBlackGlossMaterial}>
          <boxGeometry args={[pianoBodyWidth, 0.32, 0.06]} />
        </mesh>
      </group>

      {/* Music Desk above fallboard */}
      <group position={[0, 0.38, 0.42]} rotation={[-0.35, 0, 0]}>
        <mesh material={pianoBlackGlossMaterial}>
          <boxGeometry args={[1.45, 0.32, 0.026]} />
        </mesh>
        <mesh position={[0, -0.15, 0.03]} material={pianoBlackGlossMaterial}>
          <boxGeometry args={[1.48, 0.035, 0.065]} />
        </mesh>
      </group>

      {/* 3. OPEN GRAND LID & PROP STICK (Opens UPWARD ~36 degrees around bass hinge) */}
      <group position={[-halfW + 0.02, 0.76, 0.18]}>
        <group rotation={[0, 0, 0.62]}>
          <group rotation={[Math.PI / 2, 0, 0]} position={[halfW - 0.02, 0, 0]}>
            <mesh material={pianoBlackGlossMaterial} castShadow>
              <extrudeGeometry args={[grandRimShape, extrudeSettingsLid]} />
            </mesh>
          </group>
        </group>
      </group>

      {/* Wooden Prop Stick supporting the open lid */}
      <mesh
        position={[halfW - 0.22, 1.25, 1.2]}
        rotation={[-0.1, 0.1, -0.45]}
        material={pianoBlackGlossMaterial}
      >
        <cylinderGeometry args={[0.018, 0.018, 1.15, 16]} />
      </mesh>

      {/* 4. LEGS & PEDAL LYRE */}
      {/* Front Left Leg */}
      <group position={[-halfW + 0.22, -pianoHeight / 2, 0.2]}>
        <mesh material={pianoBlackGlossMaterial} castShadow>
          <cylinderGeometry args={[0.068, 0.046, pianoHeight - 0.08, 16]} />
        </mesh>
        <mesh position={[0, -pianoHeight / 2 + 0.04, 0]} material={brassMaterial}>
          <cylinderGeometry args={[0.042, 0.042, 0.065, 16]} />
        </mesh>
      </group>

      {/* Front Right Leg */}
      <group position={[halfW - 0.22, -pianoHeight / 2, 0.2]}>
        <mesh material={pianoBlackGlossMaterial} castShadow>
          <cylinderGeometry args={[0.068, 0.046, pianoHeight - 0.08, 16]} />
        </mesh>
        <mesh position={[0, -pianoHeight / 2 + 0.04, 0]} material={brassMaterial}>
          <cylinderGeometry args={[0.042, 0.042, 0.065, 16]} />
        </mesh>
      </group>

      {/* Rear Tail Leg */}
      <group position={[0.05, -pianoHeight / 2, 2.7]}>
        <mesh material={pianoBlackGlossMaterial} castShadow>
          <cylinderGeometry args={[0.068, 0.046, pianoHeight - 0.08, 16]} />
        </mesh>
        <mesh position={[0, -pianoHeight / 2 + 0.04, 0]} material={brassMaterial}>
          <cylinderGeometry args={[0.042, 0.042, 0.065, 16]} />
        </mesh>
      </group>

      {/* Pedal Lyre in center underneath keyboard */}
      <group position={[0, -pianoHeight / 2 + 0.1, 0.15]}>
        <mesh position={[-0.12, 0.2, 0]} material={pianoBlackGlossMaterial}>
          <boxGeometry args={[0.04, 0.65, 0.04]} />
        </mesh>
        <mesh position={[0.12, 0.2, 0]} material={pianoBlackGlossMaterial}>
          <boxGeometry args={[0.04, 0.65, 0.04]} />
        </mesh>
        <mesh position={[0, -0.15, 0]} material={pianoBlackGlossMaterial}>
          <boxGeometry args={[0.34, 0.08, 0.1]} />
        </mesh>
        {/* 3 Brass Pedals */}
        <mesh position={[-0.08, -0.16, -0.06]} material={brassMaterial}>
          <boxGeometry args={[0.035, 0.02, 0.14]} />
        </mesh>
        <mesh position={[0, -0.16, -0.06]} material={brassMaterial}>
          <boxGeometry args={[0.035, 0.02, 0.14]} />
        </mesh>
        <mesh
          position={[0.08, pedalDown ? -0.18 : -0.16, -0.06]}
          rotation={[pedalDown ? 0.12 : 0, 0, 0]}
          material={brassMaterial}
        >
          <boxGeometry args={[0.035, 0.02, 0.14]} />
        </mesh>
      </group>
    </group>
  );
};
