import React, { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';

export interface GrandPianoProps {
  activePitches: number[];
  results?: Record<string, 'exact' | 'off_by_one' | 'wrong'>;
  pedalDown: boolean;
  glowColor?: string;
  isTopDownView?: boolean;
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
  glowColor,
  isTopDownView = false,
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

  // High-Gloss Black Lacquer with deep clearcoat reflections
  const pianoBlackGlossMaterial = useMemo(
    () =>
      new THREE.MeshPhysicalMaterial({
        color: new THREE.Color('#0a0c10'),
        roughness: 0.12,
        metalness: 0.16,
        clearcoat: 1.0,
        clearcoatRoughness: 0.08,
        reflectivity: 0.95,
      }),
    []
  );

  const whiteKeyMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#faf9f6'),
        roughness: 0.15,
        metalness: 0.02,
      }),
    []
  );

  const glowingWhiteMaterial = useMemo(() => {
    const mat = whiteKeyMaterial.clone();
    if (glowColor) {
      mat.emissive = new THREE.Color(glowColor);
      mat.emissiveIntensity = 0.95;
    }
    return mat;
  }, [whiteKeyMaterial, glowColor]);

  const blackKeyMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#14161a'),
        roughness: 0.25,
        metalness: 0.08,
      }),
    []
  );

  const glowingBlackMaterial = useMemo(() => {
    const mat = blackKeyMaterial.clone();
    if (glowColor) {
      mat.emissive = new THREE.Color(glowColor);
      mat.emissiveIntensity = 0.95;
    }
    return mat;
  }, [blackKeyMaterial, glowColor]);

  const soundboardWoodMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#caa064'),
        roughness: 0.36,
        metalness: 0.04,
      }),
    []
  );

  const bridgeWoodMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#582d12'),
        roughness: 0.45,
        metalness: 0.02,
      }),
    []
  );

  const goldCastIronMaterial = useMemo(
    () =>
      new THREE.MeshPhysicalMaterial({
        color: new THREE.Color('#cca03f'),
        roughness: 0.26,
        metalness: 0.82,
        clearcoat: 0.35,
      }),
    []
  );

  const copperStringMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#c97c42'),
        roughness: 0.22,
        metalness: 0.92,
      }),
    []
  );

  const steelStringMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#e2eaf4'),
        roughness: 0.12,
        metalness: 0.96,
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

  // Key depression and glow animation loop
  useFrame((_, delta) => {
    const lerpSpeed = Math.min(1, delta * 24);

    allKeys.forEach((key) => {
      const mesh = keyMeshRefs.current[key.pitch];
      if (!mesh) return;

      const pitchStr = key.pitch.toString();
      const isPressed = activePitches.includes(key.pitch);
      const result = results ? results[pitchStr] : undefined;

      let targetDepression = 0;
      let targetRotationX = 0;

      if (isPressed || result) {
        if (result === 'wrong') {
          targetDepression = key.isBlack ? 0.045 : 0.052;
          targetRotationX = key.isBlack ? 0.050 : 0.056;
        } else {
          targetDepression = key.isBlack ? 0.028 : 0.034;
          targetRotationX = key.isBlack ? 0.032 : 0.038;
        }
      }

      const baseHeight = isTopDownView
        ? (key.isBlack ? blackKeyHeight / 2 + 0.016 : whiteKeyHeight / 2)
        : (key.isBlack ? whiteKeyHeight + 0.016 : whiteKeyHeight / 2);

      mesh.position.y = THREE.MathUtils.lerp(
        mesh.position.y,
        baseHeight - targetDepression,
        lerpSpeed
      );

      mesh.rotation.x = THREE.MathUtils.lerp(mesh.rotation.x, targetRotationX, lerpSpeed);

      // Handle glowing key state
      if (glowColor) {
        mesh.material = isPressed
          ? (key.isBlack ? glowingBlackMaterial : glowingWhiteMaterial)
          : (key.isBlack ? blackKeyMaterial : whiteKeyMaterial);
      }
    });
  });

  const pianoHeight = 2.1;
  const pianoBodyWidth = keyboardWidth + 0.28;
  const halfW = pianoBodyWidth / 2;

  // -------------------------------------------------------------
  // TOP-DOWN TRUE GRAND PIANO GEOMETRY (Hollow Rim, Open Soundboard)
  // -------------------------------------------------------------
  const { topDownRimGeometry, topDownLidGeometry, topDownSoundboardShape } = useMemo(() => {
    const Z_front = 0.20;
    const L_tail = 2.45;

    // Outer rim perimeter curve
    const outerShape = new THREE.Shape();
    outerShape.moveTo(-halfW, Z_front);
    outerShape.lineTo(halfW, Z_front);
    outerShape.bezierCurveTo(halfW - 0.02, L_tail - 2.05, halfW - 0.16, L_tail - 1.75, halfW - 0.32, L_tail - 1.45);
    outerShape.bezierCurveTo(halfW - 0.44, L_tail - 1.15, halfW - 0.38, L_tail - 0.70, 0.60, L_tail - 0.35);
    outerShape.bezierCurveTo(0.45, L_tail - 0.12, 0.25, L_tail, -0.1, L_tail);
    outerShape.bezierCurveTo(-0.6, L_tail, -halfW, L_tail - 0.08, -halfW, L_tail - 0.45);
    outerShape.lineTo(-halfW, Z_front);
    outerShape.closePath();

    // Sample outer points to compute centroid and clean inset hole
    const pts = outerShape.getPoints(60);
    let cx = 0, cy = 0;
    pts.forEach((p) => { cx += p.x; cy += p.y; });
    cx /= pts.length;
    cy /= pts.length;

    // Guaranteed non-self-intersecting inner hole (scaled inward by 0.94)
    const scale = 0.935;
    const hole = new THREE.Path();
    const p0 = pts[pts.length - 1];
    hole.moveTo(cx + (p0.x - cx) * scale, cy + (p0.y - cy) * scale);
    for (let i = pts.length - 2; i >= 0; i--) {
      const p = pts[i];
      hole.lineTo(cx + (p.x - cx) * scale, cy + (p.y - cy) * scale);
    }
    hole.closePath();

    const hollowRimShape = outerShape.clone();
    hollowRimShape.holes = [hole];

    // Extrude rim walls: 30cm high with soft glossy bevel
    const rimGeo = new THREE.ExtrudeGeometry(hollowRimShape, {
      depth: 0.30,
      bevelEnabled: true,
      bevelSegments: 3,
      steps: 1,
      bevelSize: 0.016,
      bevelThickness: 0.016,
    });

    // Propped Lid Shape
    const lidShape = new THREE.Shape();
    lidShape.moveTo(0, Z_front);
    lidShape.lineTo(0, L_tail - 0.45);
    lidShape.bezierCurveTo(0, L_tail - 0.08, halfW - 0.6, L_tail, halfW - 0.1, L_tail);
    lidShape.bezierCurveTo(halfW + 0.25, L_tail, halfW + 0.45, L_tail - 0.12, halfW + 0.60, L_tail - 0.35);
    lidShape.bezierCurveTo(pianoBodyWidth - 0.38, L_tail - 0.70, pianoBodyWidth - 0.44, L_tail - 1.15, pianoBodyWidth - 0.32, L_tail - 1.45);
    lidShape.bezierCurveTo(pianoBodyWidth - 0.16, L_tail - 1.75, pianoBodyWidth - 0.02, L_tail - 2.05, pianoBodyWidth, Z_front);
    lidShape.lineTo(0, Z_front);
    lidShape.closePath();

    const lidGeo = new THREE.ExtrudeGeometry(lidShape, {
      depth: 0.024,
      bevelEnabled: true,
      bevelSegments: 2,
      steps: 1,
      bevelSize: 0.012,
      bevelThickness: 0.012,
    });

    // Soundboard Floor Plate (Flat interior polygon matching inner hole)
    const sbShape = new THREE.Shape();
    sbShape.moveTo(cx + (p0.x - cx) * scale, cy + (p0.y - cy) * scale);
    for (let i = 1; i < pts.length; i++) {
      const p = pts[i];
      sbShape.lineTo(cx + (p.x - cx) * scale, cy + (p.y - cy) * scale);
    }
    sbShape.closePath();

    return {
      topDownRimGeometry: rimGeo,
      topDownLidGeometry: lidGeo,
      topDownSoundboardShape: sbShape,
    };
  }, [halfW, pianoBodyWidth]);

  // Strings calculation
  const strings = useMemo(() => {
    const list: { start: [number, number, number]; end: [number, number, number]; isBass: boolean }[] = [];
    // 8 Bass strings (copper, angled)
    for (let i = 0; i < 8; i++) {
      const frac = i / 7;
      const startX = -0.85 + frac * 0.55;
      const endX = -1.10 + frac * 0.45;
      list.push({
        start: [startX, 0.12, -0.32],
        end: [endX, 0.10, -1.95],
        isBass: true,
      });
    }
    // 24 Treble strings (silver-steel, grouped)
    for (let i = 0; i < 24; i++) {
      const frac = i / 23;
      const startX = -0.15 + frac * 1.05;
      const endX = -0.45 + frac * 1.05;
      const endZ = -1.85 + (1 - frac) * 1.05;
      list.push({
        start: [startX, 0.12, -0.32],
        end: [endX, 0.10, endZ],
        isBass: false,
      });
    }
    return list;
  }, []);

  // -------------------------------------------------------------
  // ORIGINAL CONCERT VIEW GEOMETRY (Preserved for /concert)
  // -------------------------------------------------------------
  const grandRimShapeConcert = useMemo(() => {
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

  const extrudeSettingsRimConcert = useMemo(
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

  const extrudeSettingsLidConcert = useMemo(
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

  // =============================================================
  // RENDER TOP-DOWN SIMPLE VIEW
  // =============================================================
  if (isTopDownView) {
    return (
      <group position={[0, 1.6, 0]}>
        {/* 1. KEYBOARD ASSEMBLY (Facing viewer in +Z, C2 on Left, A#5 on Right) */}
        <group position={[0, 0, 0]}>
          {/* Red damper felt strip behind key stems */}
          <mesh position={[0, whiteKeyHeight + 0.005, -0.21]} material={redFeltMaterial}>
            <boxGeometry args={[keyboardWidth + 0.02, 0.014, 0.02]} />
          </mesh>

          {/* White Keys */}
          {whiteKeys.map((k) => (
            <mesh
              key={k.pitch}
              ref={(el) => (keyMeshRefs.current[k.pitch] = el)}
              position={[k.x, whiteKeyHeight / 2, 0.0]}
              material={whiteKeyMaterial}
            >
              <boxGeometry args={[whiteKeyWidth * 0.96, whiteKeyHeight, whiteKeyLength]} />
            </mesh>
          ))}

          {/* Black Keys */}
          {blackKeys.map((k) => (
            <mesh
              key={k.pitch}
              ref={(el) => (keyMeshRefs.current[k.pitch] = el)}
              position={[k.x, blackKeyHeight / 2 + 0.016, -0.07]}
              material={blackKeyMaterial}
            >
              <boxGeometry args={[blackKeyWidth * 0.94, blackKeyHeight, blackKeyLength]} />
            </mesh>
          ))}

          {/* Keybed bottom shelf */}
          <mesh position={[0, -0.04, 0.0]} material={pianoBlackGlossMaterial}>
            <boxGeometry args={[pianoBodyWidth, 0.08, whiteKeyLength + 0.08]} />
          </mesh>

          {/* Left Cheek */}
          <mesh position={[-keyboardWidth / 2 - 0.07, 0.05, 0.0]} material={pianoBlackGlossMaterial}>
            <boxGeometry args={[0.14, 0.14, whiteKeyLength + 0.06]} />
          </mesh>

          {/* Right Cheek */}
          <mesh position={[keyboardWidth / 2 + 0.07, 0.05, 0.0]} material={pianoBlackGlossMaterial}>
            <boxGeometry args={[0.14, 0.14, whiteKeyLength + 0.06]} />
          </mesh>

          {/* Low Fallboard (just behind keys, low enough to leave interior 100% visible) */}
          <mesh position={[0, 0.06, -0.23]} material={pianoBlackGlossMaterial}>
            <boxGeometry args={[pianoBodyWidth, 0.08, 0.04]} />
          </mesh>
        </group>

        {/* 2. GRAND PIANO CASE & INTERIOR (Hollow Rim, Soundboard, Plate, Strings) */}
        <group position={[0, 0, 0]}>
          {/* Hollow Rim Wall (Glossy Black Lacquer) */}
          <group rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.28, 0]}>
            <mesh geometry={topDownRimGeometry} material={pianoBlackGlossMaterial} />
          </group>

          {/* Soundboard Floor (Warm Spruce Wood) */}
          <group rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.04, 0]}>
            <mesh material={soundboardWoodMaterial}>
              <shapeGeometry args={[topDownSoundboardShape]} />
            </mesh>
          </group>

          {/* Curved Bridges on Soundboard */}
          <mesh position={[-0.85, 0.06, -1.65]} rotation={[0, -0.25, 0]} material={bridgeWoodMaterial}>
            <boxGeometry args={[0.04, 0.025, 0.75]} />
          </mesh>
          <mesh position={[0.12, 0.06, -1.35]} rotation={[0, 0.35, 0]} material={bridgeWoodMaterial}>
            <boxGeometry args={[0.04, 0.025, 1.45]} />
          </mesh>

          {/* Cast-Iron Harp / Plate (Dull Gold Frame with Struts and Tuning Pins) */}
          <group position={[0, 0, 0]}>
            {/* Front Pinblock Plate */}
            <mesh position={[0, 0.08, -0.28]} material={goldCastIronMaterial}>
              <boxGeometry args={[pianoBodyWidth * 0.90, 0.03, 0.09]} />
            </mesh>
            {/* Tuning pins row across pinblock */}
            {Array.from({ length: 24 }).map((_, i) => (
              <mesh
                key={i}
                position={[-0.95 + i * 0.082, 0.105, -0.28]}
                material={brassMaterial}
              >
                <cylinderGeometry args={[0.005, 0.005, 0.024, 8]} />
              </mesh>
            ))}

            {/* Rear Tail Hitch Flange (Curved plate following the tail) */}
            <mesh position={[-0.2, 0.08, -2.15]} rotation={[0, 0.1, 0]} material={goldCastIronMaterial}>
              <cylinderGeometry args={[0.55, 0.65, 0.025, 24, 1, false, 0, Math.PI]} />
            </mesh>
            {/* Bentside Flange */}
            <mesh position={[0.42, 0.08, -1.35]} rotation={[0, -0.45, 0]} material={goldCastIronMaterial}>
              <boxGeometry args={[0.12, 0.025, 1.4]} />
            </mesh>

            {/* Structural Tension Struts (Golden Diagonal Beams) */}
            <mesh position={[-0.50, 0.11, -1.15]} rotation={[0, 0.20, 0]} material={goldCastIronMaterial}>
              <boxGeometry args={[0.048, 0.032, 1.65]} />
            </mesh>
            <mesh position={[0.08, 0.11, -1.30]} rotation={[0, -0.16, 0]} material={goldCastIronMaterial}>
              <boxGeometry args={[0.048, 0.032, 1.85]} />
            </mesh>
            <mesh position={[0.58, 0.11, -0.95]} rotation={[0, -0.44, 0]} material={goldCastIronMaterial}>
              <boxGeometry args={[0.048, 0.032, 1.25]} />
            </mesh>

            {/* Sound Ports (Circular Beveled Cutouts in Plate) */}
            <mesh position={[-0.15, 0.085, -1.6]} rotation={[-Math.PI / 2, 0, 0]} material={goldCastIronMaterial}>
              <ringGeometry args={[0.08, 0.15, 32]} />
            </mesh>
            <mesh position={[0.22, 0.085, -1.1]} rotation={[-Math.PI / 2, 0, 0]} material={goldCastIronMaterial}>
              <ringGeometry args={[0.07, 0.13, 32]} />
            </mesh>
            <mesh position={[-0.55, 0.085, -1.75]} rotation={[-Math.PI / 2, 0, 0]} material={goldCastIronMaterial}>
              <ringGeometry args={[0.06, 0.11, 32]} />
            </mesh>
          </group>

          {/* Genuine Piano Strings (Copper Bass + Silver Steel Treble) */}
          <group position={[0, 0, 0]}>
            {strings.map((str, idx) => {
              const dx = str.end[0] - str.start[0];
              const dy = str.end[1] - str.start[1];
              const dz = str.end[2] - str.start[2];
              const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
              const midX = (str.start[0] + str.end[0]) / 2;
              const midY = (str.start[1] + str.end[1]) / 2;
              const midZ = (str.start[2] + str.end[2]) / 2;
              const rotY = Math.atan2(dx, dz);

              return (
                <mesh
                  key={idx}
                  position={[midX, midY, midZ]}
                  rotation={[0, rotY, 0]}
                  material={str.isBass ? copperStringMaterial : steelStringMaterial}
                >
                  <boxGeometry args={[str.isBass ? 0.007 : 0.004, 0.003, len]} />
                </mesh>
              );
            })}
          </group>

          {/* 3. PROPPED OPEN GRAND LID (Hinged on Left Bass Rim, Open ~58° to fully reveal interior) */}
          <group position={[-halfW, 0.28, 0]}>
            {/* Rotate open ~58° (1.02 rad) around bass hinge */}
            <group rotation={[0, 0, 1.02]}>
              <mesh
                geometry={topDownLidGeometry}
                rotation={[-Math.PI / 2, 0, 0]}
                material={pianoBlackGlossMaterial}
              />
            </group>
          </group>

          {/* Slender Brass Prop Stick supporting the lid */}
          <mesh
            position={[0.20, 0.95, -1.15]}
            rotation={[0.12, 0, -0.38]}
            material={brassMaterial}
          >
            <cylinderGeometry args={[0.012, 0.012, 1.35, 16]} />
          </mesh>
        </group>
      </group>
    );
  }

  // =============================================================
  // RENDER ORIGINAL CONCERT VIEW (Used by PianoScene.tsx at /concert)
  // =============================================================
  return (
    <group position={[0, pianoHeight, 0]}>
      {/* 1. KEYBOARD ASSEMBLY (Keys sit from Z = -0.28 to Z = 0.16) */}
      <group position={[0, 0, 0]}>
        {/* Red Felt damper rail */}
        <mesh position={[0, whiteKeyHeight + 0.005, -0.26]} material={redFeltMaterial} visible={false}>
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
        {/* Left Cheek */}
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
            <extrudeGeometry args={[grandRimShapeConcert, extrudeSettingsRimConcert]} />
          </mesh>
        </group>

        {/* Soundboard Floor strictly inside rim */}
        <group rotation={[Math.PI / 2, 0, 0]} position={[0, 0.06, 0]}>
          <mesh material={soundboardWoodMaterial}>
            <shapeGeometry args={[grandRimShapeConcert]} />
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

      {/* 3. OPEN GRAND LID & PROP STICK */}
      <group position={[-halfW + 0.02, 0.76, 0.18]}>
        <group rotation={[0, 0, 0.62]}>
          <group rotation={[Math.PI / 2, 0, 0]} position={[halfW - 0.02, 0, 0]}>
            <mesh material={pianoBlackGlossMaterial} castShadow>
              <extrudeGeometry args={[grandRimShapeConcert, extrudeSettingsLidConcert]} />
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
