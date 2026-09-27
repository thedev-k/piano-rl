import React, { useRef, useMemo } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';

interface PianistHandsProps {
  activePitches: number[];
  results: Record<string, 'exact' | 'off_by_one' | 'wrong'>;
}

export const PianistHands: React.FC<PianistHandsProps> = ({ activePitches }) => {
  const leftForearmRef = useRef<THREE.Mesh>(null);
  const rightForearmRef = useRef<THREE.Mesh>(null);

  const leftCuffRef = useRef<THREE.Mesh>(null);
  const rightCuffRef = useRef<THREE.Mesh>(null);

  const leftHandRef = useRef<THREE.Group>(null);
  const rightHandRef = useRef<THREE.Group>(null);

  // Keyboard mapping constants
  const totalWhites = 27;
  const whiteKeyWidth = 0.082;
  const totalWidth = totalWhites * whiteKeyWidth;
  const startX = -totalWidth / 2 + whiteKeyWidth / 2;

  // Convert MIDI pitch (36 to 82) to X position on keyboard
  const getPitchX = (pitch: number): number => {
    let whiteCount = 0;
    for (let p = 36; p < pitch; p++) {
      const noteName = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][p % 12];
      if (!noteName.includes('#')) whiteCount++;
    }
    return startX + whiteCount * whiteKeyWidth;
  };

  // Materials
  const suitFabricMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#0c0f16'),
        roughness: 0.82,
        metalness: 0.05,
      }),
    []
  );

  const shirtCuffMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#f0f0f5'),
        roughness: 0.5,
      }),
    []
  );

  const skinMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#e0b699'),
        roughness: 0.45,
        metalness: 0.05,
      }),
    []
  );

  // Bench position anchor: seated at bench top (Z = -0.74, Y = 1.62)
  const leftElbow = useMemo(() => new THREE.Vector3(-0.35, 1.62, -0.74), []);
  const rightElbow = useMemo(() => new THREE.Vector3(0.25, 1.62, -0.74), []);

  const leftCurrentPos = useRef(new THREE.Vector3(-0.35, 2.18, 0.02));
  const rightCurrentPos = useRef(new THREE.Vector3(0.35, 2.18, 0.02));

  const leftTargetX = useRef(-0.35);
  const rightTargetX = useRef(0.35);

  const leftStrikeY = useRef(0);
  const rightStrikeY = useRef(0);

  const upVector = useMemo(() => new THREE.Vector3(0, 1, 0), []);

  useFrame((_, delta) => {
    const leftPitches = activePitches.filter((p) => p < 60);
    const rightPitches = activePitches.filter((p) => p >= 60);

    if (leftPitches.length > 0) {
      const avg = leftPitches.reduce((a, b) => a + b, 0) / leftPitches.length;
      leftTargetX.current = getPitchX(avg);
      leftStrikeY.current = -0.04;
    } else {
      leftTargetX.current = -0.35;
      leftStrikeY.current = 0;
    }

    if (rightPitches.length > 0) {
      const avg = rightPitches.reduce((a, b) => a + b, 0) / rightPitches.length;
      rightTargetX.current = getPitchX(avg);
      rightStrikeY.current = -0.04;
    } else {
      rightTargetX.current = 0.35;
      rightStrikeY.current = 0;
    }

    const lerpSpeed = Math.min(1, delta * 15);

    // Update Left Hand Position
    leftCurrentPos.current.x = THREE.MathUtils.lerp(leftCurrentPos.current.x, leftTargetX.current, lerpSpeed);
    leftCurrentPos.current.y = THREE.MathUtils.lerp(leftCurrentPos.current.y, 2.18 + leftStrikeY.current, lerpSpeed * 1.8);

    // Update Right Hand Position
    rightCurrentPos.current.x = THREE.MathUtils.lerp(rightCurrentPos.current.x, rightTargetX.current, lerpSpeed);
    rightCurrentPos.current.y = THREE.MathUtils.lerp(rightCurrentPos.current.y, 2.18 + rightStrikeY.current, lerpSpeed * 1.8);

    // 1. Position and orient Left Hand & Forearm
    if (leftHandRef.current) {
      leftHandRef.current.position.copy(leftCurrentPos.current);
    }
    if (leftForearmRef.current) {
      const dir = new THREE.Vector3().subVectors(leftCurrentPos.current, leftElbow);
      const dist = dir.length();
      dir.normalize();
      const mid = new THREE.Vector3().addVectors(leftElbow, leftCurrentPos.current).multiplyScalar(0.5);
      leftForearmRef.current.position.copy(mid);
      leftForearmRef.current.quaternion.setFromUnitVectors(upVector, dir);
      leftForearmRef.current.scale.set(1, dist / 0.85, 1);

      if (leftCuffRef.current) {
        const cuffPos = new THREE.Vector3().copy(leftCurrentPos.current).sub(dir.clone().multiplyScalar(0.04));
        leftCuffRef.current.position.copy(cuffPos);
        leftCuffRef.current.quaternion.setFromUnitVectors(upVector, dir);
      }
    }

    // 2. Position and orient Right Hand & Forearm
    if (rightHandRef.current) {
      rightHandRef.current.position.copy(rightCurrentPos.current);
    }
    if (rightForearmRef.current) {
      const dir = new THREE.Vector3().subVectors(rightCurrentPos.current, rightElbow);
      const dist = dir.length();
      dir.normalize();
      const mid = new THREE.Vector3().addVectors(rightElbow, rightCurrentPos.current).multiplyScalar(0.5);
      rightForearmRef.current.position.copy(mid);
      rightForearmRef.current.quaternion.setFromUnitVectors(upVector, dir);
      rightForearmRef.current.scale.set(1, dist / 0.85, 1);

      if (rightCuffRef.current) {
        const cuffPos = new THREE.Vector3().copy(rightCurrentPos.current).sub(dir.clone().multiplyScalar(0.04));
        rightCuffRef.current.position.copy(cuffPos);
        rightCuffRef.current.quaternion.setFromUnitVectors(upVector, dir);
      }
    }
  });

  return (
    <group>
      {/* LEFT FOREARM & SHIRT CUFF */}
      <mesh ref={leftForearmRef} material={suitFabricMaterial} castShadow>
        <cylinderGeometry args={[0.046, 0.064, 0.85, 16]} />
      </mesh>
      <mesh ref={leftCuffRef} material={shirtCuffMaterial}>
        <cylinderGeometry args={[0.045, 0.047, 0.04, 16]} />
      </mesh>

      {/* LEFT HAND & FINGERS (Directly resting above keys at keybed level) */}
      <group ref={leftHandRef} position={[-0.35, 2.18, 0.02]}>
        {/* Palm */}
        <mesh position={[0, 0.015, -0.01]} rotation={[0.08, 0.05, 0]} material={skinMaterial} castShadow>
          <boxGeometry args={[0.088, 0.034, 0.096]} />
        </mesh>
        {/* Thumb */}
        <mesh position={[0.046, -0.012, -0.02]} rotation={[0.2, 0.35, 0]} material={skinMaterial}>
          <capsuleGeometry args={[0.009, 0.038, 6, 8]} />
        </mesh>
        {/* Arched Fingers pointing into keys */}
        <mesh position={[0.022, -0.016, 0.038]} rotation={[0.55, 0, 0]} material={skinMaterial}>
          <capsuleGeometry args={[0.008, 0.046, 6, 8]} />
        </mesh>
        <mesh position={[0, -0.018, 0.042]} rotation={[0.55, 0, 0]} material={skinMaterial}>
          <capsuleGeometry args={[0.008, 0.048, 6, 8]} />
        </mesh>
        <mesh position={[-0.022, -0.016, 0.038]} rotation={[0.55, 0, 0]} material={skinMaterial}>
          <capsuleGeometry args={[0.0075, 0.045, 6, 8]} />
        </mesh>
        <mesh position={[-0.042, -0.014, 0.028]} rotation={[0.5, 0, 0]} material={skinMaterial}>
          <capsuleGeometry args={[0.007, 0.038, 6, 8]} />
        </mesh>
      </group>

      {/* RIGHT FOREARM & SHIRT CUFF */}
      <mesh ref={rightForearmRef} material={suitFabricMaterial} castShadow>
        <cylinderGeometry args={[0.046, 0.064, 0.85, 16]} />
      </mesh>
      <mesh ref={rightCuffRef} material={shirtCuffMaterial}>
        <cylinderGeometry args={[0.045, 0.047, 0.04, 16]} />
      </mesh>

      {/* RIGHT HAND & FINGERS */}
      <group ref={rightHandRef} position={[0.35, 2.18, 0.02]}>
        {/* Palm */}
        <mesh position={[0, 0.015, -0.01]} rotation={[0.08, -0.05, 0]} material={skinMaterial} castShadow>
          <boxGeometry args={[0.088, 0.034, 0.096]} />
        </mesh>
        {/* Thumb */}
        <mesh position={[-0.046, -0.012, -0.02]} rotation={[0.2, -0.35, 0]} material={skinMaterial}>
          <capsuleGeometry args={[0.009, 0.038, 6, 8]} />
        </mesh>
        {/* Arched Fingers */}
        <mesh position={[-0.022, -0.016, 0.038]} rotation={[0.55, 0, 0]} material={skinMaterial}>
          <capsuleGeometry args={[0.008, 0.046, 6, 8]} />
        </mesh>
        <mesh position={[0, -0.018, 0.042]} rotation={[0.55, 0, 0]} material={skinMaterial}>
          <capsuleGeometry args={[0.008, 0.048, 6, 8]} />
        </mesh>
        <mesh position={[0.022, -0.016, 0.038]} rotation={[0.55, 0, 0]} material={skinMaterial}>
          <capsuleGeometry args={[0.0075, 0.045, 6, 8]} />
        </mesh>
        <mesh position={[0.042, -0.014, 0.028]} rotation={[0.5, 0, 0]} material={skinMaterial}>
          <capsuleGeometry args={[0.007, 0.038, 6, 8]} />
        </mesh>
      </group>
    </group>
  );
};
