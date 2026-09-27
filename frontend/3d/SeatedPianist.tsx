import React, { useRef, useMemo } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';

interface SeatedPianistProps {
  activePitches: number[];
  results?: Record<string, 'exact' | 'off_by_one' | 'wrong'>;
}

export const SeatedPianist: React.FC<SeatedPianistProps> = ({ activePitches }) => {
  // Rig references for real-time 2-segment arm kinematics
  const leftUpperArmRef = useRef<THREE.Mesh>(null);
  const leftElbowMeshRef = useRef<THREE.Mesh>(null);
  const leftForearmRef = useRef<THREE.Mesh>(null);
  const leftCuffRef = useRef<THREE.Mesh>(null);
  const leftHandGroupRef = useRef<THREE.Group>(null);

  const rightUpperArmRef = useRef<THREE.Mesh>(null);
  const rightElbowMeshRef = useRef<THREE.Mesh>(null);
  const rightForearmRef = useRef<THREE.Mesh>(null);
  const rightCuffRef = useRef<THREE.Mesh>(null);
  const rightHandGroupRef = useRef<THREE.Group>(null);

  // Keyboard mapping constants (47 keys, C2 to A#5)
  const totalWhites = 27;
  const whiteKeyWidth = 0.082;
  const totalWidth = totalWhites * whiteKeyWidth;
  const startX = -totalWidth / 2 + whiteKeyWidth / 2;

  const getPitchX = (pitch: number): number => {
    let whiteCount = 0;
    for (let p = 36; p < pitch; p++) {
      const noteName = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][p % 12];
      if (!noteName.includes('#')) whiteCount++;
    }
    return startX + whiteCount * whiteKeyWidth;
  };

  // Materials
  const suitFabric = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#1e2330'),
        roughness: 0.68,
        metalness: 0.08,
      }),
    []
  );

  const shirtWhite = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#ffffff'),
        roughness: 0.35,
        metalness: 0.02,
      }),
    []
  );

  const bowTieBlack = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#08080c'),
        roughness: 0.35,
        metalness: 0.1,
      }),
    []
  );

  const skinTone = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#e8bfa2'),
        roughness: 0.36,
        metalness: 0.04,
      }),
    []
  );

  const hairDark = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#16100c'),
        roughness: 0.65,
        metalness: 0.05,
      }),
    []
  );

  const shoesLeather = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#08080a'),
        roughness: 0.28,
        metalness: 0.2,
      }),
    []
  );

  // Bench position anchor: seated at bench (X = -0.12, Y = 1.42, Z = -0.78)
  // Shoulders attach directly to the torso:
  const leftShoulderPos = useMemo(() => new THREE.Vector3(-0.31, 2.12, -0.74), []);
  const rightShoulderPos = useMemo(() => new THREE.Vector3(0.07, 2.12, -0.74), []);

  // Hand tracking state
  const leftCurrentPos = useRef(new THREE.Vector3(-0.35, 2.18, -0.06));
  const rightCurrentPos = useRef(new THREE.Vector3(0.35, 2.18, -0.06));

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

    // Update Hand positions
    leftCurrentPos.current.x = THREE.MathUtils.lerp(leftCurrentPos.current.x, leftTargetX.current, lerpSpeed);
    leftCurrentPos.current.y = THREE.MathUtils.lerp(leftCurrentPos.current.y, 2.18 + leftStrikeY.current, lerpSpeed * 1.8);

    rightCurrentPos.current.x = THREE.MathUtils.lerp(rightCurrentPos.current.x, rightTargetX.current, lerpSpeed);
    rightCurrentPos.current.y = THREE.MathUtils.lerp(rightCurrentPos.current.y, 2.18 + rightStrikeY.current, lerpSpeed * 1.8);

    // Update Left Hand Group
    if (leftHandGroupRef.current) {
      leftHandGroupRef.current.position.copy(leftCurrentPos.current);
    }
    // Update Right Hand Group
    if (rightHandGroupRef.current) {
      rightHandGroupRef.current.position.copy(rightCurrentPos.current);
    }

    // Kinematic Elbow calculation:
    // Elbows sit naturally below the shoulders and slightly behind the keys
    const leftElbowPos = new THREE.Vector3(
      leftShoulderPos.x * 0.6 + leftCurrentPos.current.x * 0.4 - 0.04,
      1.72,
      leftShoulderPos.z * 0.55 + leftCurrentPos.current.z * 0.45 - 0.05
    );

    const rightElbowPos = new THREE.Vector3(
      rightShoulderPos.x * 0.6 + rightCurrentPos.current.x * 0.4 + 0.04,
      1.72,
      rightShoulderPos.z * 0.55 + rightCurrentPos.current.z * 0.45 - 0.05
    );

    // Position elbow joint spheres
    if (leftElbowMeshRef.current) {
      leftElbowMeshRef.current.position.copy(leftElbowPos);
    }
    if (rightElbowMeshRef.current) {
      rightElbowMeshRef.current.position.copy(rightElbowPos);
    }

    // 1. LEFT UPPER ARM (from leftShoulder to leftElbow)
    if (leftUpperArmRef.current) {
      const dir = new THREE.Vector3().subVectors(leftElbowPos, leftShoulderPos);
      const len = dir.length();
      dir.normalize();
      const mid = new THREE.Vector3().addVectors(leftShoulderPos, leftElbowPos).multiplyScalar(0.5);
      leftUpperArmRef.current.position.copy(mid);
      leftUpperArmRef.current.quaternion.setFromUnitVectors(upVector, dir);
      leftUpperArmRef.current.scale.set(1, len / 0.5, 1);
    }

    // 2. LEFT FOREARM (from leftElbow to leftHand)
    if (leftForearmRef.current) {
      const dir = new THREE.Vector3().subVectors(leftCurrentPos.current, leftElbowPos);
      const len = dir.length();
      dir.normalize();
      const mid = new THREE.Vector3().addVectors(leftElbowPos, leftCurrentPos.current).multiplyScalar(0.5);
      leftForearmRef.current.position.copy(mid);
      leftForearmRef.current.quaternion.setFromUnitVectors(upVector, dir);
      leftForearmRef.current.scale.set(1, len / 0.5, 1);

      if (leftCuffRef.current) {
        const cuffPos = new THREE.Vector3().copy(leftCurrentPos.current).sub(dir.clone().multiplyScalar(0.04));
        leftCuffRef.current.position.copy(cuffPos);
        leftCuffRef.current.quaternion.setFromUnitVectors(upVector, dir);
      }
    }

    // 3. RIGHT UPPER ARM (from rightShoulder to rightElbow)
    if (rightUpperArmRef.current) {
      const dir = new THREE.Vector3().subVectors(rightElbowPos, rightShoulderPos);
      const len = dir.length();
      dir.normalize();
      const mid = new THREE.Vector3().addVectors(rightShoulderPos, rightElbowPos).multiplyScalar(0.5);
      rightUpperArmRef.current.position.copy(mid);
      rightUpperArmRef.current.quaternion.setFromUnitVectors(upVector, dir);
      rightUpperArmRef.current.scale.set(1, len / 0.5, 1);
    }

    // 4. RIGHT FOREARM (from rightElbow to rightHand)
    if (rightForearmRef.current) {
      const dir = new THREE.Vector3().subVectors(rightCurrentPos.current, rightElbowPos);
      const len = dir.length();
      dir.normalize();
      const mid = new THREE.Vector3().addVectors(rightElbowPos, rightCurrentPos.current).multiplyScalar(0.5);
      rightForearmRef.current.position.copy(mid);
      rightForearmRef.current.quaternion.setFromUnitVectors(upVector, dir);
      rightForearmRef.current.scale.set(1, len / 0.5, 1);

      if (rightCuffRef.current) {
        const cuffPos = new THREE.Vector3().copy(rightCurrentPos.current).sub(dir.clone().multiplyScalar(0.04));
        rightCuffRef.current.position.copy(cuffPos);
        rightCuffRef.current.quaternion.setFromUnitVectors(upVector, dir);
      }
    }
  });

  return (
    <group>
      {/* ======================================================== */}
      {/* 1. SEATED BODY: LEGS, PELVIS, TORSO, HEAD (Fixed to Bench) */}
      {/* ======================================================== */}
      <group position={[-0.12, 1.42, -0.78]}>
        {/* Pelvis / Hips seated firmly on cushion */}
        <mesh position={[0, 0.08, 0]} material={suitFabric} castShadow receiveShadow>
          <boxGeometry args={[0.38, 0.16, 0.32]} />
        </mesh>

        {/* Thighs extending forward horizontally toward piano */}
        <mesh position={[-0.11, 0.05, 0.18]} rotation={[Math.PI / 2 + 0.08, 0, 0]} material={suitFabric} castShadow>
          <cylinderGeometry args={[0.07, 0.06, 0.44, 16]} />
        </mesh>
        <mesh position={[0.11, 0.05, 0.18]} rotation={[Math.PI / 2 + 0.08, 0, 0]} material={suitFabric} castShadow>
          <cylinderGeometry args={[0.07, 0.06, 0.44, 16]} />
        </mesh>

        {/* Knee joints */}
        <mesh position={[-0.11, 0.05, 0.4]} material={suitFabric} castShadow>
          <sphereGeometry args={[0.065, 12, 12]} />
        </mesh>
        <mesh position={[0.11, 0.05, 0.4]} material={suitFabric} castShadow>
          <sphereGeometry args={[0.065, 12, 12]} />
        </mesh>

        {/* Shins / Trouser legs descending to floor */}
        <mesh position={[-0.11, -0.65, 0.4]} rotation={[0.04, 0, 0]} material={suitFabric} castShadow>
          <cylinderGeometry args={[0.06, 0.052, 0.8, 16]} />
        </mesh>
        <mesh position={[0.11, -0.65, 0.4]} rotation={[0.04, 0, 0]} material={suitFabric} castShadow>
          <cylinderGeometry args={[0.06, 0.052, 0.8, 16]} />
        </mesh>

        {/* Black Dress Shoes on the stage floor */}
        <mesh position={[-0.11, -1.04, 0.48]} material={shoesLeather} castShadow>
          <boxGeometry args={[0.1, 0.08, 0.26]} />
        </mesh>
        <mesh position={[0.11, -1.04, 0.48]} material={shoesLeather} castShadow>
          <boxGeometry args={[0.1, 0.08, 0.26]} />
        </mesh>

        {/* Lower Torso / Waist */}
        <mesh position={[0, 0.28, 0.01]} rotation={[0.08, 0, 0]} material={suitFabric} castShadow>
          <boxGeometry args={[0.35, 0.3, 0.25]} />
        </mesh>

        {/* Upper Torso / Chest with tailored suit cut, tilted slightly forward */}
        <mesh position={[0, 0.58, 0.03]} rotation={[0.1, 0, 0]} material={suitFabric} castShadow>
          <boxGeometry args={[0.42, 0.36, 0.27]} />
        </mesh>

        {/* Shoulders Yolk / Deltoid foundation */}
        <mesh position={[0, 0.72, 0.04]} rotation={[0.1, 0, 0]} material={suitFabric} castShadow>
          <boxGeometry args={[0.48, 0.12, 0.25]} />
        </mesh>

        {/* Left and Right Shoulder Joint Sockets (matching arm origin points) */}
        <mesh position={[-0.19, 0.70, 0.04]} material={suitFabric} castShadow>
          <sphereGeometry args={[0.075, 16, 16]} />
        </mesh>
        <mesh position={[0.19, 0.70, 0.04]} material={suitFabric} castShadow>
          <sphereGeometry args={[0.075, 16, 16]} />
        </mesh>

        {/* White Dress Shirt V-Neck & Wing Collar */}
        <mesh position={[0, 0.65, 0.17]} rotation={[0.1, 0, 0]} material={shirtWhite}>
          <boxGeometry args={[0.12, 0.22, 0.02]} />
        </mesh>
        <mesh position={[0, 0.78, 0.04]} rotation={[0.1, 0, 0]} material={shirtWhite}>
          <cylinderGeometry args={[0.078, 0.088, 0.06, 16]} />
        </mesh>

        {/* Black Formal Bow Tie */}
        <group position={[0, 0.76, 0.14]} rotation={[0.1, 0, 0]}>
          <mesh material={bowTieBlack}>
            <sphereGeometry args={[0.022, 8, 8]} />
          </mesh>
          <mesh position={[-0.04, 0, 0]} material={bowTieBlack}>
            <boxGeometry args={[0.05, 0.035, 0.015]} />
          </mesh>
          <mesh position={[0.04, 0, 0]} material={bowTieBlack}>
            <boxGeometry args={[0.05, 0.035, 0.015]} />
          </mesh>
        </group>

        {/* Neck */}
        <mesh position={[0, 0.84, 0.04]} rotation={[0.12, 0, 0]} material={skinTone} castShadow>
          <cylinderGeometry args={[0.05, 0.054, 0.09, 16]} />
        </mesh>

        {/* Head (sculpted jawline, tilted downward toward keys) */}
        <group position={[0, 0.98, 0.06]} rotation={[0.22, 0, 0]}>
          <mesh material={skinTone} castShadow>
            <sphereGeometry args={[0.118, 24, 24]} />
          </mesh>
          {/* Jaw / Chin structure */}
          <mesh position={[0, -0.06, 0.05]} rotation={[-0.3, 0, 0]} material={skinTone}>
            <boxGeometry args={[0.1, 0.08, 0.09]} />
          </mesh>
          {/* Nose silhouette in side profile */}
          <mesh position={[0, 0.01, 0.13]} rotation={[0.2, 0, 0]} material={skinTone}>
            <boxGeometry args={[0.024, 0.04, 0.03]} />
          </mesh>
          {/* Dark Concert Hair */}
          <mesh position={[0, 0.03, -0.01]} material={hairDark} castShadow>
            <sphereGeometry args={[0.124, 24, 16, 0, Math.PI * 2, 0, Math.PI / 1.7]} />
          </mesh>
        </group>
      </group>

      {/* ======================================================== */}
      {/* 2. CONTINUOUS RIGGED ARMS & HANDS (Seamlessly Connected)  */}
      {/* ======================================================== */}
      {/* LEFT ARM */}
      {/* Left Upper Arm (Originates from Left Shoulder) */}
      <mesh ref={leftUpperArmRef} material={suitFabric} castShadow>
        <cylinderGeometry args={[0.052, 0.048, 0.5, 16]} />
      </mesh>
      {/* Left Elbow Joint */}
      <mesh ref={leftElbowMeshRef} material={suitFabric} castShadow>
        <sphereGeometry args={[0.052, 12, 12]} />
      </mesh>
      {/* Left Forearm (Originates from Left Elbow) */}
      <mesh ref={leftForearmRef} material={suitFabric} castShadow>
        <cylinderGeometry args={[0.048, 0.044, 0.5, 16]} />
      </mesh>
      {/* Left Shirt Cuff at wrist */}
      <mesh ref={leftCuffRef} material={shirtWhite}>
        <cylinderGeometry args={[0.045, 0.047, 0.035, 16]} />
      </mesh>
      {/* Left Hand & Articulated Curved Fingers resting on keys */}
      <group ref={leftHandGroupRef} position={[-0.35, 2.18, -0.06]}>
        <mesh position={[0, 0.012, 0]} rotation={[0.08, 0.05, 0]} material={skinTone} castShadow>
          <boxGeometry args={[0.084, 0.028, 0.088]} />
        </mesh>
        {/* Thumb */}
        <mesh position={[0.044, -0.01, -0.01]} rotation={[0.2, 0.35, 0]} material={skinTone}>
          <capsuleGeometry args={[0.008, 0.034, 6, 8]} />
        </mesh>
        {/* 4 Arched Fingers pointing into keys */}
        <mesh position={[0.022, -0.012, 0.036]} rotation={[0.55, 0, 0]} material={skinTone}>
          <capsuleGeometry args={[0.0075, 0.042, 6, 8]} />
        </mesh>
        <mesh position={[0, -0.014, 0.04]} rotation={[0.55, 0, 0]} material={skinTone}>
          <capsuleGeometry args={[0.0075, 0.044, 6, 8]} />
        </mesh>
        <mesh position={[-0.022, -0.012, 0.036]} rotation={[0.55, 0, 0]} material={skinTone}>
          <capsuleGeometry args={[0.007, 0.042, 6, 8]} />
        </mesh>
        <mesh position={[-0.042, -0.01, 0.028]} rotation={[0.5, 0, 0]} material={skinTone}>
          <capsuleGeometry args={[0.0065, 0.036, 6, 8]} />
        </mesh>
      </group>

      {/* RIGHT ARM */}
      {/* Right Upper Arm (Originates from Right Shoulder) */}
      <mesh ref={rightUpperArmRef} material={suitFabric} castShadow>
        <cylinderGeometry args={[0.052, 0.048, 0.5, 16]} />
      </mesh>
      {/* Right Elbow Joint */}
      <mesh ref={rightElbowMeshRef} material={suitFabric} castShadow>
        <sphereGeometry args={[0.052, 12, 12]} />
      </mesh>
      {/* Right Forearm (Originates from Right Elbow) */}
      <mesh ref={rightForearmRef} material={suitFabric} castShadow>
        <cylinderGeometry args={[0.048, 0.044, 0.5, 16]} />
      </mesh>
      {/* Right Shirt Cuff at wrist */}
      <mesh ref={rightCuffRef} material={shirtWhite}>
        <cylinderGeometry args={[0.045, 0.047, 0.035, 16]} />
      </mesh>
      {/* Right Hand & Articulated Curved Fingers resting on keys */}
      <group ref={rightHandGroupRef} position={[0.35, 2.18, -0.06]}>
        <mesh position={[0, 0.012, 0]} rotation={[0.08, -0.05, 0]} material={skinTone} castShadow>
          <boxGeometry args={[0.084, 0.028, 0.088]} />
        </mesh>
        {/* Thumb */}
        <mesh position={[-0.044, -0.01, -0.01]} rotation={[0.2, -0.35, 0]} material={skinTone}>
          <capsuleGeometry args={[0.008, 0.034, 6, 8]} />
        </mesh>
        {/* 4 Arched Fingers pointing into keys */}
        <mesh position={[-0.022, -0.012, 0.036]} rotation={[0.55, 0, 0]} material={skinTone}>
          <capsuleGeometry args={[0.0075, 0.042, 6, 8]} />
        </mesh>
        <mesh position={[0, -0.014, 0.04]} rotation={[0.55, 0, 0]} material={skinTone}>
          <capsuleGeometry args={[0.0075, 0.044, 6, 8]} />
        </mesh>
        <mesh position={[0.022, -0.012, 0.036]} rotation={[0.55, 0, 0]} material={skinTone}>
          <capsuleGeometry args={[0.007, 0.042, 6, 8]} />
        </mesh>
        <mesh position={[0.042, -0.01, 0.028]} rotation={[0.5, 0, 0]} material={skinTone}>
          <capsuleGeometry args={[0.0065, 0.036, 6, 8]} />
        </mesh>
      </group>
    </group>
  );
};
