import React, { useMemo } from 'react';
import * as THREE from 'three';

export const PianoBench: React.FC = () => {
  const benchWidth = 0.95;
  const benchDepth = 0.44;
  const benchSeatHeight = 1.42; // Ergonomic height relative to key top at 2.15
  const cushionThickness = 0.12;
  const legHeight = benchSeatHeight - cushionThickness;

  const woodMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#0e0f13'),
        roughness: 0.22,
        metalness: 0.15,
      }),
    []
  );

  const tuftedLeatherMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#16181d'),
        roughness: 0.45,
        metalness: 0.05,
      }),
    []
  );

  const knobBrassMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#cfa04c'),
        roughness: 0.3,
        metalness: 0.85,
      }),
    []
  );

  return (
    // Bench sits directly in front of the keyboard center at Z = -0.78
    <group position={[-0.12, 0, -0.78]}>
      {/* Tufted Leather Cushion Top */}
      <mesh
        position={[0, benchSeatHeight - cushionThickness / 2, 0]}
        material={tuftedLeatherMaterial}
        castShadow
        receiveShadow
      >
        <boxGeometry args={[benchWidth, cushionThickness, benchDepth]} />
      </mesh>

      {/* Button-tufted indentations detail on seat top */}
      {[-0.32, -0.16, 0, 0.16, 0.32].map((x, i) => (
        <group key={i}>
          <mesh position={[x, benchSeatHeight + 0.002, -0.1]} material={woodMaterial}>
            <sphereGeometry args={[0.014, 8, 8]} />
          </mesh>
          <mesh position={[x, benchSeatHeight + 0.002, 0.1]} material={woodMaterial}>
            <sphereGeometry args={[0.014, 8, 8]} />
          </mesh>
        </group>
      ))}

      {/* Wooden Apron/Casing under cushion */}
      <mesh position={[0, benchSeatHeight - cushionThickness - 0.04, 0]} material={woodMaterial}>
        <boxGeometry args={[benchWidth - 0.04, 0.08, benchDepth - 0.04]} />
      </mesh>

      {/* Adjustment knobs on left & right sides */}
      <mesh
        position={[-benchWidth / 2 - 0.025, benchSeatHeight - cushionThickness - 0.04, 0]}
        rotation={[0, 0, Math.PI / 2]}
        material={knobBrassMaterial}
      >
        <cylinderGeometry args={[0.032, 0.032, 0.05, 16]} />
      </mesh>
      <mesh
        position={[benchWidth / 2 + 0.025, benchSeatHeight - cushionThickness - 0.04, 0]}
        rotation={[0, 0, Math.PI / 2]}
        material={knobBrassMaterial}
      >
        <cylinderGeometry args={[0.032, 0.032, 0.05, 16]} />
      </mesh>

      {/* Four Turned Bench Legs */}
      {/* Front Left */}
      <mesh position={[-benchWidth / 2 + 0.08, legHeight / 2, -benchDepth / 2 + 0.08]} material={woodMaterial} castShadow>
        <cylinderGeometry args={[0.038, 0.025, legHeight, 12]} />
      </mesh>
      {/* Front Right */}
      <mesh position={[benchWidth / 2 - 0.08, legHeight / 2, -benchDepth / 2 + 0.08]} material={woodMaterial} castShadow>
        <cylinderGeometry args={[0.038, 0.025, legHeight, 12]} />
      </mesh>
      {/* Back Left */}
      <mesh position={[-benchWidth / 2 + 0.08, legHeight / 2, benchDepth / 2 - 0.08]} material={woodMaterial} castShadow>
        <cylinderGeometry args={[0.038, 0.025, legHeight, 12]} />
      </mesh>
      {/* Back Right */}
      <mesh position={[benchWidth / 2 - 0.08, legHeight / 2, benchDepth / 2 - 0.08]} material={woodMaterial} castShadow>
        <cylinderGeometry args={[0.038, 0.025, legHeight, 12]} />
      </mesh>
    </group>
  );
};
