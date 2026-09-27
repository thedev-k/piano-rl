import React, { useMemo } from 'react';
import * as THREE from 'three';

export const ConcertStage: React.FC = () => {
  // 1. Procedural Wood Plank Stage Flooring Texture (Warm walnut / honey tone with visible planks)
  const woodTexture = useMemo(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 1024;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    // Base walnut warm tone (#6B4226)
    ctx.fillStyle = '#6b4226';
    ctx.fillRect(0, 0, 1024, 1024);

    // Plank lines and wood grain simulation
    const plankHeight = 85;
    const numPlanks = Math.ceil(1024 / plankHeight);

    for (let i = 0; i < numPlanks; i++) {
      const y = i * plankHeight;

      // Natural plank-to-plank warmth variation (#6b4226 to #8b5a2b)
      const r = Math.floor(107 + (i % 3) * 14 + (Math.sin(i * 1.7) * 10));
      const g = Math.floor(66 + (i % 3) * 9 + (Math.sin(i * 1.7) * 7));
      const b = Math.floor(38 + (i % 3) * 5 + (Math.sin(i * 1.7) * 4));
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(0, y, 1024, plankHeight);

      // Fine grain streaks in warm honey
      ctx.fillStyle = 'rgba(168, 112, 58, 0.22)';
      for (let gIdx = 0; gIdx < 8; gIdx++) {
        const gy = y + 8 + (gIdx * 9);
        ctx.fillRect(0, gy, 1024, 1.5);
      }

      // Darker grain accent lines
      ctx.fillStyle = 'rgba(64, 34, 16, 0.28)';
      ctx.fillRect(0, y + plankHeight * 0.45, 1024, 1);
      ctx.fillRect(0, y + plankHeight * 0.75, 1024, 1);

      // Distinct plank seam groove
      ctx.fillStyle = 'rgba(28, 14, 8, 0.95)';
      ctx.fillRect(0, y, 1024, 3); // horizontal joint

      // Staggered plank butt joints
      const butt1 = (i * 340) % 1024;
      const butt2 = (i * 340 + 560) % 1024;
      ctx.fillRect(butt1, y, 3, plankHeight);
      ctx.fillRect(butt2, y, 3, plankHeight);
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(4, 4);
    return tex;
  }, []);

  // Polished Stage Wood Material (Warm walnut/honey tone with specular sheen)
  const floorMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        map: woodTexture || undefined,
        color: new THREE.Color('#784c2e'),
        roughness: 0.22,
        metalness: 0.12,
      }),
    [woodTexture]
  );

  // Soft contact reflection / shadow puddle under the piano
  const contactShadowTex = useMemo(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    const grad = ctx.createRadialGradient(256, 256, 20, 256, 256, 240);
    grad.addColorStop(0, 'rgba(10, 6, 8, 0.75)');
    grad.addColorStop(0.5, 'rgba(15, 8, 10, 0.4)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 512, 512);

    const tex = new THREE.CanvasTexture(canvas);
    return tex;
  }, []);

  // 2. Folded Velvet Curtains Geometry (Deep 3D drape folds)
  const createCurtainGeometry = (width: number, height: number, foldFrequency: number, foldDepth: number) => {
    const segmentsX = Math.round(width * 9);
    const segmentsY = 16;
    const geo = new THREE.PlaneGeometry(width, height, segmentsX, segmentsY);
    const pos = geo.attributes.position;

    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      // Sinusoidal fold displacement along Z for realistic velvet draping
      const zOffset =
        Math.sin(x * foldFrequency) * foldDepth +
        Math.sin(x * foldFrequency * 2.05) * (foldDepth * 0.38) +
        Math.cos(x * foldFrequency * 0.5) * (foldDepth * 0.15);
      pos.setZ(i, zOffset);
    }
    geo.computeVertexNormals();
    return geo;
  };

  const backdropCurtainGeo = useMemo(() => createCurtainGeometry(30, 16, 3.2, 0.35), []);
  const wingCurtainGeo = useMemo(() => createCurtainGeometry(18, 16, 3.2, 0.35), []);

  // Velvet Curtain Material: Rich deep burgundy-red (~#5C1420 in shadow, ~#8B1E2E lit)
  const curtainMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#941a2a'),
        roughness: 0.68,
        metalness: 0.05,
      }),
    []
  );

  // 3. Theatrical Stage Atmosphere Light Beam (Warm golden follow-spot cone)
  const spotBeamGeo = useMemo(() => {
    const geo = new THREE.CylinderGeometry(0.35, 3.8, 11.0, 32, 1, true);
    return geo;
  }, []);

  const spotBeamMaterial = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: new THREE.Color('#f2c572'),
        transparent: true,
        opacity: 0.045,
        side: THREE.DoubleSide,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    []
  );

  return (
    <group>
      {/* Polished Stage Wood Floor with visible plank seams and sheen */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0, 0]}
        material={floorMaterial}
        receiveShadow
      >
        <planeGeometry args={[44, 44]} />
      </mesh>

      {/* Grounded contact reflection & shadow under piano and bench */}
      {contactShadowTex && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.005, 0.9]}>
          <planeGeometry args={[5.8, 5.5]} />
          <meshBasicMaterial
            map={contactShadowTex}
            transparent
            opacity={0.85}
            depthWrite={false}
          />
        </mesh>
      )}

      {/* STAGE CURTAINS (Burgundy Velvet with physical 3D drape folds) */}
      {/* 1. Main Stage Backdrop Curtain — tall wall of red directly behind the piano & performer */}
      <mesh
        position={[3.8, 7.0, 1.0]}
        rotation={[0, -Math.PI / 2, 0]}
        geometry={backdropCurtainGeo}
        material={curtainMaterial}
        castShadow
        receiveShadow
      />

      {/* 2. Left Stage Wing Curtain — framing behind the pianist/bench (stage left) */}
      <mesh
        position={[-0.5, 7.0, -4.2]}
        rotation={[0, 0.32, 0]}
        geometry={wingCurtainGeo}
        material={curtainMaterial}
        castShadow
        receiveShadow
      />

      {/* 3. Right Stage Wing Curtain — framing beyond the piano tail (stage right) */}
      <mesh
        position={[0.5, 7.0, 6.2]}
        rotation={[0, Math.PI - 0.32, 0]}
        geometry={wingCurtainGeo}
        material={curtainMaterial}
        castShadow
        receiveShadow
      />

      {/* Velvet Curtain Valance / Pelmet Header along top */}
      <mesh position={[3.6, 12.0, 1.0]} rotation={[0, -Math.PI / 2, 0]}>
        <boxGeometry args={[32, 2.2, 0.4]} />
        <meshStandardMaterial
          color="#5c1420"
          roughness={0.8}
        />
      </mesh>

      {/* Theatrical Stage Atmosphere Light Beam (Angled follow-spot cone) */}
      <group position={[-1.4, 7.2, 0.8]} rotation={[0.12, 0.12, -0.22]}>
        <mesh geometry={spotBeamGeo} material={spotBeamMaterial} />
      </group>
    </group>
  );
};
