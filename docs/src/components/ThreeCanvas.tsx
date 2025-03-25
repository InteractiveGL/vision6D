import React, { useEffect, useRef } from "react";
import * as THREE from "three";

export default function ThreeScene() {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const width = window.innerWidth;
    const height = 300; // Hero-height
    const aspect = width / height;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 2000);
    camera.position.z = 800;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current?.appendChild(renderer.domElement);

    // Light
    const light = new THREE.PointLight(0xffffff, 1);
    light.position.set(100, 100, 100);
    scene.add(light);
    scene.add(new THREE.AmbientLight(0x444444));

    // Orange
    const parentMesh = new THREE.Mesh(
      new THREE.SphereGeometry(100, 32, 16),
      new THREE.MeshBasicMaterial({ color: 0xffaa55, wireframe: true })
    );
    scene.add(parentMesh);

    // Green
    const childMesh = new THREE.Mesh(
      new THREE.SphereGeometry(100, 32, 16),
      new THREE.MeshBasicMaterial({ color: 0xffa5e8, wireframe: true })
    );
    childMesh.position.z = 50;
    parentMesh.add(childMesh);

    // Blue
    const floatingDot = new THREE.Mesh(
      new THREE.SphereGeometry(100, 32, 16),
      new THREE.MeshBasicMaterial({ color: 0x0088ff, wireframe: true })
    );
    floatingDot.position.z = 50;
    scene.add(floatingDot);

    // Add three more colored spheres
    const mesh4 = new THREE.Mesh(
      new THREE.SphereGeometry(100, 32, 16),
      new THREE.MeshBasicMaterial({ color: 0xffd700, wireframe: true })
    );
    mesh4.position.set(-300, 0, 0);
    scene.add(mesh4);

    const mesh5 = new THREE.Mesh(
      new THREE.SphereGeometry(100, 32, 16),
      new THREE.MeshBasicMaterial({ color: 0x8bc34a, wireframe: true })
    );
    mesh5.position.set(0, -200, 0);
    scene.add(mesh5);

    const mesh6 = new THREE.Mesh(
      new THREE.SphereGeometry(100, 32, 16),
      new THREE.MeshBasicMaterial({ color: 0xde9eea, wireframe: true })
    );
    mesh6.position.set(300, 0, 0);
    scene.add(mesh6);

    // Animate
    const animate = () => {
      const t = Date.now() * 0.0008;

      // Parent — large 3D orbit
      parentMesh.position.set(
        800 * Math.cos(t * 0.8),
        600 * Math.sin(t * 1.1),
        400 * Math.sin(t * 0.9)
      );

      // Child — independent, off‑centre orbit
      childMesh.position.set(
        600 * Math.cos(t * 1.4) + 200,
        500 * Math.sin(t * 1.7) - 150,
        300 * Math.cos(t * 1.2)
      );

      // Floating dot — big bounce + drift
      floatingDot.position.set(
        700 * Math.cos(t * 1.3) - 250,
        700 * Math.sin(t * 1.5),
        200 * Math.sin(t * 2.0)
      );
      floatingDot.rotation.y += 0.03;

      // New three — wide, randomized orbits
      mesh4.position.set(
        900 * Math.cos(t * 1.1) - 300,
        800 * Math.sin(t * 1.3) + 200,
        250 * Math.sin(t * 1.7)
      );

      mesh5.position.set(
        750 * Math.cos(t * 1.2) + 300,
        600 * Math.sin(t * 1.4) - 200,
        350 * Math.cos(t * 1.6)
      );

      mesh6.position.set(
        850 * Math.cos(t * 1.0) - 100,
        700 * Math.sin(t * 1.8) + 100,
        450 * Math.sin(t * 1.1)
      );

      parentMesh.rotation.x += 0.01;
      parentMesh.rotation.y += 0.015;

      camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    };
    animate();

    // Resize
    const handleResize = () => {
      const newWidth = window.innerWidth;
      const newAspect = newWidth / height;
      camera.aspect = newAspect;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, height);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      mountRef.current?.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  return <div ref={mountRef} style={{ width: "100%", height: "300px" }} />;
}
