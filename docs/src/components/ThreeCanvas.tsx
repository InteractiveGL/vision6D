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
      new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true })
    );
    childMesh.position.y = 50;
    parentMesh.add(childMesh);

    // Blue
    const floatingDot = new THREE.Mesh(
      new THREE.SphereGeometry(100, 32, 16),
      new THREE.MeshBasicMaterial({ color: 0x0088ff, wireframe: true })
    );
    floatingDot.position.z = 50;
    scene.add(floatingDot);

    // Animate
    const animate = () => {
      const time = Date.now() * 0.001;

      // 🌀 Dramatic oscillation for parent mesh
      parentMesh.position.x = 400 * Math.cos(time * 1.2);
      parentMesh.position.y = 300 * Math.sin(time * 1.5);
      parentMesh.position.z = 200 * Math.sin(time * 1.1);

      // 🛰️ Orbiting motion for child
      childMesh.position.x = 100 * Math.cos(time * 1);
      childMesh.position.y = 200 * Math.sin(time * 1);
      childMesh.position.z = 300 * Math.sin(time * 1);

      // ✨ Vertical bounce for floating dot
      floatingDot.position.y = 350 * Math.abs(Math.sin(time * 2));
      floatingDot.position.x = 500 * Math.cos(time * 1.5);
      floatingDot.position.z = 100 * Math.sin(time * 3);
      floatingDot.rotation.y += 0.05;

      // Optional: Add a rotation to the parent mesh
      parentMesh.rotation.x += 0.01;
      parentMesh.rotation.y += 0.015;

      // Camera tracking
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
