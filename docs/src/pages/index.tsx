import React, { useEffect, useState } from "react";
import { FiEdit3, FiGrid, FiSmile } from "react-icons/fi";
import ThreeScene from "../components/ThreeCanvas";
import Layout from "@theme/Layout";
import useBaseUrl from "@docusaurus/useBaseUrl";
import Link from "@docusaurus/Link";
import styles from "./index.module.css";
import { JSX } from "react";

export default function Home(): JSX.Element {
  useEffect(() => {
    const footer = document.querySelector(".footer");

    const handleScroll = () => {
      if (!footer) return;

      const scrolledToBottom =
        window.innerHeight + window.scrollY >= document.body.offsetHeight - 10;

      if (scrolledToBottom) {
        footer.classList.add("footer--visible");
      } else {
        footer.classList.remove("footer--visible");
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    const navbar = document.querySelector(".navbar");
    const onScroll = () => {
      if (!navbar) return;
      if (window.scrollY > 10) {
        navbar.classList.add("navbar--scrolled");
      } else {
        navbar.classList.remove("navbar--scrolled");
      }
    };
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <Layout title="Vision6D" description="Interactive 6D Pose Annotation Tool">
      <main>
        {/* Hero Section */}
        <section className={styles.hero}>
          <div className={styles.heroBackground}>
            <ThreeScene />
          </div>
          <div className={styles.heroContent}>
            <h1><a href="https://github.com/InteractiveGL/vision6D">Vision6D</a></h1>
            <p>Redefining Pose Annotations</p>
            <div className={styles.heroButtons}>
              <Link
                className={styles.buttonPrimary}
                to="https://github.com/InteractiveGL/vision6D/releases/tag/0.5.4/"
              >
                Download Now
              </Link>
              <Link className={styles.buttonSecondary} to="/docs">
                Get Started
              </Link>
            </div>
          </div>
        </section>

        {/* Video Section */}
        <section className={styles.videoSection}>
          <div className={styles.videoBackground}>
            <div className={styles.videoWrapper}>
              <video width="100%" autoPlay muted loop playsInline>
                <source
                  src={useBaseUrl("/videos/vision6D_demo.mp4")}
                  type="video/mp4"
                />
                Your browser does not support the video tag.
              </video>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className={styles.features}>
          <div className={styles.feature}>
            <div className={styles.featureIcon}>
              <FiEdit3 />
            </div>
            <h2>Powerful Annotation Tools</h2>
            <p>
              Efficiently label 6D object poses with intuitive 2D-3D interaction
              tools.
            </p>
          </div>

          <div className={styles.feature}>
            <div className={styles.featureIcon}>
              <FiGrid />
            </div>
            <h2>Intraoperative Integration</h2>
            <p>Real-time visualization for 2D-to-3D registration.</p>
          </div>

          <div className={styles.feature}>
            <div className={styles.featureIcon}>
              <FiSmile />
            </div>
            <h2>User Friendly</h2>
            <p>
              Built for rapid 6D pose registration and custom vision system
              integration.
            </p>
          </div>
        </section>

        {/* Split feature section */}
        <section className={styles.splitFeature}>
          <div className={styles.textContent}>
            <h2>Real-time Pose Visualization</h2>
            <p>
              Vision6D provides a seamless experience for annotating 6D poses
              with real-time feedback. The intuitive interface allows users to
              interact with 2D and 3D views, making it easy to adjust and refine
              annotations.
            </p>
            {/* <a
              className={styles.link}
              href="https://github.com/features/copilot"
            >
              Build with Copilot Edits
            </a> */}
          </div>

          <div className={styles.imageContent}>
            <img
              src={useBaseUrl("/img/sample.png")}
              alt="Copilot multi-file edits preview"
            />
          </div>
        </section>

        <section className={styles.splitFeature}>
          <div className={styles.textContent}>
            <h2>Depth Estimation</h2>
            <p>
              Besides of 6D pose annotations, Vision6D also supports depth
              estimation. This feature allows users to determine depth information
              alongside 6D pose data, supporting a broader usability of the interactive
              annotations.
            </p>
          </div>

          <div className={styles.imageContent}>
            <img
              src={useBaseUrl("/img/sample.png")}
              alt="Copilot multi-file edits preview"
            />
          </div>
        </section>

        <section className={styles.splitFeature}>
          <div className={styles.textContent}>
            <h2>Single-Object Annotation</h2>
            <p>
              Vision6D offers an effective process for annotating single objects in 6D space. 
              Users can easily select and interact with the targeting objects, ensuring accurate annotations with minimal effort.
            </p>
          </div>

          <div className={styles.imageContent}>
            <img
              src={useBaseUrl("/img/sample.png")}
              alt="Copilot multi-file edits preview"
            />
          </div>
        </section>

        <section className={styles.splitFeature}>
          <div className={styles.textContent}>
            <h2>Multi-Object Annotation</h2>
            <p>
              Vision6D is also a robust tool for annotating multiple objects within a scene. 
              Users can easily link the multiple objects together and manipulate them, providing comprehensive and accurate annotations among multiple objects shown in the scene.
            </p>
            </div>

          <div className={styles.imageContent}>
            <img
              src={useBaseUrl("/img/sample.png")}
              alt="Copilot multi-file edits preview"
            />
          </div>
        </section>

        <section className={styles.splitFeature}>
          <div className={styles.textContent}>
            <h2>PnP Registration</h2>
            <p>
              Vision6D supports PnP (Perspective-n-Point) registration, 
              allowing users to easily register 2D images with 3D models by simple clicking. 
              This feature will need a minimal of six pair of corresponding 2D and 3D points.
            </p>
            </div>

          <div className={styles.imageContent}>
            <img
              src={useBaseUrl("/img/sample.png")}
              alt="Copilot multi-file edits preview"
            />
          </div>
        </section>

        <section className={styles.splitFeature}>
          <div className={styles.textContent}>
            <h2>Mask Overlay</h2>
            <p>
              Vision6D provides a mask overlay feature that allows users to visualize the segmentation masks of 3D objects in the 2D scene.
              This feature can be helpful in the annotation process by providing a clear visual representation of object boundaries, and it can be potentially beneficial for downstream tasks such as object detection and segmentation.
            </p>
          </div>

          <div className={styles.imageContent}>
            <img
              src={useBaseUrl("/img/sample.png")}
              alt="Copilot multi-file edits preview"
            />
          </div>
        </section>

        {/* Grid Section */}
        <section className={styles.gridSection}>
          <div className={styles.grid}>
            <Link to="/docs" className={styles.card}>
              <h3>Getting Started</h3>
              <p>Quick setup instructions and requirements.</p>
            </Link>
            <Link to="/docs/Download" className={styles.card}>
              <h3>Download</h3>
              <p>Detailed download instructions of Vision6D.</p>
            </Link>
            <Link to="/blog" className={styles.card}>
              <h3>Latest Blog</h3>
              <p>Updated regularly with the latest news and tutorials.</p>
            </Link>
            <Link to="/faq" className={styles.card}>
              <h3>FAQ</h3>
              <p>Answers to frequently asked questions.</p>
            </Link>
          </div>
        </section>
      </main>
    </Layout>
  );
}
