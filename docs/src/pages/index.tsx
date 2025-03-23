import React, { useEffect, useState } from "react";
import { FiEdit3, FiGrid, FiSmile } from "react-icons/fi";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import styles from "./index.module.css";
import { JSX } from "react";

export default function Home(): JSX.Element {
  const [osDownloadLink, setOsDownloadLink] = useState("/docs/download");
  const [osName, setOsName] = useState("your OS");

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

  useEffect(() => {
    const platform = navigator.platform.toLowerCase();
    if (platform.includes("win")) {
      setOsDownloadLink("/downloads/vision6d-windows.exe");
      setOsName("Windows");
    } else if (platform.includes("mac") || platform.includes("darwin")) {
      setOsDownloadLink(
        "https://github.com/InteractiveGL/vision6D/releases/tag/0.5.3/"
      );
      setOsName("macOS");
    } else if (platform.includes("linux")) {
      setOsDownloadLink("/downloads/vision6d-linux.AppImage");
      setOsName("Linux");
    }
  }, []);

  return (
    <Layout title="Vision6D" description="Interactive 6D Pose Annotation Tool">
      <main>
        {/* Hero Section */}
        <section className={styles.hero}>
          <div className={styles.heroContent}>
            <h1>Vision6D</h1>
            <p>Redefine the Pose Annotation.</p>
            <div className={styles.heroButtons}>
              <Link className={styles.buttonPrimary} to={osDownloadLink}>
                Download for {osName}
              </Link>
              <Link
                className={styles.buttonSecondary}
                to="/docs/getting-started"
              >
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
                <source src="/videos/vision6d_demo.mp4" type="video/mp4" />
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
              Describe what you want to build in natural language, and Copilot
              Edits takes care of the rest. Copilot Edits makes changes across
              files in your codebase with a UI designed for rapid iteration.
              It's the fastest way to add new functionality to your apps.
            </p>
            <a
              className={styles.link}
              href="https://github.com/features/copilot"
            >
              Build with Copilot Edits
            </a>
          </div>

          <div className={styles.imageContent}>
            <img src="/img/sample.png" alt="Copilot multi-file edits preview" />
          </div>
        </section>

        <section className={styles.splitFeature}>
          <div className={styles.textContent}>
            <h2>Depth Estimation</h2>
            <p>
              Describe what you want to build in natural language, and Copilot
              Edits takes care of the rest. Copilot Edits makes changes across
              files in your codebase with a UI designed for rapid iteration.
              It's the fastest way to add new functionality to your apps.
            </p>
            <a
              className={styles.link}
              href="https://github.com/features/copilot"
            >
              Build with Copilot Edits
            </a>
          </div>

          <div className={styles.imageContent}>
            <img src="/img/sample.png" alt="Copilot multi-file edits preview" />
          </div>
        </section>

        <section className={styles.splitFeature}>
          <div className={styles.textContent}>
            <h2>Single-Object Annotation</h2>
            <p>
              Describe what you want to build in natural language, and Copilot
              Edits takes care of the rest. Copilot Edits makes changes across
              files in your codebase with a UI designed for rapid iteration.
              It's the fastest way to add new functionality to your apps.
            </p>
            <a
              className={styles.link}
              href="https://github.com/features/copilot"
            >
              Build with Copilot Edits
            </a>
          </div>

          <div className={styles.imageContent}>
            <img src="/img/sample.png" alt="Copilot multi-file edits preview" />
          </div>
        </section>

        <section className={styles.splitFeature}>
          <div className={styles.textContent}>
            <h2>Multi-Object Annotation</h2>
            <p>
              Describe what you want to build in natural language, and Copilot
              Edits takes care of the rest. Copilot Edits makes changes across
              files in your codebase with a UI designed for rapid iteration.
              It's the fastest way to add new functionality to your apps.
            </p>
            <a
              className={styles.link}
              href="https://github.com/features/copilot"
            >
              Build with Copilot Edits
            </a>
          </div>

          <div className={styles.imageContent}>
            <img src="/img/sample.png" alt="Copilot multi-file edits preview" />
          </div>
        </section>

        <section className={styles.splitFeature}>
          <div className={styles.textContent}>
            <h2>PnP Registration</h2>
            <p>
              Describe what you want to build in natural language, and Copilot
              Edits takes care of the rest. Copilot Edits makes changes across
              files in your codebase with a UI designed for rapid iteration.
              It's the fastest way to add new functionality to your apps.
            </p>
            <a
              className={styles.link}
              href="https://github.com/features/copilot"
            >
              Build with Copilot Edits
            </a>
          </div>

          <div className={styles.imageContent}>
            <img src="/img/sample.png" alt="Copilot multi-file edits preview" />
          </div>
        </section>

        <section className={styles.splitFeature}>
          <div className={styles.textContent}>
            <h2>Mask Overlay</h2>
            <p>
              Describe what you want to build in natural language, and Copilot
              Edits takes care of the rest. Copilot Edits makes changes across
              files in your codebase with a UI designed for rapid iteration.
              It's the fastest way to add new functionality to your apps.
            </p>
            <a
              className={styles.link}
              href="https://github.com/features/copilot"
            >
              Build with Copilot Edits
            </a>
          </div>

          <div className={styles.imageContent}>
            <img src="/img/sample.png" alt="Copilot multi-file edits preview" />
          </div>
        </section>

        {/* Grid Section */}
        <section className={styles.gridSection}>
          <div className={styles.grid}>
            <Link to="/docs/getting-started" className={styles.card}>
              <h3>Getting Started</h3>
              <p>Quick setup instructions and requirements.</p>
            </Link>
            <Link to="/docs/user-guide" className={styles.card}>
              <h3>User Guide</h3>
              <p>Detailed walkthrough of Vision6D interface and tools.</p>
            </Link>
            <Link to="/docs/youtube_toturials" className={styles.card}>
              <h3>YouTube Toturials</h3>
              <p>API reference, custom extensions, and build instructions.</p>
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
