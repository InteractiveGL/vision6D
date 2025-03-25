import React from "react";
import Layout from "@theme/Layout";
import styles from "./index.module.css";
import Link from "@docusaurus/Link";

export default function DocsHome() {
  return (
    <Layout title="Docs" description="Vision6D documentation overview">
      <main className={styles.docsHome}>
        <h1>Vision6D Documentation</h1>
        <p className={styles.subheading}>
          Learn how to install, use, and develop with Vision6D.
        </p>

        {/* Section 1 */}
        <h2 className={styles.sectionTitle}>Getting Started</h2>
        <div className={styles.grid}>
          <Card
            title="Download Vision6D"
            description="Download the tool for Windows, macOS, or Linux."
            to="/docs/getting-started"
            icon="🛠️"
          />
          <Card
            title="Getting Started"
            description="Step-by-step setup instructions."
            to="/docs/getting-started"
            icon="🚀"
          />
          <Card
            title="Video Tutorials"
            description="Learn by watching walkthroughs."
            to="https://youtube.com"
            icon="🎥"
          />
        </div>

        {/* Section 2 */}
        <h2 className={styles.sectionTitle}>Core Features</h2>
        <div className={styles.grid}>
          <Card
            title="2D-3D Annotation"
            description="Label poses with interactive tools."
            to="/docs/user-guide"
            icon="✏️"
          />
          <Card
            title="Developer API"
            description="Extend Vision6D with custom logic."
            to="/docs/developer-guide"
            icon="🧩"
          />
          <Card
            title="FAQ"
            description="Find answers to common questions."
            to="docs/faq"
            icon="❓"
          />
        </div>
      </main>
    </Layout>
  );
}

function Card({ title, description, to, icon }) {
  return (
    <Link className={styles.card} to={to}>
      <div className={styles.cardIcon}>{icon}</div>
      <div className={styles.cardTitle}>{title}</div>
      <p className={styles.cardDesc}>{description}</p>
    </Link>
  );
}
