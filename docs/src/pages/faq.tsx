import React from "react";
import Layout from "@theme/Layout";
import { JSX } from "react";

export default function FAQ(): JSX.Element {
  return (
    <Layout title="FAQ" description="Frequently Asked Questions">
      <main
        style={{ padding: "2rem 1rem", maxWidth: "800px", margin: "0 auto" }}
      >
        <h1>Frequently Asked Questions</h1>
        <h2>What is Vision6D?</h2>
        <p>
          <a href="https://github.com/InteractiveGL/vision6D">Vision6D</a> is an interactive annotation GUI for 2D-to-3D 6D pose
          annotation.
        </p>

        <h2>How do I download it?</h2>
        <p>
          Use the <a href="docs/download">Download</a> link to get the installer
          for your OS.
        </p>

        <h2>Can I customize it?</h2>
        <p>
          Yes, it's open-source online. See the{" "}
          <a href="https://github.com/InteractiveGL/vision6D">GitHub repo</a>.
        </p>
      </main>
    </Layout>
  );
}
