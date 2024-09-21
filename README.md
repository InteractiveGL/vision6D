<h1 align="center">
  <br>
  <a href=""><img src="./teaser/vision6D_logo.png" alt="Vision6D" width="300"></a>
  <br>
  VISION6D
  <br>
</h1>

<h4 align="center">VISION6D: 3D-to-2D visualization and annotation tool for 6D pose estimation desktop app. This python-based application is designed to work on Windows and Linux (Ubuntu-tested).</h4>

<p align="center">
  <a href="https://github.com/InteractiveGL/vision6D/releases">
    <img src="https://img.shields.io/github/v/release/InteractiveGL/vision6D"
         alt="github_release">
  </a>
  <!-- <a href="https://github.com/InteractiveGL/vision6D/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/InteractiveGL/vision6D"
         alt="github_license">
  </a> -->
  <a href="https://github.com/InteractiveGL/vision6D/blob/main/LICENSE">
    <img src="https://img.shields.io/github/last-commit/InteractiveGL/vision6D/main"
         alt="github_commit">
  </a>
  <a href="https://github.com/InteractiveGL/vision6D/">
    <img src="https://img.shields.io/github/downloads/InteractiveGL/vision6D/total"
         alt="github_downloads">
  </a>
  <a href="https://github.com/InteractiveGL/vision6D/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/InteractiveGL/vision6D"
         alt="github_contributes">
  </a>
  <a href="https://github.com/InteractiveGL/vision6D">
    <img src="https://img.shields.io/github/stars/InteractiveGL/vision6D"
         alt="github_stars">
  </a>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#examples">Examples</a> •
  <a href="#download">Download</a> •
  <a href="#emailware">Emailware</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

![screenshot](./teaser/freehand_registration.gif)

## Key Features

* LivePreview - Make changes, See changes
  - Instantly see what your pose annotation in Vision6D as you move the 3D objects!
* Provide built-in [NOCS](https://arxiv.org/abs/1901.02970) color representation for the 3D meshes
  - color the meshes with NOCS.
* Load the textures for the 3D meshes
  - color the meshes with their own textures.
* Segmentation Mask/Bounding Box Drawing
  - create a segmentation mask in Vision6D on top of the provided 2D image.
* Real-time rendering results
  - renders the annotated results.
* Cross platform
  - Windows and Linux (Ubuntu-tested) ready (highly recommend to use with a mouse).

## How To Use

To clone and run this application, you'll need [Git](https://git-scm.com) and [Node.js](https://nodejs.org/en/download/) (which comes with [npm](http://npmjs.com)) installed on your computer. From your command line:

```bash
# (Optional) Create a conda environment
$ conda create -n vision6D python=3.10

# Clone this repository
$ git clone https://github.com/InteractiveGL/vision6D.git

# Go into the repository
$ cd vision6D

# Install dependencies
$ pip install .

# Run the app
$ Vision6D
```

## Examples
Note that when fisrt load the application, it may take some time. Once it load successfully, the interactive experience will be smooth.

**PnP resgitration of the benchvise**
<p float="left">
  <img src="./teaser/pnp1.png" alt="screenshot 1" width="45%" style="margin-right: 20px;" />
  <img src="./teaser/pnp2.png" alt="screenshot 2" width="45%" />
</p>

**Set a ground-truth pose for visualization of the benchvise (ground-truth pose is obtained from the public 6D pose dataset Linemod)**
<p float="left">
  <img src="./teaser/set_pose1.png" alt="screenshot 1" width="45%" style="margin-right: 20px;" />
  <img src="./teaser/set_pose2.png" alt="screenshot 2" width="45%" />
</p>

**Free-hand registration of the benchvise**
<p>
  <img src="./teaser/freehand_registration.gif" alt="screenshot" width="92%" />
</p>

**Draw a segmentation mask on the duck in this scene**
<p float="left">
  <img src="./teaser/draw_mask1.png" alt="screenshot 1" width="45%" style="margin-right: 20px;" />
  <img src="./teaser/draw_mask2.png" alt="screenshot 2" width="45%" />
</p>

**Draw a bounding box around the duck in this scene**
<p float="left">
  <img src="./teaser/draw_bbox_1.png" alt="screenshot 1" width="45%" style="margin-right: 20px;" />
  <img src="./teaser/draw_bbox_2.png" alt="screenshot 2" width="45%" />
</p>

**Render the benchwise mesh**
<p float="left">
  <img src="./teaser/render_mesh_1.png" alt="screenshot 1" width="45%" style="margin-right: 20px;" />
  <img src="./teaser/render_mesh_2.png" alt="screenshot 2" width="45%" />
</p>

## Download

You can [download](https://github.com/InteractiveGL/vision6D/releases/tag/0.3.9) the latest installable version of Vision6D for Windows, macOS and Linux.

## Emailware

Vision6D is an [emailware](https://en.wiktionary.org/wiki/emailware). Meaning, if you liked using this app or it has helped you in any way, I'd like you send me an email at <yike.zhang@vanderbilt.edu> about anything you'd want to say about this software. I'd really appreciate it! Alternatively, you can also submit an issue regarding using this software, I'll answer it as soon as I see it!

## Credits

This software uses the following open source packages:

- [Pyvista](https://docs.pyvista.org)
- [VTK](https://vtk.org/)
- [PyQt5](https://www.riverbankcomputing.com/static/Docs/PyQt5/)
- [pyvistaqt](https://github.com/pyvista/pyvistaqt)
- [NumPy](https://numpy.org/)

## License

GNU

---