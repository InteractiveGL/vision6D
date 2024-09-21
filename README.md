<h1 align="center">
  <br>
  <a href=""><img src="https://github.com/user-attachments/assets/77a97754-8a34-44e9-b22a-a506842699a2" alt="Vision6D" width="300"></a>
  <br>
  VISION6D
  <br>
</h1>

<h4 align="center">3D-to-2D visualization and annotation desktop app for 6D pose estimation related tasks. This python-based application is designed to work on Windows and Linux (Ubuntu-tested).</h4>

<p align="center">
  <a href="https://pypi.org/project/vision6D/">
    <img src="https://img.shields.io/pypi/v/vision6D"
         alt="pypi_release">
  </a>
  <a href="https://github.com/InteractiveGL/vision6D/releases">
    <img src="https://img.shields.io/github/v/release/InteractiveGL/vision6D"
         alt="github_release">
  </a>
  <a href="https://github.com/InteractiveGL/vision6D/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/InteractiveGL/vision6D"
         alt="github_license">
  </a>
  <a href="https://github.com/InteractiveGL/vision6D/">
    <img src="https://img.shields.io/github/downloads/InteractiveGL/vision6D/total"
         alt="github_downloads">
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

![screenshot](https://github.com/user-attachments/assets/a3697ff5-1270-4da6-9273-a1f2ae1e19be)


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

To run this application, you'll need [Git](https://git-scm.com), [Python](https://www.python.org/), and [Miniconda](https://docs.anaconda.com/miniconda/) (optional) installed on your computer. From your command line:

**Vision6D can be directly installed from PyPi**
```bash
$ pip install vision6D
```

**Another way to use this software is to clone from this repository**
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
  <img src="https://github.com/user-attachments/assets/e1968f74-dcdd-4c01-a3dc-69c54b97dc84" alt="screenshot 1" width="45%" style="margin-right: 20px;" />
  <img src="https://github.com/user-attachments/assets/2fb62058-b5d1-4adf-851d-036065735fcd" alt="screenshot 2" width="45%" />
</p>

**Set a ground-truth pose for visualization of the benchvise (ground-truth pose is obtained from the public 6D pose dataset Linemod)**
<p float="left">
  <img src="https://github.com/user-attachments/assets/da9c1eb8-abbd-4045-bc54-200bef15452d" alt="screenshot 1" width="45%" style="margin-right: 20px;" />
  <img src="https://github.com/user-attachments/assets/9f6fed68-0c45-4969-b0e4-62bfc6efc255" alt="screenshot 2" width="45%" />
</p>

**Free-hand registration of the benchvise**
<p>
  <img src="https://github.com/user-attachments/assets/0b5468bb-3fdc-4932-8821-da253d33d722" alt="screenshot" width="92%" />
</p>

**Draw a segmentation mask on the duck in this scene**
<p float="left">
  <img src="https://github.com/user-attachments/assets/fe1e71a0-edab-46c0-84d0-e9e5296c4841" alt="screenshot 1" width="45%" style="margin-right: 20px;" />
  <img src="https://github.com/user-attachments/assets/76b7bfcc-df15-4bfd-a4e0-e5348e559ac3" alt="screenshot 2" width="45%" />
</p>

**Draw a bounding box around the duck in this scene**
<p float="left">
  <img src="https://github.com/user-attachments/assets/fcf2b64d-3d3f-4b8a-8b57-19a5b3e7117e" alt="screenshot 1" width="45%" style="margin-right: 20px;" />
  <img src="https://github.com/user-attachments/assets/7e3a8f39-e54a-463a-a1e9-1f4bc266c2ac" alt="screenshot 2" width="45%" />
</p>

**Render the benchwise mesh**
<p float="left">
  <img src="https://github.com/user-attachments/assets/1fb83ebe-68e3-469c-b704-53cbbc7570d8" alt="screenshot 1" width="45%" style="margin-right: 20px;" />
  <img src="https://github.com/user-attachments/assets/2cb28dc9-65af-4037-bcfa-e6ea5907e253" alt="screenshot 2" width="45%" />
</p>

## Download

You can [download](https://github.com/InteractiveGL/vision6D/releases/) the latest installable version of Vision6D for Windows, macOS and Linux.

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

GNU General Public License v3.0

---
