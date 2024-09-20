<h1 align="center">
  <br>
  <a href="http://www.amitmerchant.com/electron-markdownify"><img src="https://raw.githubusercontent.com/amitmerchant1990/electron-markdownify/master/app/img/markdownify.png" alt="Vision6D" width="200"></a>
  <br>
  Vision6D
  <br>
</h1>

<h4 align="center">Vision6D: 3D-to-2D visualization and annotation tool for 6D pose estimation desktop app built on top of <a href="https://docs.pyvista.org" target="_blank">Pyvista</a>, <a href="https://vtk.org/" target="_blank">VTK</a>, <a href="https://www.riverbankcomputing.com/static/Docs/PyQt5/" target="_blank">PyQT5</a>, <a href="https://github.com/pyvista/pyvistaqt" target="_blank">pyvistaqt</a>. This python-based app is designed to be cross-platform, and the generated 6D pose results can be seamlessly integrate into any scripts. </h4>

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
  <a href="#download">Download</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

![screenshot](https://raw.githubusercontent.com/amitmerchant1990/electron-markdownify/master/app/img/markdownify.gif)

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
  - Windows, Linux and macOS ready (highly recommend to use with a mouse).

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

## Example Usage
Set a ground-truth pose for visualization

PnP resgitration

Free-hand resgitration

Draw a segmentation mask

Draw a bounding box

Render the mesh


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

<!-- ## Related

[markdownify-web](https://github.com/amitmerchant1990/markdownify-web) - Web version of Markdownify -->

## License

GNU

---

<!-- > [amitmerchant.com](https://www.amitmerchant.com) &nbsp;&middot;&nbsp; -->
> GitHub [@ykzzyk](https://github.com/ykzzyk) &nbsp;&middot;&nbsp;


