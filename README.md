<h1 align="center">
  <br>
  <a href=""><img src="https://github.com/user-attachments/assets/77a97754-8a34-44e9-b22a-a506842699a2" alt="Vision6D" width="300"></a>
  <br>
  VISION6D
  <br>
</h1>

<h4 align="center">3D-to-2D visualization and annotation desktop app for 6D pose estimation related tasks.</h4>

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
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

![screenshot](https://github.com/user-attachments/assets/a3697ff5-1270-4da6-9273-a1f2ae1e19be)

## Introduction

Source code for paper **Vision6D: 3D-to-2D Interactive Visualization and Annotation Tool for 6D Pose Estimation**. We compared the human annotations (only rely on visual cues) with the provided ground-truth camera poses from public dataset Linemod [1] and HANDAL [2]. In an user study, the rotation and translation errors are minimal [3].

The contributions of this work can be summarized as the following:

1. Vision6D provides an interactive framework that effectively aligns 3D models onto 2D images, enabling precise 6D pose annotation. This bridges the gap between 2D image projection and the spatial complexity of 3D scenes. The tool allows users to efficiently annotate and refine 6D poses via an interactive user interface, simplifying the 6D camera pose related dataset generation process.

2. We validate the effectiveness of Vision6D through a comprehensive user study, demonstrating that it offers an intuitive and accurate solution for 6D pose annotation. The user study used public 6D pose estimation datasets named Linemod [1] and HANDAL [2], where user-annotated poses were compared against ground-truth poses. The results illustrate the tool’s accuracy, efficiency, and usability, highlighting its potential as a standardized solution for 6D pose annotation.

[1] S. Hinterstoisser, V. Lepetit, S. Ilic, S. Holzer, G. Bradski, K. Konolige, and N. Navab, “Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes,” in Computer Vision – ACCV 2012, K. M. Lee, Y. Matsushita, J. M. Rehg, and Z. Hu, Eds. Berlin, Heidelberg: Springer Berlin Heidelberg, 2013, pp. 548–562.

[2] A. Guo, B. Wen, J. Yuan, J. Tremblay, S. Tyree, J. Smith, and S. Birchfield, “Handal: A dataset of real-world manipulable object categories with pose annotations, affordances, and reconstructions,” 2023. [Online]. Available: https://arxiv.org/abs/2308.01477.

[3] Zhang, Y., Davalos, E., & Noble, J. (2025). Vision6D: 3D-to-2D Interactive Visualization and Annotation Tool for 6D Pose Estimation. arXiv [Cs.GR]. Retrieved from http://arxiv.org/abs/2504.15329

## Key Features

- LivePreview - Make changes, See changes
  - Instantly see what your pose annotation in Vision6D as you move the 3D objects!
- Provide built-in [NOCS](https://arxiv.org/abs/1901.02970) color representation for the 3D meshes
  - color the meshes with NOCS.
- Load the textures for the 3D meshes
  - color the meshes with their own textures.
- Segmentation Mask/Bounding Box Drawing
  - create a segmentation mask in Vision6D on top of the provided 2D image.
- Real-time rendering results
  - renders the annotated results.
- Cross platform
  - Windows, Linux (Ubuntu-tested), and Mac (Apple Silicon) ready (highly recommend to use with a mouse).

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

You can [download](https://github.com/InteractiveGL/vision6D/releases/) the latest installable version of Vision6D for Windows, macOS (support both Apple Silicon (ARM-based) and Intel (x86-based)), and Linux Ubuntu.

## Credits

This software uses the following open source packages:

- [Pyvista](https://docs.pyvista.org)
- [VTK](https://vtk.org/)
- [PyQt5](https://www.riverbankcomputing.com/static/Docs/PyQt5/)
- [pyvistaqt](https://github.com/pyvista/pyvistaqt)
- [NumPy](https://numpy.org/)

## Citation

If you find this work is helpful, please consider cite the following paper:

```
@misc{zhang2025vision6d3dto2dinteractivevisualization,
title={Vision6D: 3D-to-2D Interactive Visualization and Annotation Tool for 6D Pose Estimation},
author={Yike Zhang and Eduardo Davalos and Jack Noble},
year={2025},
eprint={2504.15329},
archivePrefix={arXiv},
primaryClass={cs.GR},
url={https://arxiv.org/abs/2504.15329},
}
```

Thank you for your support.

## License

GNU General Public License v3.0

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=InteractiveGL/vision6D&type=Date)](https://www.star-history.com/#InteractiveGL/vision6D&Date)
