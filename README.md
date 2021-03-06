# Engineering Transfer Learning

## Overview

Code is a collection of transfer learning algorithms that have been generated or applied in engineering contexts, particularly for structural health monitoring applications.

Code mainly to reproduce paper results (as close as possible given data-availability). Scripts for papers can be found in the [demo](https://github.com/pagard/EngineeringTransferLearning/tree/main/demos) folder.

To setup the files in MATLAB run [setup.m](https://github.com/pagard/EngineeringTransferLearning/blob/main/setup.m) to add folders to path.

---

## Algorithms

These are the transfer learning models which can be found in the [models](https://github.com/pagard/EngineeringTransferLearning/tree/main/models) folder.

* Homogeneous transfer learning
  * [Transfer component analysis, (TCA)](https://doi.org/10.1109/TNN.2010.2091281)
  * [Joint distribution adaptation, (JDA)](https://doi.org/10.1109/ICCV.2013.274)
  * [Balanced distribution adaptation, (BDA)](https://doi.org/10.1109/ICDM.2017.150)
  * [Metric-informed joint distribution adaptation, (M-JDA)](https://doi.org/10.1016/j.jsv.2021.116245)
  * [Expectation Maximisation Domain adaptated Gaussian mixture model, (DA-GMM)](https://doi.org/10.1007/s13349-022-00565-5)
* Heterogeneous transfer learning
  * [Kernelised Bayesian transfer learning (KBTL)](https://users.ics.aalto.fi/gonen/files/gonen_aaai14_paper.pdf)

*Please cite the linked method papers if you use this code, as well as any corresponding application papers.*

---

## Applications and demo files

* [On the application of domain adaptation for structural health monitoring](https://doi.org/10.1016/j.ymssp.2019.106550)
  * Application of TCA, JDA and ARTL to structural health monitoring applications
  * Demo script is underconstruction []()

* [Foundations of population-based SHM Part III: Heterogeneous populations - Mapping and Transfer](https://doi.org/10.1016/j.ymssp.2020.107142)
  * Development of JDA for an (L+1)-problem, i.e. where the target label space is one greater than the source label space
  * Demo script is underconstruction []()

* [Machine learning at the inferface of structural health monitoring and non-destructive evaluation](https://doi.org/10.1098/rsta.2019.0581) [[Open Access]](https://pagard.github.io/publications/gardner-2020-d/gardner-2020-d.pdf)
  * Application of TCA to ultrasonic inspection on composite plates of differing construction
  * Demo script is underconstruction []()

* [Overcoming the problem of repair in structural health monitoring: Metric-informed transfer learning](https://doi.org/10.1016/j.jsv.2021.116245)
  * Development and demonstration of M-JDA on repair scenarios involving a Gnat aircraft
  * Demo script is [mjda_demo_gnat.m](https://github.com/pagard/EngineeringTransferLearning/blob/main/demos/mjda_demo_gnat.m) and is accompanied by a [readme](https://github.com/pagard/EngineeringTransferLearning/blob/main/demos/mjda_demo_gnat.md)

* [On the application of kernelised Bayesian transfer learning to population-based structural health monitoring](https://doi.org/10.1016/j.ymssp.2021.108519) [[Open Access]](https://pagard.github.io/publications/gardner-2022-a/gardner-2022-a.pdf)
  * Application of KBTL to multiple structural health monitoring applications; numerical and experimental shear buildings, an aircraft wing with different sensor configurations, and numerical and experimental eight degree-of-freedom structures with differing signal properties
  * Demo scripts are [kbtl_demo_binary.m](https://github.com/pagard/EngineeringTransferLearning/blob/main/demos/kbtl_demo_binary.m) and [kbtl_demo_multiclass.m](https://github.com/pagard/EngineeringTransferLearning/blob/main/demos/kbtl_demo_multiclass.m) and are accompanied by a [readme](https://github.com/pagard/EngineeringTransferLearning/blob/main/demos/kbtl_demo.md)

* [A population-based SHM methodology for heterogeneous structures: Transferring damage localisation knowledge between different aircraft wings](https://doi.org/10.1016/j.ymssp.2022.108918) [[Open Access]](https://pagard.github.io/publications/gardner-2022-b/gardner-2022-b.pdf)
  * A population-based methodology for performing damage localisation on two heterogeneous aircraft wings. The demo files demonstrate the abstract graphical representations, distance metrics and transfer learning for the four maximum common subgraphs (see paper).
  * Demo scripts are [gnat_piper_graph_visualisation.m](https://github.com/pagard/EngineeringTransferLearning/blob/main/demos/gnat_piper_graph_visualisation.m) and [gnat_piper_demo.m](https://github.com/pagard/EngineeringTransferLearning/blob/main/demos/gnat_piper_demo.m) and are accompanied by a [readme](https://github.com/pagard/EngineeringTransferLearning/blob/main/demos/gnat_piper_demo.md)

* [Domain???adapted Gaussian mixture models for population???based structural health monitoring](https://doi.org/10.1007/s13349-022-00565-5)
  * Development of the DA-GMM and demonstration on two real world bridge datasets.
  * Demo scripts are [dagmm_em_demo.m](https://github.com/pagard/EngineeringTransferLearning/blob/main/demos/dagmm_em_demo.md)