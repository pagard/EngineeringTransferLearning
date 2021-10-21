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

* Heterogeneous transfer learning
  * [Kernelised Bayesian transfer learning (KBTL)](https://users.ics.aalto.fi/gonen/files/gonen_aaai14_paper.pdf)

---

## Applications and demo files

* [On the application of domain adaptation for structural health monitoring](https://doi.org/10.1016/j.ymssp.2019.106550)
  * Application of TCA, JDA and ARTL to structural health monitoring applications
  * Demo script is underconstruction []()

* [Foundations of population-based SHM Part III: Heterogeneous populations - Mapping and Transfer](https://doi.org/10.1016/j.ymssp.2020.107142)
  * Development of JDA for an (L+1)-problem, i.e. where the target label space is one greater than the source label space
  * Demo script is underconstruction []()

* [Machine learning at the inferface of structural health monitoring and non-destructive evaluation](https://doi.org/10.1098/rsta.2019.0581)
  * Application of TCA to ultrasonic inspection
  * Demo script is underconstruction []()

* [Overcoming the problem of repair in structural health monitoring: Metric-informed transfer learning](https://www.sciencedirect.com/science/article/pii/S0022460X21003175)
  * Demonstration of M-JDA on repair scenarios involving a Gnat aircraft
  * Demo script is [mjda_demo_gnat](https://github.com/pagard/EngineeringTransferLearning/blob/main/demos/mjda_demo_gnat.m)

* On the application of kernelised Bayesian transfer learning to population-based structural health monitoring, *in press*
  * Application of KBTL to structural health monitoring applications
  * Demo script is under construction []()