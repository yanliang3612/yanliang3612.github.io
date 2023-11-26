---
title: "论文笔记之Learning Harmonic Molecular Representations on Riemannian Manifold(1/2)"
date: 2023-11-15 
categories: [AI for Science] 
tags: [Molecular Representation Learning] 
layout: post 
---

# 论文笔记之Learning Harmonic Molecular Representations on Riemannian Manifold

**Authors:** Yiqun Wang, Yuning Shen, Shi Chen, Lihao Wang, Fei Ye, Hao Zhou  
**Affiliations:** ByteDance Research, University of Wisconsin-Madison, Institute for AI Industry Research (AIR), Tsinghua University  
**Published:** ICLR 2023

### 1. Abstract

- Introduces a new framework, Harmonic Molecular Representation (HMR), for molecular representation learning.
- Utilizes Laplace-Beltrami eigenfunctions for a multi-resolution molecular surface representation.
- Shows improved performance in ligand-binding protein pocket classification and rigid protein docking.

### 2. Introduction

- Laplace-Beltrami eigenfunctions on the molecular surface (a 2D manifold).
- Manifold harmonic message passing for realizing holistic molecular representations
- Learning regional functional correspondence for molecular surface matching

#### Molecular Surface Representation

- Already have some work

  

#### Geometry Processing

- Recently, deep learning has been applied to learn representative features to facilitate shape recognition and matching.

#### Spectral Message Passing

- In other words, the underlying manifold and its spectrum remain the same with different surface discretizations, hence is a robust representation of the surface shape (Coifman et al., 2005).

### 3. Preliminaries

- The molecular surface 1 can be viewed as a 2D Riemannian manifold (M), which adopts a discrete set of eigenfunctions $\phi$ that solves

$$
\Delta \phi_i = \lambda_i \phi_i, \quad i = 0, 1, \ldots
$$

Here, $\Delta$ is the Laplace-Beltrami (LB) operator acting on surface scalar fields, defined as $\Delta f = -div(\Delta f)$. $\phi_0,\phi_1,...$ are a set of orthonormal eigenfunctions (i.e., $\langle \phi_i, \phi_j \rangle_M = \delta_{ij}$.) 

- Basic of Manifold Harmonic analysis

Now, we introduce basic manifold harmonic analysis, a merit of using the Riemannian manifold representation. “Harmonic analysis” refers to the representation of functions as the superposition of some basic waves. Specifically in our case, given the molecular surface manifold. 

- Specifically in our case, given the molecular surface manifold M and its LB eigenfunctions $\{\phi_i\}_{i=0}^{\infty}$, any scalar-valued function f that is square-integrable on M can be decomposed into a generalized Fourier series:

$$
f(x) = \sum_{i=0}^{\infty} \langle f, \phi_i \rangle_{\mathcal{M}} \phi_i(x)
$$

In other words, f can be represented as the linear combination of the LB eigenfunctions.



### 4. Methodology

how do we properly learn these geometric and chemical features on the molecular surface? One viable solution is to emulate the message passing framework commonly used in GNNs, whose goal is to propagate information between distant surface regions to encode surface features at different scales.

#### 4.1 Learning Hamrmonic Molecular Representations

- A discretized molecular surface manifold (i.e., a triangle mesh) with $N$ vertices $\{x_1, x_2, x_3, ...x_N\} \in \mathbb{R}^3$  and the corresponding faces.

- A set of $n$ per-vertex features, $F \in \mathbb{R}^{N \times n}$, which can be viewed as $n$ learned surface functions.
  - We use MSMS (Ewing & Hermisson, 2010) to compute the molecular solvent-excluded surface as a triangle mesh with $N$ vertices. Then, we compute the first $k$ LB eigenfunctions $\{\phi_i\}_{i>0}^{k-1}$ with ascending eigenvalues as described in Reuter et al. (2009), and stack them into an array $\Phi \in \mathbb{R}^{N \times k}$, where each column stores an eigenfunction.

For each vertex, we encode its neighboring atoms within a predefined radius (e.g., 6) through MLP, then sum over the neighbors to obtain its chemical embedding. We use another MLP to combine the per-vertex initial features $F_{\text{inp}} \leftarrow MLP(\text{concat}(F_{\text{geom}}, F_{\text{chem}})), \quad F_{\text{inp}} \in \mathbb{R}^{N \times n}.$ These $n$ features reflect the local geometric and chemical environment of each surface vertex, which will be used as input to the harmonic message passing module.

- The output of the surface preparation module includes
  - the molecular surface triangle mesh.
  - the surface Laplace-Beltrami eigenfunctions $\Phi$.
  - the per-vertex features $F^{\text{inp}}$.

#### 4.2 Harmonic Message Passing

Our proposed harmonic message passing mechanism is closely related to the heat diffusion process on an arbitrary surface. Joseph Fourier developed spectral analysis methods to solve the heat equation $∂f/∂t + ∆f = 0$, where $f$ is some heat distributed on the surface. This concise partial differential equation describes how a heat distribution f evolves over time, whose solution can be expressed using the heat operator $\text{exp}(-\Delta t)$, i.e., $f(t) = exp(-\Delta t) f_0 $ for initial heat distribution $f_0$ at $t=0$. Intuitively, heat will flow from hot regions to cool regions on the surface. As time approaches infinity, the heat distribution $f$ will converge to a constant value (i.e., the global average temperature on the surface), assuming that total energy is conserved.

In fact, heat diffusion can be thought of as a message passing process, where surface regions with different temperatures communicate with each other and propagate the initial heat distribution deterministically. 

- In fact, heat diffusion can be thought of as a message passing process, where surface regions with different temperatures communicate with each other and propagate the initial heat distribution deterministically. The heat exchange rate is dependent on the difference in temperature (determined by the LB operator), while the message passing distance is determined by the heat diffusion time $t$.

Following this idea, we generalize the heat diffusion process by proposing a function propagation operator $\mathcal{P}$ with nerual network-learned frequency filter $F_{\theta}(\lambda)$:
$$
\mathcal{P}f = \sum_{i}F_{\theta}(\lambda_i) \langle f, \phi_i \rangle_\mathcal{M}\phi_i
$$

$$
F_{\theta}(\lambda) = \text{exp}(-\frac{(\lambda-\mu)^2}{\sigma^2}) \cdot \text(exp)(-\lambda t ), 
\text{where} \ \theta = (\mu, \sigma, t).
$$

**In addition, molecule-level representations could be obtained through global pooling (see Sec. 5.1).**

#### 4.3 Learning Surface Correspondence for Rigid Protein Docking

- given two proteins surfaces, predict the region where binding might occur（i.e. binding site prediction, locating the missing piece)
- establish functional correspondence between the ligand/receptor binding surfaces, and convert it to real-space vertex-to-vertex correspondence (shape/pattern matching). Rigid docking could then be achieved by aligning the corresponding binding site surface vertices.
- Given the ligand and receptor protein surface meshes, we first predict the regions where they interact, which is a per-vertex binary classification problem. We iteratively apply HMR and cross-attention layers (Fig. 5) to encode the surfaces with intra- and inter-surface communications. Next, we use the learned features on each vertex to classify whether it belongs to the binding interface. Detailed descriptions of this module are available in Appendix D.
- **Moore-Penrose pseudo-inverse**: Moore-Penrose 伪逆
- **Frobenius norm**: Frobenius 范数是矩阵范数的一种，用于量化一个矩阵的大小或者复杂度。它是由矩阵中所有元素的绝对值的平方和的平方根给出的。
- **Kabsch algorithm**: Kabsch算法是一种用于分子动力学和结构生物学中，用于确定两组点之间最优旋转的方法。

#### 5. Experiements

- **QM9 Molecular Property Regression**

  Interestingly, despite H MR completely discards the bonding information and only performs massage passing over the molecular surface, it still shows comparable performance in predicting these molecular properties.

- **Ligand-Binding Pocket Classification**

  In addition, we draw a similar conclusion that both geometric and chemical information of the binding pockets are important in predicting the type of its binding molecules.

- **Rigid Protein Docking**

  HMR predicts multiple binding sites for some proteins (either due to certain protein symmetries or model uncertainty). Therefore, we also assess the model performance by including candidate poses from top 3 binding site pairs, ranked by the mean probability predicted by the binding site classifer.











