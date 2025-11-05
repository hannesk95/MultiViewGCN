## <center>TomoGraphView: 3D Medical Image Classification with Omnidirectional Slice Representations and Graph Neural Networks</center>


<center> Johannes Kiechle<sup>1,2,3,4</sup>, Stefan M. Fischer<sup>1,2,3,4</sup>, Daniel M. Lang<sup>1,3</sup>, Cosmin I. Bercea<sup>1,3</sup>, Mattew J. Nyflot<sup>6</sup>, <br> Lina Felsner<sup>1,3</sup>, Julia A. Schnabel<sup>1,3,4,5</sup>, and Jan C. Peeken<sup>2,3</sup>

<br>
<sup>1</sup> Technical University of Munich, Germany, 
<sup>2</sup> Klinikum rechts der Isar, Munich, Germany,
<sup>3</sup> Helmholtz Munich, Germany, <br>
<sup>4</sup> Munich Center for Machine Learning,
<sup>5</sup> King's College London, United Kingdom,
<sup>6</sup> University of Washingtion, USA </center>
<br>


<p align="center">
  <img src="./figures/method.png" width="800"/>
</p>

Submitted to [Medical Image Analysis (MedIA)](https://www.sciencedirect.com/journal/medical-image-analysis) | [preprint](https://arxiv.org/pdf/XXXXX)

**Abstract:** The growing number of medical tomography examinations has necessitated the development of automated methods capable of extracting comprehensive imaging features to facilitate downstream tasks such as tumor characterization, while assisting physicians in managing their growing workload. However, 3D medical image classification remains a challenging task due to the complex spatial relationships and long-range dependencies inherent in volumetric data. Training models from scratch suffers from low data regimes, and the absence of 3D large-scale multimodal datasets has limited the development of 3D medical imaging foundation models. Recent studies, however, have highlighted the potential of 2D vision foundation models, originally trained on natural images, as powerful feature extractors for medical image analysis. Despite these advances, existing approaches that apply 2D models to 3D volumes via slice-based decomposition remain suboptimal. Conventional volume slicing strategies, which rely on canonical planes such as axial, sagittal, or coronal, may inadequately capture the spatial extent of target structures when these are misaligned with standardized viewing planes. Furthermore, existing slice-wise aggregation strategies rarely account for preserving the volumetric structure, resulting in a loss of spatial coherence across slices. To overcome these limitations, we propose TomoGraphView, a novel framework that integrates omnidirectional volume slicing with spherical graph-based feature aggregation. Unlike traditional methods, which are restricted to canonical views, our approach samples both canonical and non-canonical cross-sections. These non-canonical views are derived from uniformly distributed points on a sphere, which visually encompasses the 3D volume, thereby producing a richer set of cross-sectional representations. As the spherically distributed viewpoints naturally define a spherical graph topology after triangulation, we allow for the explicit encoding of spatial relationships across views as nodes and corresponding edges in the underlying graph topology, and leverage a graph neural network for spatial-aware feature aggregation. Experiments across six oncology 3D medical image classification datasets demonstrate that omnidirectional volume slicing improves the average performance in Area Under the Receiver Operating Characteristic Curve (AUROC) from 0.7701 to 0.8154 compared with traditional slicing approaches relying on canonical view planes. Moreover, we can further improve AUROC performance from 0.8198 to 0.8372 by leveraging our proposed graph neural network-based feature aggregation. Notably, TomoGraphView surpasses large-scale pretrained 3D medical imaging models across all datasets and tasks, underscoring its effectiveness as a powerful framework for volumetric analysis and therefore represents a key step toward bridging the gap until fully native 3D foundation models become available in medical image analysis.

**Keywords:** 3D Medical Image Classification · Omnidirectional Volume Slicing · Graph Neural Networks