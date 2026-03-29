# Style Transfer Results

This folder contains qualitative results of the style transfer experiments.

## Overview

We evaluate the effectiveness of our style transfer framework in preserving anatomical structures while adapting the visual appearance to the target domain. The goal is to ensure that clinically relevant features (e.g., lesions, optic disc boundaries, vascular patterns) remain intact after style transformation.

## Directory Structure

- `source/`: Original input images from the source domain.
- `target_style/`: Reference images representing the target style.
- `generated/`: Style-transferred images produced by the model.
- `comparison/`: Side-by-side visual comparisons (source / generated / target).

## Evaluation Criteria

The quality of style transfer is assessed based on the following aspects:

- **Structural Consistency**: Preservation of anatomical structures (e.g., vessels, optic disc, lesions).
- **Style Fidelity**: Degree to which the generated image matches the visual characteristics of the target domain.
- **Artifact Suppression**: Absence of visual artifacts or distortions.
- **Clinical Reliability**: No introduction of misleading or non-existent pathological features.

## Key Observations

- The proposed method effectively transfers global appearance (e.g., color distribution, illumination) while maintaining fine-grained structures.
- Compared to baseline methods, our approach better preserves lesion boundaries and vascular details.
- Minimal hallucination artifacts are observed, ensuring clinical interpretability.

## Notes

- All images are resized and normalized before processing.
- No additional post-processing is applied to the generated results.
- For quantitative evaluation, please refer to the main paper.
