# IGGCN source audit

## Source material

- Primary source: Wangxing Chen, Haifeng Sang, Jinyu Wang and Zishan Zhao, “IGGCN: Individual-guided graph convolution network for pedestrian trajectory prediction,” *Digital Signal Processing* 156 (2025), article 104862, DOI [10.1016/j.dsp.2024.104862](https://doi.org/10.1016/j.dsp.2024.104862).
- The user supplied the 12-page publisher PDF at `/media/lrj/54926A1D926A0438/nuScence data/IGGCN.pdf`. Its SHA-256 is `c8e022306aaf58bc67bfeafd4563fdf1785e21cf0ce05d817c37a79676b11714`.
- The PDF specifies the main graph equations, Gaussian similarity, four residual 3×3 deformable-convolution layers, a five-layer TCN, bivariate Gaussian negative log-likelihood, 8 observed / 12 predicted frames at 2.5 Hz, leave-one-scene-out ETH/UCY evaluation, and the principal optimizer settings.
- The publisher article is also listed at [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S105120042400486X).

## Code provenance

- The requested repository `duchangchen11/IGGCN_NEW` was empty at initial inspection. This project contains a paper-based implementation; no author IGGCN source code was found in the source audit.
- ETH/UCY trajectory text files were obtained from the pinned [Social-STGCNN data mirror commit](https://github.com/abduallahmohamed/Social-STGCNN/tree/333d3a57b4d2705e129b21aefefa09c79b2b9ae1/datasets/raw/all_data). That mirror is used only as a data source, not as an IGGCN model implementation. File checksums and source URLs are in the external data manifest on the data disk.
- The author's public `DSTIGCN_Master` code is a different model and is not substituted for IGGCN.

## Paper details and implementation assumptions

The paper does not fully specify several choices needed for executable code: exact tensor encoding of graph node features, where the upper-triangular temporal graph mask is applied, deformable-convolution offset/padding/activation details, the TCN channel width/kernel/dilation and output projection, the activation `δ`, coordinate parameterization, validation/checkpoint selection, and random seed. Those are implementation assumptions, not reported paper facts.

The implementation will preserve the paper's stated structure and equations: displacement node features; separate spatial and temporal embedding and Q/K projections; unscaled QK products; four residual 3×3 deformable-convolution layers per branch; fixed Gaussian similarity with an identity self-edge; spatial and temporal graph aggregation; a five-layer temporal convolutional predictor; and the bivariate Gaussian NLL. The exact assumptions chosen for the unspecified items are recorded in the experiment config and run manifest. There is no adaptive or learnable sigma, added network module, decoder replacement, or loss replacement.

Training uses seed-controlled initialization and sample order, with cuDNN bitwise-deterministic algorithms disabled after a direct timing comparison showed a severe slowdown on the available GPU. The same kernel policy is used for every sigma and recorded in each run manifest.

## Reproduction gate

Run the complete fixed `sigma=3` leave-one-scene-out baseline first. Compare its per-scene and mean ADE/FDE to the paper's ETH/UCY table. If the mean error is more than approximately 10–15% worse, or scene behavior is clearly anomalous, stop before running the sigma sweep and investigate the implementation and data protocol.

## Gate outcome

The seed-42 baseline completed all five folds. Its mean ADE/FDE is 0.7578/1.4931 m versus the paper's 0.34/0.59 m, a +122.9% / +153.1% difference. The stop condition is met; fixed sigma values 1, 2, 4, and 5 were not trained, and no state/oracle analysis was run.
