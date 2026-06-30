# Figures and Tables: CoherentRaster: Efficient 3D Gaussian Splatting for Light Field Displays

Raw source: [[coherentraster-raw.md]]

## Figures

### Fig. 1. Through-the-lens comparison of rendered light field image. We visualize the rendered con
Page: 1
Page image: ![page 001](../image/coherentraster/pages/page-001.png)

Caption: Fig. 1. Through-the-lens comparison of rendered light field image. We visualize the rendered content on a physical Light Field Display, captured from left and right viewpoints. Comparing Full-frame rendering 3DGS and CoherentRaster, our method achieves significantly higher frame rates (FPS) while maintaining visual quality. (Garden and Bicycle scenes from the Mip-NeRF 360 dataset) Light field displays (LFDs) require rendering an interlaced image that encodes many view-dependent observations. This multi-view requirement introduces substantial computational overhead, making real-time rendering difficult

^fig-001

### Fig. 2. Principle of Lenticular Light Field Displays. (a) Viewpoint index matrix. Based on displ
Page: 3
Page image: ![page 003](../image/coherentraster/pages/page-003.png)

Caption: Fig. 2. Principle of Lenticular Light Field Displays. (a) Viewpoint index matrix. Based on display parameters, each subpixel is assigned to a unique viewpoint index, forming the viewpoint index matrix V. (b) Lenticular Light Field Display. The lenticular lens array refracts light from the LCD panel, directing each subpixel to a specific viewing angle. glasses-free 3D presentation via an interlaced image at panel reso-

^fig-002

### Fig. 3. Overall pipeline of proposed CoherentRaster. The framework synthesizes high-resolution l
Page: 5
Page image: ![page 005](../image/coherentraster/pages/page-005.png)

Caption: Fig. 3. Overall pipeline of proposed CoherentRaster. The framework synthesizes high-resolution light field images from the input 3DGS and target viewpoints. In the projection and key generation stage, Cross-view Coherent Attribute Reuse eliminates redundant computations by reusing projected attributes and generating sorting keys per cluster. Subsequently, during alpha blending, View-coherent Remapping reorganizes thread execution based on viewpoint indices, thereby restoring coalesced memory access for efficient rendering. clustering strategy mitigates the overhead of per-view evaluation that the subpixel-level 3DGS rasterization would otherwise incur.

^fig-003

### Fig. 4. Qualitative comparison on the light field display. We present photographs captured direc
Page: 10
Page image: ![page 010](../image/coherentraster/pages/page-010.png)

Caption: Fig. 4. Qualitative comparison on the light field display. We present photographs captured directly from the display to compare the visual quality of CoherentRaster with the baseline. Our method achieves real-time frame rates while maintaining perceptual quality indistinguishable from the high-cost full-frame rendering. Note that slight color shifts or misalignments may appear due to the capture process. (Rows 1 and 4: Bonsai and Bicycle scenes from the Mip-NeRF 360 dataset; Rows 2 and 3: Ship and Chair scenes from the Synthetic Blender dataset) SIGGRAPH Conference Papers ’26, July 19-23, 2026, Los Angeles, CA, USA.

^fig-004

### Fig. 5. Ablation study on cluster size. Across different cluster sizes, our method maintains fea
Page: 11
Page image: ![page 011](../image/coherentraster/pages/page-011.png)

Caption: Fig. 5. Ablation study on cluster size. Across different cluster sizes, our method maintains feasible visual quality without noticeable distortion. (Counter and Bonsai scenes from the Mip-NeRF 360 dataset; Drums and Ship scenes from the Synthetic Blender dataset) SIGGRAPH Conference Papers ’26, July 19-23, 2026, Los Angeles, CA, USA.

^fig-005

### Fig. 6. Artifacts on specular surfaces. Cross-view Coherent Attribute Reuse can introduce artifa
Page: 13
Page image: ![page 013](../image/coherentraster/pages/page-013.png)

Caption: Fig. 6. Artifacts on specular surfaces. Cross-view Coherent Attribute Reuse can introduce artifacts on highly specular surfaces (Materials scene from the Synthetic Blender dataset) C.2 Rendering Time Breakdown To analyze the performance gains of CoherentRaster, we report the per-stage execution time, peak VRAM usage, and the total number of

^fig-006

## Tables

### Table 1. Quantitative Evaluation. We evaluate rendering speed (FPS) and image quality (PSNR, SSIM
Page: 7
Page image: ![page 007](../image/coherentraster/pages/page-007.png)

Caption / raw table cue: Table 1. Quantitative Evaluation. We evaluate rendering speed (FPS) and image quality (PSNR, SSIM, LPIPS) on the Synthetic Blender [Mildenhall et al. 2021] and Mip-NeRF 360 [Barron et al. 2022] datasets. |V𝑘 | = 16 for the 63-view 2K setup and |V𝑘 | = 18 for the 71-view 4K setup are selected to ensure balanced view partitioning for each display specification.

[table note: reconstruction requires visual verification against the page image before using exact numeric cells as evidence.]

^table-001

### Table 2. Comparison with Baselines. CoherentRaster outperforms the baselines under both 2K and 4K
Page: 7
Page image: ![page 007](../image/coherentraster/pages/page-007.png)

Caption / raw table cue: Table 2. Comparison with Baselines. CoherentRaster outperforms the baselines under both 2K and 4K display configurations. Rendering Method Synthetic Blender Mip-NeRF 360 2K FPS 4K FPS 2K FPS 4K FPS Full-Frame 3DGS 5.8 4.1 3.9 2.1 3DGS (batch=36) 20 13 7.4 4.0 Subpixel Subpixel-3DGS 28 19 11 5.7

[table note: reconstruction requires visual verification against the page image before using exact numeric cells as evidence.]

^table-002

### Table 3. Ablation study on the overall pipeline. Our proposed strategies alleviate redundant comp
Page: 8
Page image: ![page 008](../image/coherentraster/pages/page-008.png)

Caption / raw table cue: Table 3. Ablation study on the overall pipeline. Our proposed strategies alleviate redundant computation and irregular memory access.

[table note: reconstruction requires visual verification against the page image before using exact numeric cells as evidence.]

^table-003

### Table 4. Rendering speed on an RTX 3090. Even on the cache-limited GPU, CoherentRaster consistent
Page: 13
Page image: ![page 013](../image/coherentraster/pages/page-013.png)

Caption / raw table cue: Table 4. Rendering speed on an RTX 3090. Even on the cache-limited GPU, CoherentRaster consistently outperforms the baselines.

[table note: reconstruction requires visual verification against the page image before using exact numeric cells as evidence.]

^table-004

### Table 5. Rendering time breakdown on the Synthetic Blender dataset. All times are in milliseconds
Page: 14
Page image: ![page 014](../image/coherentraster/pages/page-014.png)

Caption / raw table cue: Table 5. Rendering time breakdown on the Synthetic Blender dataset. All times are in milliseconds. The number of Gaussian-tile pairs are in millions.

[table note: reconstruction requires visual verification against the page image before using exact numeric cells as evidence.]

^table-005

### Table 6. Rendering time breakdown on the Mip-NeRF 360 dataset. All times are in milliseconds. The
Page: 14
Page image: ![page 014](../image/coherentraster/pages/page-014.png)

Caption / raw table cue: Table 6. Rendering time breakdown on the Mip-NeRF 360 dataset. All times are in milliseconds. The number of Gaussian-tile pairs are in millions.

[table note: reconstruction requires visual verification against the page image before using exact numeric cells as evidence.]

^table-006
