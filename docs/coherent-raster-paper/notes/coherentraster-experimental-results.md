# CoherentRaster: Efficient 3D Gaussian Splatting for Light Field Displays - Experimental Results

이 노트는 datasets (데이터셋), baselines (비교 방법), metrics (평가지표), 정량/정성 결과, 한계를 중심으로 정리한다.

Ref : [[../raw/coherentraster-raw.md#^p004-results-01|results evidence]]

### 실험 및 결과 요약

구현은 `gsplat` 기반이며, tile intersection에는 AccuTile을 포함했다. 실제 장면에는 3DGS-MCMC regularization을 적용했다. 실험 GPU는 RTX 5090 32GB이고, 추가 분석에서 RTX 3090도 확인했다.

평가 데이터셋은 Synthetic Blender 8개 장면과 Mip-NeRF 360 7개 장면이다. 2K 설정은 Looking Glass Go, 4K 설정은 Looking Glass 16" LFD를 기준으로 하며, 논문은 53도 범위에서 2K는 63 views, 4K는 71 views를 사용한다.

주요 정량 결과는 다음과 같다.

- 기본 cluster size `|V_k| = 8`에서 CoherentRaster는 Synthetic Blender 기준 2K 88 FPS, 4K 56 FPS를 기록했다.
- 같은 설정에서 Mip-NeRF 360은 2K 30 FPS, 4K 16 FPS였다.
- full-frame 3DGS는 각각 Synthetic 2K/4K 5.8/4.1 FPS, Mip-NeRF 360 2K/4K 3.9/2.1 FPS였다.
- baseline 비교에서 batched 3DGS, Subpixel-3DGS, MPI보다 모두 빠르다. 특히 Mip-NeRF 360 4K에서는 full-frame 3DGS 2.1 FPS 대비 CoherentRaster 16 FPS로 약 7.6배다.
- MPI baseline은 2K/4K에서 대체로 0.8/0.4 FPS 수준으로 느리고, PSNR도 CoherentRaster보다 낮다고 보고된다.
- ablation에서 Reuse와 Remap을 모두 끈 경우는 Synthetic 2K/4K 28/19 FPS, Mip-NeRF 360 2K/4K 11/5.7 FPS다. 둘을 모두 켜면 88/56, 30/16 FPS로 상승한다.
- RTX 3090 보충 실험에서도 CoherentRaster는 Synthetic 2K/4K 36/23 FPS, Mip-NeRF 360 2K/4K 12/6 FPS로 baseline보다 빠르다.

품질 평가는 test trajectory의 실제 ground truth가 없어서 원본 3DGS 렌더링을 pseudo ground-truth로 사용한다. `|V_k|`가 커질수록 속도는 좋아지지만 PSNR/SSIM/LPIPS는 악화된다. 논문은 기본값 `|V_k| = 8`을 속도와 품질의 균형점으로 선택한다.

### 강점, 한계, 가정

강점:

- LFD의 최종 인터레이스드 구조를 직접 목표로 삼아 full-frame multi-view 낭비를 제거한다.
- 3DGS 표현을 MPI나 depth plane stack으로 변환하지 않아 중간 표현의 plane discretization 문제를 피한다.
- GPU warp의 실제 메모리 접근 패턴까지 고려해, 알고리즘 최적화와 하드웨어 최적화를 함께 다룬다.
- 보충 실험에서 L2 cache가 작은 RTX 3090에서도 개선이 유지되어, RTX 5090에만 의존하는 결과는 아니다.

한계:

- Cross-view reuse는 클러스터 내 시점 변화가 작고 속성이 부드럽다는 가정에 의존한다.
- 고주파 specular highlight처럼 시점 변화에 민감한 효과에서는 artifact가 생길 수 있다.
- 현재 프레임워크는 정적 장면을 대상으로 한다.
- quality metric은 실제 캡처 ground truth가 아니라 full-frame 3DGS pseudo ground-truth와 비교한다.
- baselines 중 일부는 원 논문 코드가 없어 저자들이 핵심 아이디어를 재구현한 비교다.

가정:

- LFD의 렌즈/패널 파라미터가 고정되어 있고, viewpoint index matrix와 remapping table을 사전 계산할 수 있다.
- 목표 시점들이 dense하고 인접해 있어 클러스터 기반 속성 재사용이 성립한다.
- 3DGS 모델은 이미 장면별로 학습되어 있으며, 이 논문은 주로 렌더링 단계 최적화에 초점을 둔다.

Ref : [[../raw/coherentraster-figures-tables.md#^table-001|Table 1 or first quantitative table]]
Ref : [[../raw/coherentraster-raw.md#^p005-conclusion-01|limitation or conclusion evidence]]
