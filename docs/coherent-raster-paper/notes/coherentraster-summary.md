---
type: paper
title: "CoherentRaster: Efficient 3D Gaussian Splatting for Light Field Displays"
authors:
  - "Gyujin Sim"
  - "Seungjoo Shin"
  - "Hosung Jeon"
  - "Gwangsoon Lee"
  - "Hyon-Gon Choo"
  - "Sunghyun Cho"
venue: "SIGGRAPH Conference Papers '26"
journal: null
conference: "ACM SIGGRAPH Conference Papers"
year: 2026
doi: "10.1145/3799902.3811217"
canonical_url: "https://arxiv.org/abs/2605.04509"
pdf_source: "../CoherentRaster.pdf"
raw_markdown: "../raw/coherentraster-raw.md"
figures_tables_markdown: "../raw/coherentraster-figures-tables.md"
images_dir: "../image/coherentraster"
paper_abbrev: "coherentraster"
tags:
  - "paper"
  - "research"
  - "light-field-display"
  - "3d-gaussian-splatting"
  - "rasterization"
topics:
  - "subpixel rasterization"
  - "cross-view coherence"
  - "GPU memory efficiency"
direct_prior_work:
  - title: "DirectL: Efficient Radiance Fields Rendering for 3D Light Field Displays"
    authors: ["Zongyuan Yang", "Baolin Liu", "Yingde Song", "Yongping Xiong"]
    year: 2024
    venue: "ACM Transactions on Graphics"
    doi: "10.1145/3687897"
    url: "https://doi.org/10.1145/3687897"
    relationship: "prior subpixel-level rendering approach that CoherentRaster explicitly builds beyond for 3DGS rasterization"
    evidence: "The introduction names DirectL as the first subpixel-level rendering approach and explains the remaining GPU inefficiency."
  - title: "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
    authors: ["Bernhard Kerbl", "Georgios Kopanas", "Thomas Leimkuehler", "George Drettakis"]
    year: 2023
    venue: "ACM Transactions on Graphics"
    doi: "10.1145/3592433"
    url: "https://doi.org/10.1145/3592433"
    relationship: "base renderer that CoherentRaster adapts to interlaced LFD outputs"
    evidence: "The method is a 3DGS-based light field rendering framework."
verified_sources:
  - label: "arXiv"
    url: "https://arxiv.org/abs/2605.04509"
  - label: "Project repository"
    url: "https://github.com/sgj0402/coherent-raster"
verification_notes:
  - "The DOI and SIGGRAPH 2026 citation are taken from the local PDF front matter."
created: "2026-06-02"
---
# CoherentRaster: Efficient 3D Gaussian Splatting for Light Field Displays

## Abstract
#### 초록 및 핵심 주장 요약

라이트필드 디스플레이(LFD)는 하나의 2D 화면 픽셀 배열에 여러 시점의 관측값을 서브픽셀 단위로 섞어 넣는다. 따라서 단일 카메라 뷰를 빠르게 렌더링하는 기존 3DGS를 그대로 적용하면, 수십 개 이상의 시점을 모두 렌더링한 뒤 인터레이싱해야 하므로 계산량과 메모리가 시점 수에 비례해 증가한다.

논문은 이 문제를 세 단계로 줄인다. 첫째, 최종 LFD 이미지에 실제로 쓰이는 서브픽셀만 계산한다. 둘째, 가까운 시점끼리는 Gaussian의 공분산, 깊이, SH 기반 색이 완만하게 변한다고 보고 클러스터 대표 시점에서 계산한 속성을 재사용한다. 셋째, 인터레이스드 배치 때문에 깨지는 GPU 메모리 coalescing을 회복하도록, 스레드가 처리할 서브픽셀 순서를 시점 인덱스 기준으로 재정렬한다.

핵심 주장은 이 두 최적화가 서로 보완적이라는 점이다. Cross-view Coherent Attribute Reuse는 projection/key generation/sort의 중복을 줄이고, View-coherent Remapping은 alpha blending 단계의 메모리 접근 효율을 높인다. 결과적으로 별도의 MPI 같은 무거운 중간 표현 없이, 3DGS 표현에서 바로 고해상도 LFD용 인터레이스드 이미지를 만든다.

Ref : [[../raw/coherentraster-raw.md#^p001-abstract-01|p1 abstract and main claim]]

## Methodology
#### 주요 아이디어와 파이프라인

입력은 학습된 3D Gaussian 집합, 목표 시점 집합, 그리고 디스플레이 보정 파라미터에서 정해지는 viewpoint index matrix이다. 출력은 패널 해상도와 RGB 서브픽셀 채널을 갖는 인터레이스드 light-field 이미지다.

파이프라인은 다음 순서로 동작한다.

1. **디스플레이 서브픽셀-시점 매핑 계산**: 렌티큘러 렌즈의 기울기, grating line count, panel offset으로 각 `(x, y, u)` 서브픽셀이 어느 시점 `j`에 속하는지 정한다.
2. **Subpixel-level rasterization**: 모든 시점의 full RGB frame을 만들지 않고, 최종 LFD 이미지의 각 서브픽셀에 기여하는 Gaussian만 평가한다.
3. **Cross-view Coherent Attribute Reuse**: 시점들을 인접 클러스터로 나누고, 2D mean은 시점별로 계산하되 2D covariance, depth, SH color는 클러스터 대표 시점에서 계산해 재사용한다.
4. **Key generation / sort**: Gaussian-tile pair를 view 단위가 아니라 tile-cluster 단위로 만든다. 정렬 key는 tile, cluster, depth 순서를 보존한다.
5. **View-coherent Remapping**: alpha blending 스레드를 raster order가 아니라 viewpoint index가 가까운 순서로 배치하여 같은 warp가 같은/인접 Gaussian list를 읽도록 한다.
6. **Alpha blending**: remapped subpixel 위치에서 tile과 cluster에 해당하는 Gaussian list를 순회하며 front-to-back 합성하고, 결과를 바로 LFD 이미지 위치에 쓴다.

Ref : [[../raw/coherentraster-raw.md#^p003-methodology-01|methodology evidence]]

## Conclusion
#### 강점, 한계, 가정

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

Ref : [[../raw/coherentraster-raw.md#^p005-conclusion-01|conclusion or limitation evidence]]

## Detailed Notes
- [[coherentraster-abstract]]
- [[coherentraster-related-work]]
- [[coherentraster-methodology]]
- [[coherentraster-experimental-results]]

## Raw Data
- [Raw Markdown](../raw/coherentraster-raw.md)
- [Figures and Tables](../raw/coherentraster-figures-tables.md)
- [Images](../image/coherentraster)

## Sources
- arXiv: https://arxiv.org/abs/2605.04509
- Project repository: https://github.com/sgj0402/coherent-raster
