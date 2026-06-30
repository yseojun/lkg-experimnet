# CoherentRaster: Efficient 3D Gaussian Splatting for Light Field Displays - Related Work

PDF URL: https://arxiv.org/abs/2605.04509

## Direct Prior Work

## Direct Prior Work - DirectL: Efficient Radiance Fields Rendering for 3D Light Field Displays
- 저자/연도: Zongyuan Yang, Baolin Liu, Yingde Song et al., 2024
- 관계: prior subpixel-level rendering approach that CoherentRaster explicitly builds beyond for 3DGS rasterization
- 근거: The introduction names DirectL as the first subpixel-level rendering approach and explains the remaining GPU inefficiency.
- URL/DOI: https://doi.org/10.1145/3687897

Ref : [[../raw/coherentraster-raw.md#^p002-related-work-01|related work evidence]]


## Direct Prior Work - 3D Gaussian Splatting for Real-Time Radiance Field Rendering
- 저자/연도: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler et al., 2023
- 관계: base renderer that CoherentRaster adapts to interlaced LFD outputs
- 근거: The method is a 3DGS-based light field rendering framework.
- URL/DOI: https://doi.org/10.1145/3592433

Ref : [[../raw/coherentraster-raw.md#^p002-related-work-01|related work evidence]]

## General Background

### 문제와 동기

LFD는 안경 없이 운동 시차와 입체감을 제공하지만, 화면의 각 서브픽셀이 서로 다른 시점을 담당한다. 일반 2D 디스플레이와 달리 한 프레임이 하나의 카메라 이미지가 아니라 여러 시점 이미지의 조합이다.

기존 접근의 병목은 세 가지다.

- Full-frame multi-view: 모든 시점의 전체 이미지를 렌더링한 뒤 필요한 서브픽셀만 뽑는다. 최종 이미지에 쓰이지 않는 픽셀까지 계산한다.
- 단순 subpixel rendering: 필요한 서브픽셀만 계산하지만, 인접 서브픽셀이 서로 다른 시점에 속해 GPU warp 내 스레드들이 서로 다른 Gaussian list를 읽게 된다.
- MPI 기반 중간 표현: 인접 시점 간 공유는 가능하지만, 깊이 plane 수가 늘면 메모리와 per-pixel plane traversal 비용이 커지고, plane discretization artifact가 생긴다.

따라서 LFD 전용 3DGS 렌더러는 “필요한 서브픽셀만 계산”하면서도 “시점 간 중복 계산을 줄이고” “GPU 메모리 접근 패턴을 정돈”해야 한다.

### Quilt / Light-field Display 워크플로와의 관련성

Quilt 워크플로는 여러 시점 이미지를 2D quilt texture나 인터레이스드 패널 입력으로 배치하는 과정과 맞닿아 있다. 이 논문은 “모든 view image를 먼저 만들고 quilt/interlace한다”는 전통적 흐름을 우회한다. 즉, 3DGS에서 각 LFD 서브픽셀이 요구하는 view index를 직접 참조해 최종 패널 이미지를 만든다.

실무적으로는 다음 지점이 중요하다.

- 기존 quilt renderer가 `N`개의 full-resolution view를 만드는 구조라면, CoherentRaster식 접근은 최종 panel subpixel만 직접 샘플링하는 대안이다.
- Looking Glass 계열처럼 디스플레이별 calibration으로 `V[x,y,u]`가 정해지는 경우, viewpoint index matrix와 tile-local remapping table을 캐시할 수 있다.
- quilt texture를 중간 산출물로 유지해야 하는 파이프라인에서는 이 방법을 그대로 적용하기보다, “quilt tile 전체를 렌더링하지 않고 필요한 panel sample만 계산”하는 sparse quilt/interlace backend로 해석할 수 있다.
- light-field preview와 interactive camera control에서는 full-view batch rendering보다 latency가 낮을 가능성이 크다.
- 다만 후처리, depth editing, per-view export처럼 명시적인 view image가 필요한 작업에는 별도 per-view 렌더링 경로가 여전히 필요하다.

Ref : [[../raw/coherentraster-raw.md#^p002-related-work-01|related work and background]]
