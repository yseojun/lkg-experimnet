# CoherentRaster: Efficient 3D Gaussian Splatting for Light Field Displays - Abstract

이 노트는 논문 초록과 문제 제기를 한국어로 재구성한 것이다. 원문을 그대로 옮기지 않고, 문제, 공백, 기여, 방법, 결과 주장을 분리해 읽기 쉽게 정리했다.

Ref : [[../raw/coherentraster-raw.md#^p001-abstract-01|p1 abstract and contribution]]

### 초록 및 핵심 주장 요약

라이트필드 디스플레이(LFD)는 하나의 2D 화면 픽셀 배열에 여러 시점의 관측값을 서브픽셀 단위로 섞어 넣는다. 따라서 단일 카메라 뷰를 빠르게 렌더링하는 기존 3DGS를 그대로 적용하면, 수십 개 이상의 시점을 모두 렌더링한 뒤 인터레이싱해야 하므로 계산량과 메모리가 시점 수에 비례해 증가한다.

논문은 이 문제를 세 단계로 줄인다. 첫째, 최종 LFD 이미지에 실제로 쓰이는 서브픽셀만 계산한다. 둘째, 가까운 시점끼리는 Gaussian의 공분산, 깊이, SH 기반 색이 완만하게 변한다고 보고 클러스터 대표 시점에서 계산한 속성을 재사용한다. 셋째, 인터레이스드 배치 때문에 깨지는 GPU 메모리 coalescing을 회복하도록, 스레드가 처리할 서브픽셀 순서를 시점 인덱스 기준으로 재정렬한다.

핵심 주장은 이 두 최적화가 서로 보완적이라는 점이다. Cross-view Coherent Attribute Reuse는 projection/key generation/sort의 중복을 줄이고, View-coherent Remapping은 alpha blending 단계의 메모리 접근 효율을 높인다. 결과적으로 별도의 MPI 같은 무거운 중간 표현 없이, 3DGS 표현에서 바로 고해상도 LFD용 인터레이스드 이미지를 만든다.

### 문제와 동기

LFD는 안경 없이 운동 시차와 입체감을 제공하지만, 화면의 각 서브픽셀이 서로 다른 시점을 담당한다. 일반 2D 디스플레이와 달리 한 프레임이 하나의 카메라 이미지가 아니라 여러 시점 이미지의 조합이다.

기존 접근의 병목은 세 가지다.

- Full-frame multi-view: 모든 시점의 전체 이미지를 렌더링한 뒤 필요한 서브픽셀만 뽑는다. 최종 이미지에 쓰이지 않는 픽셀까지 계산한다.
- 단순 subpixel rendering: 필요한 서브픽셀만 계산하지만, 인접 서브픽셀이 서로 다른 시점에 속해 GPU warp 내 스레드들이 서로 다른 Gaussian list를 읽게 된다.
- MPI 기반 중간 표현: 인접 시점 간 공유는 가능하지만, 깊이 plane 수가 늘면 메모리와 per-pixel plane traversal 비용이 커지고, plane discretization artifact가 생긴다.

따라서 LFD 전용 3DGS 렌더러는 “필요한 서브픽셀만 계산”하면서도 “시점 간 중복 계산을 줄이고” “GPU 메모리 접근 패턴을 정돈”해야 한다.

Ref : [[../raw/coherentraster-raw.md#^p004-results-01|reported outcome evidence]]
