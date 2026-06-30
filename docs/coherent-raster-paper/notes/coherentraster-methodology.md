# CoherentRaster: Efficient 3D Gaussian Splatting for Light Field Displays - Methodology

아래 설명은 methodology (방법론)를 단계별로 풀어 쓴 것이다. 영어 technical term (기술 용어)은 한국어 의미를 함께 붙였고, 변수나 모듈은 원 raw md에서 확인 가능한 범위 안에서만 설명했다.

Ref : [[../raw/coherentraster-raw.md#^p003-methodology-01|methodology core block]]

### 주요 아이디어와 파이프라인

입력은 학습된 3D Gaussian 집합, 목표 시점 집합, 그리고 디스플레이 보정 파라미터에서 정해지는 viewpoint index matrix이다. 출력은 패널 해상도와 RGB 서브픽셀 채널을 갖는 인터레이스드 light-field 이미지다.

파이프라인은 다음 순서로 동작한다.

1. **디스플레이 서브픽셀-시점 매핑 계산**: 렌티큘러 렌즈의 기울기, grating line count, panel offset으로 각 `(x, y, u)` 서브픽셀이 어느 시점 `j`에 속하는지 정한다.
2. **Subpixel-level rasterization**: 모든 시점의 full RGB frame을 만들지 않고, 최종 LFD 이미지의 각 서브픽셀에 기여하는 Gaussian만 평가한다.
3. **Cross-view Coherent Attribute Reuse**: 시점들을 인접 클러스터로 나누고, 2D mean은 시점별로 계산하되 2D covariance, depth, SH color는 클러스터 대표 시점에서 계산해 재사용한다.
4. **Key generation / sort**: Gaussian-tile pair를 view 단위가 아니라 tile-cluster 단위로 만든다. 정렬 key는 tile, cluster, depth 순서를 보존한다.
5. **View-coherent Remapping**: alpha blending 스레드를 raster order가 아니라 viewpoint index가 가까운 순서로 배치하여 같은 warp가 같은/인접 Gaussian list를 읽도록 한다.
6. **Alpha blending**: remapped subpixel 위치에서 tile과 cluster에 해당하는 Gaussian list를 순회하며 front-to-back 합성하고, 결과를 바로 LFD 이미지 위치에 쓴다.

### 핵심 기술 세부사항

#### LFD viewpoint index matrix

LCD 좌표 `(x, y)`와 RGB 서브픽셀 채널 `u`에 대해 렌티큘러 grating 안의 수평 offset을 계산한다. 논문 표기상 핵심 형태는 다음과 같다.

```text
d_offset = 3x + u + 3y tan(alpha) - K_offset
x_offset = d_offset mod L_x
j = floor(N * x_offset / L_x)
```

여기서 `N`은 총 시점 수다. 모든 서브픽셀에 대해 `j`를 모으면 `V[x, y, u]`가 되고, 이것이 최종 인터레이싱 규칙이다. 중요한 점은 RGB 채널도 별도 서브픽셀로 취급한다는 것이다.

#### 3DGS의 LFD 확장

표준 3DGS는 한 시점에서 Gaussian을 2D 화면으로 project하고, tile별로 depth sort한 뒤 alpha blending한다. LFD에서는 이 과정을 시점 수 `N`만큼 반복하면 비용이 급증한다. 단순한 subpixel 3DGS는 각 서브픽셀의 시점 `V[x,y,u]`에 맞는 Gaussian list를 읽지만, 같은 tile 안에서도 인접 스레드가 서로 다른 시점 list를 접근해 메모리 coalescing이 나빠진다.

#### Cross-view Coherent Attribute Reuse

논문은 Gaussian의 projected attribute를 두 부류로 나눈다.

- 시점별로 유지: `mu_ij^2D`. 작은 시점 변화에도 화면 위치와 tile coverage가 바뀔 수 있어 재사용하면 기하학적 오류가 커진다.
- 클러스터 단위로 재사용: `Sigma_ik^2D`, `d_ik`, `c_ik`. 인접 시점에서는 공분산, 깊이, SH 색 변화가 상대적으로 완만하다고 가정한다.

즉, 클러스터 `V_k`의 대표 시점 `v'_k`에서 다음 속성을 한 번 계산해 같은 클러스터의 시점들이 공유한다.

```text
Sigma_ik^2D = Pi_cov(v'_k; G_i)
d_ik        = Pi_depth(v'_k; G_i)
c_ik        = Pi_SH(v'_k; G_i)
```

이 전략은 `M x N` 규모의 모든 속성 계산을 일부 `M x K` 규모로 낮춘다. 단, 2D mean은 `M x N`으로 유지해 서브픽셀 위치와 tile coverage 오류를 줄인다.

#### Tile-cluster key

기존 key가 tile과 depth 중심이라면, CoherentRaster는 tile, cluster, depth를 묶는다. 보충 알고리즘의 bit layout은 다음과 같은 구조다.

```text
key = (tile_id << (32 + Bit_K)) | (cluster_id << 32) | depth_bits
```

이렇게 하면 정렬 결과가 `(tile, cluster)`별 Gaussian list로 모이고, 각 list 안에서는 depth 순서를 따른다. Gaussian이 클러스터 내 일부 시점에서는 실제로 타일과 겹치지 않을 수 있지만, 최종 alpha blending에서 mean/covariance 기반 opacity가 다시 평가되므로 영향이 작다고 설명한다.

#### View-coherent Remapping

렌티큘러 LFD에서는 물리적으로 가까운 서브픽셀이 같은 시점에 속하지 않는다. 따라서 raster order로 스레드를 배치하면 한 warp 안의 32개 스레드가 서로 다른 viewpoint/cluster의 Gaussian list를 읽기 쉽다.

해결책은 tile 내부 서브픽셀 좌표를 `V[x]` 기준으로 정렬한 lookup table `Psi`를 미리 만드는 것이다.

```text
V[Psi(r)] <= V[Psi(r + 1)]
```

렌즈 구조가 고정되어 있으므로 이 remapping table은 프레임마다 다시 만들 필요가 없다. 렌더링 중 thread rank `r`은 `x_hat = Psi(r)`를 통해 실제 쓸 서브픽셀 위치를 얻고, 해당 위치의 tile, viewpoint, cluster를 찾아 blending한다.

#### Alpha blending

최종 색은 정렬된 Gaussian list를 front-to-back으로 순회하며 누적한다. 논문은 색 누적을 일반적인 3DGS alpha compositing 형태로 쓴다. 핵심은 opacity 계산에서 시점별 mean `mu_ij^2D`와 클러스터 공유 covariance `Sigma_ik^2D`, 색 `c_ik`를 함께 사용한다는 점이다.

### 구현 및 재현 메모

- 코드 URL이 PDF에 명시되어 있으므로 재현은 `sgj0402/coherent-raster` 저장소 확인부터 시작한다.
- 기본 구현 기반은 `gsplat`; CUDA kernel 수정이 핵심이므로 Python 레벨 래퍼만으로는 재현하기 어렵다.
- 필요한 사전 계산:
  - display calibration 기반 `V[x,y,u]`
  - tile별 viewpoint-sorted remapping table `Psi`
  - target viewpoints와 cluster partition
- 기본 cluster size는 `|V_k| = 8`로 보고된다. 2K 63 views와 4K 71 views의 균등 분할을 위해 표 1에서는 `|V_k| = 16`, `18`도 별도로 평가했다.
- representative view는 클러스터의 median-index camera로 선택한다. 시점 수가 cluster size로 나누어떨어지지 않으면 마지막 camera를 복제해 padding한다.
- 성능 측정은 final interlaced image 생성 FPS 기준이며, 학습 시간이나 Gaussian 최적화 시간은 주 관심사가 아니다.
- 품질 비교를 할 때 실제 LFD through-the-lens capture는 색 틀어짐이나 alignment 오차가 섞일 수 있다. 논문도 이 점을 qualitative figure 해석 시 주의한다.
- specular scene에서는 cluster size를 줄이거나, SH color만 시점별로 계산하는 hybrid variant를 검토할 수 있다.

## 읽는 순서 예시
1. 먼저 입력 representation (표현)이 무엇인지 확인한다.
2. 그 입력이 display geometry (디스플레이 기하) 또는 multi-view consistency (다중 시점 일관성) 제약과 어떻게 연결되는지 본다.
3. 마지막으로 quality-speed trade-off (품질-속도 절충)를 만드는 파라미터와 실험 설정을 확인한다.

Ref : [[../raw/coherentraster-figures-tables.md#^fig-001|Fig. 1 overview or teaser]]
