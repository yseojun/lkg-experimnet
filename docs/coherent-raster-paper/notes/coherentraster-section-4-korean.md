# CoherentRaster - 4장 한국어 해석

Source: [[../raw/coherentraster-raw.md#^p003-methodology-01|raw methodology block]], [[../raw/coherentraster-figures-tables.md#^fig-003|Fig. 3 pipeline]]

이 문서는 논문 4장 `CoherentRaster`를 문단 흐름대로 한국어로 풀어 쓴 해석이다. 수식과 핵심 변수명은 원문 표기를 유지했다.

## 4 CoherentRaster

CoherentRaster의 입력은 3D Gaussian 집합 $G=\{G_i\}_{i=0}^{M-1}$과 3D 디스플레이 설정이다. 이 설정을 바탕으로 최종 출력인 고해상도 interlaced light-field image (인터레이스드 라이트 필드 이미지) $I_{LF}\in \mathbb{R}^{W\times H\times 3}$를 만든다. 디스플레이 설정에는 목표 viewpoint (시점) 집합 $\mathcal{V}=\{v_i\}_{i=0}^{N-1}$와 viewpoint index matrix (시점 인덱스 행렬) $\mathbf{V}\in \mathbb{Z}^{W\times H\times 3}$가 들어 있다. 이 행렬은 최종 패널의 각 subpixel (서브픽셀)이 어떤 시점을 표시해야 하는지를 정한다.

Fig. 3(a)의 전체 구조처럼, CoherentRaster는 3D Gaussian 표현을 light-field display (라이트 필드 디스플레이)의 subpixel-level layout (서브픽셀 단위 배치)에 맞게 바꾼다. 기존 light field coding 방식은 각 시점마다 완전한 RGB 이미지를 먼저 렌더링한 뒤, 그 이미지에서 필요한 subpixel 값만 뽑아 interlaced image를 만든다. 반면 CoherentRaster는 각 subpixel $(x,y,u)$에 대해 어떤 Gaussian이 기여하는지와 어떤 색을 낼지를 직접 계산한다. 그래서 시점별 full RGB frame을 만들 필요가 없고, rasterization (래스터화) 자체를 subpixel 단위로 확장해 여러 시점의 기여를 한 번에 모은다.

## 4.1 Subpixel-Level Rasterization

Light-field display에서는 여러 시점이 subpixel 단위로 섞여 있다. 각 subpixel $(x,y,u)$는 viewpoint index matrix에 의해 $\mathbf{V}[x,y,u]$라는 시점에 배정된다. 따라서 여기서의 rasterization은 하나의 시점에 대한 픽셀 렌더링이 아니라, 여러 시점에 대한 Gaussian 기여를 동시에 계산해야 한다. 또한 계산 단위도 pixel이 아니라 subpixel이다.

이를 위해 논문은 기존 tile-based rasterization (타일 기반 래스터화)을 subpixel-level multi-view rendering (서브픽셀 단위 다중 시점 렌더링)으로 확장한다. 확장은 네 단계로 설명된다: projection (투영), key generation (키 생성), sorting (정렬), alpha blending (알파 합성).

### Projection

각 viewpoint $v_j$에 대해 모든 Gaussian $G_i$를 image plane (이미지 평면)에 투영한다. 그 결과 해당 Gaussian의 2D mean (2D 평균 위치), 2D covariance (2D 공분산), depth (깊이), view-dependent color (시점 의존 색상)가 나온다.

$$
\mu^{2D}_{i,j}, \Sigma^{2D}_{i,j}, d_{i,j}, c_{i,j}
= \Pi(v_j; G_i)
$$

여기서 $\Pi(\cdot)$는 projection operator (투영 연산자)다. 단순하게 구현하면 Gaussian 하나마다 시점 수 $N$개만큼의 screen-space attribute (화면 공간 속성) 세트를 갖게 된다.

### Key Generation

각 시점 $v_j$에서 투영된 Gaussian이 어떤 tile (타일)과 겹치는지 찾는다. 이후 각 Gaussian-tile pair (Gaussian-타일 쌍)에 key를 붙인다. 이 key에는 tile ID, viewpoint ID $j$, view-space depth $d_{i,j}$가 들어간다. 즉 이후 정렬을 위해 "어느 타일인지", "어느 시점인지", "앞뒤 깊이 순서가 어떻게 되는지"를 함께 기록한다.

### Sorting

모아진 Gaussian-tile pair들은 tile과 viewpoint별로 독립적으로 view-space depth 오름차순, 즉 front-to-back (앞에서 뒤로) 순서가 되도록 정렬된다. 이렇게 하면 각 tile과 각 viewpoint마다 하나의 Gaussian list (Gaussian 목록)가 생긴다. 이 목록들은 메모리에서는 하나의 unified Gaussian list (통합 Gaussian 목록) 안에 연속적으로 저장되고, tile ID와 viewpoint ID 순서로 배치된다. 최종적으로 각 subpixel은 자기 tile과 자기 viewpoint에 맞는 Gaussian list를 찾아, 올바른 depth order (깊이 순서)로 합성할 수 있다.

### Alpha Blending

Depth sorting이 끝나면 rasterizer는 각 tile 내부 subpixel을 row-major order (행 우선 순서), 즉 일반적인 raster order로 순회한다. 각 subpixel $(x,y,u)$에 대해 해당 tile index와 viewpoint index $\mathbf{V}[x,y,u]$에 맞는 splat sequence (스플랫 순서)를 가져온다. 그 sequence 안의 Gaussian 기여를 front-to-back alpha blending으로 누적해 최종 light-field image $I_{LF}$를 만든다.

하지만 이 naive subpixel-level pipeline (단순 서브픽셀 파이프라인)은 두 가지 병목을 갖는다.

1. 인접 시점끼리 비슷한 계산을 반복하는 redundant per-view evaluation (시점별 중복 계산)이 크다.
2. LFD의 interlaced subpixel layout 때문에 warp 안의 인접 thread가 서로 다른 Gaussian list를 읽어 uncoalesced memory access (병합되지 않는 메모리 접근)가 발생한다.

CoherentRaster는 이 둘을 각각 다른 전략으로 푼다. Section 4.2의 Cross-view Coherent Attribute Reuse는 projection, key generation, sorting 단계의 중복 계산을 줄이고, Section 4.3의 View-coherent Remapping은 alpha blending 단계에서 thread-to-subpixel mapping (스레드-서브픽셀 대응)을 재배치해 coalesced memory access (병합 메모리 접근)를 회복한다.

## 4.2 Cross-view Coherent Attribute Reuse

Cross-view Coherent Attribute Reuse (시점 간 일관 속성 재사용)는 인접 시점들 사이에서 부드럽게 변하는 projected Gaussian attribute (투영된 Gaussian 속성)를 재사용해 per-view computation (시점별 계산)을 줄이는 방법이다. 논문은 인접 viewpoint를 cluster (클러스터)로 묶고, cluster 내부에서는 일부 투영 속성을 공유한다. 이 clustering strategy (클러스터링 전략)는 projection, key generation, sorting 단계에 일관되게 적용되어 시점별 오버헤드를 크게 줄인다.

형식적으로는 전체 viewpoint 집합 $\mathcal{V}=\{v_i\}_{i=0}^{N-1}$를 $K$개의 서로 겹치지 않는 cluster $\{\mathcal{V}_0,\ldots,\mathcal{V}_{K-1}\}$로 균등하게 나눈다. 여기서 $K<N$이다. 각 cluster $\mathcal{V}_k$는 geometric center view (기하학적 중심 시점) $v'_k\in \mathcal{V}_k$로 대표된다. 이 방식은 원래 subpixel-level 3DGS rasterization이 감당해야 하는 per-view evaluation 비용을 줄인다.

### Projection

Gaussian의 2D mean은 viewpoint가 조금만 바뀌어도 눈에 띄게 변한다. 2D mean은 3D ellipsoid (3D 타원체)의 projected center (투영 중심)에 해당하기 때문에, 작은 시점 이동만으로도 image plane 위 위치가 움직인다. 이 위치 변화는 tile assignment (타일 배정)에 직접 영향을 준다. 따라서 잘못된 tile coverage (타일 포함 범위) 같은 geometric artifact (기하학적 오류)를 피하려면 각 Gaussian $G_i$의 2D mean은 모든 view $v_j$에 대해 독립적으로 계산해야 한다.

$$
\mu^{2D}_{i,j} = \Pi_{\text{mean}}(v_j; G_i)
$$

반면 2D covariance, depth, SH-based color (구면조화 기반 색상)는 인접 시점 사이에서 훨씬 완만하게 변한다. 이 값들은 주로 local surface orientation (국소 표면 방향)과 shading (음영)에 의해 결정되고, 작은 시점 변화에서는 급격히 달라지지 않는다. 그래서 논문은 이 smoothness (부드러운 변화)를 이용해 cluster representative view $v'_k$에서 한 번만 계산하고, 같은 cluster의 나머지 시점 $v_j\in \mathcal{V}_k\setminus v'_k$에 재사용한다.

$$
\Sigma^{2D}_{i,k} = \Pi_{\text{cov}}(v'_k; G_i),\quad
d_{i,k} = \Pi_{\text{depth}}(v'_k; G_i),\quad
c_{i,k} = \Pi_{\text{SH}}(v'_k; G_i)
$$

여기서 $\Sigma^{2D}_{i,k}$, $d_{i,k}$, $c_{i,k}$는 각각 $k$번째 cluster representative view $v'_k$에서 본 $i$번째 Gaussian의 2D covariance, depth, SH color를 뜻한다. 이후 단계에서 rasterizer는 viewpoint ID로 per-view 2D mean $\mu^{2D}_{i,j}$를 가져오고, cluster ID로 공유 속성 $\Sigma^{2D}_{i,k}$, $d_{i,k}$, $c_{i,k}$를 가져온다. 이렇게 섞어 쓰면 geometric accuracy (기하 정확도)에 중요한 mean은 시점별로 유지하면서, 상대적으로 완만한 속성은 cluster 단위로 재사용할 수 있다.

### Key Generation

Key generation에서는 Fig. 3(b)처럼 각 Gaussian에 대해 cluster 내부 시점 중 하나라도 겹치는 tile을 모두 포함한다. 즉 시점마다 Gaussian-tile pair를 따로 만드는 대신, cluster 단위의 Gaussian-tile pair를 만든다. 이로써 이후 처리해야 할 pair 수가 줄어든다.

그 다음 각 cluster 안의 Gaussian-tile pair마다 하나의 64-bit sorting key (64비트 정렬 키)를 붙인다. Gaussian $G_i$와 $t$번째 tile의 pair에 대해 key는 tile ID $t$, cluster ID $k$, depth $d_{i,k}$를 묶어 만든다.

$$
key_{i,t} = (t,k,d_{i,k})
$$

여기서 $d_{i,k}$는 cluster center view $v'_k$에서 투영된 Gaussian $G_i$의 depth다. per-view key generation과 비교하면 key가 붙는 Gaussian-tile pair 수가 크게 줄기 때문에, 전체 sorting workload (정렬 작업량)가 감소한다.

### Sorting

Key를 정렬하면 Gaussian들은 각 tile-cluster pair $(t,k)$에 해당하는 연속 list로 재구성된다. 각 list 안에서는 key 구조상 Gaussian들이 depth 순서로 정렬된다. Rasterizer는 특정 tile과 cluster에 대해 메모리 범위 $[S_{t,k}, E_{t,k})$를 통해 해당 list에 접근한다. 여기서 $S_{t,k}$와 $E_{t,k}$는 그 list의 시작과 끝 offset이다. List가 연속적으로 저장되어 있기 때문에 alpha blending 중 front-to-back traversal (앞에서 뒤로 순회)이 효율적으로 수행된다.

### Discussion

Cluster 단위 Gaussian-tile pair를 쓰면 key 수를 효과적으로 줄일 수 있지만, 부작용도 있다. 어떤 Gaussian이 cluster 안의 일부 view에서는 실제로 특정 tile에 보이지 않는데도, cluster 전체 기준으로는 그 tile에 배정될 수 있다. 하지만 논문은 sorting overhead 감소로 얻는 성능 이득이 이 추가 비용보다 훨씬 크다고 본다. 또한 최종 alpha blending 단계에서 각 Gaussian의 기여도는 mean, covariance, 실제 렌더링되는 subpixel coordinate (서브픽셀 좌표)를 기준으로 다시 정확히 평가된다. 따라서 불필요하게 포함된 Gaussian은 opacity (불투명도)가 거의 0에 가까워 자연스럽게 무시되고, 최종 품질은 유지된다.

## 4.3 View-coherent Remapping

View-coherent Remapping (시점 일관 재매핑)은 subpixel-level rasterization에서 발생하는 uncoalesced memory access 문제를 해결하기 위해 GPU thread가 subpixel에 배정되는 방식을 바꾸는 방법이다. 기존처럼 thread rank를 raster order의 spatial index (공간 인덱스)에 바로 대응시키지 않고, subpixel coordinate (서브픽셀 좌표)를 viewpoint index 기준으로 정렬한 뒤 thread에 배정한다. 이렇게 하면 하나의 warp 안의 thread들이 같은 시점 또는 가까운 시점의 subpixel을 처리하게 되고, viewpoint monotonicity (시점 단조성)가 생겨 coalesced memory access가 가능해진다.\

기존 tile-based rasterization은 공간적으로 인접한 subpixel들이 비슷한 데이터를 필요로 한다고 가정한다. 하지만 light-field display에서는 lens geometry (렌즈 기하) 때문에 서로 다른 viewpoint가 panel 전체에 interleave (교차 배치)된다. 그래서 같은 tile에 배정된 thread라도 viewpoint index가 서로 크게 다를 수 있고, 결국 tile과 cluster 기준으로 조직된 서로 다른 Gaussian list를 읽게 된다. 반대로 subpixel을 viewpoint 기준으로 먼저 묶은 뒤 thread에 배정하면, warp 안의 thread들이 같은 list 또는 가까운 list를 읽게 되어 bandwidth overhead (대역폭 오버헤드)가 줄어든다.

Fig. 3(c)는 viewpoint-sorted mapping (시점 기준 정렬 매핑)을 만드는 과정을 보여준다. 각 tile에 대해 먼저 subpixel coordinate $x$를 row-major order로 linearize (1차원화)한다. 그 다음 이 좌표들을 viewpoint index $\mathbf{V}[x]$ 기준으로 정렬한다. 정렬된 index는 lookup table (조회 테이블) $\Psi$에 저장된다. 렌즈 기하는 고정되어 있으므로 이 table은 한 번만 미리 계산하면 된다.

Rasterization 중 각 thread는 이 table을 통해 subpixel에 접근한다. 따라서 연속된 thread rank $r$과 $r+1$에 대해 viewpoint index가 다음 조건을 만족한다.

$$
\mathbf{V}[\Psi(r)] \le \mathbf{V}[\Psi(r+1)]
$$

이 순서는 인접 thread가 같은 시점 또는 인접 시점에 속한 subpixel을 처리할 가능성을 높인다. 이는 곧 동일하거나 인접한 cluster ID를 의미한다. Gaussian list는 tile ID 다음 cluster ID 순서로 정리되어 있으므로, thread들은 자연스럽게 같은 Gaussian list 또는 가까운 Gaussian list에 접근한다.

이제 mapping $\Psi$가 있으면 alpha blending kernel (알파 합성 커널)은 더 효율적인 메모리 접근으로 동작한다. Thread rank $r$은 먼저 실제 공간 좌표를 가져온다.

$$
\hat{x} = (x,y,u) = \Psi(r)
$$

그 다음 해당 subpixel의 viewpoint index $j=\mathbf{V}[\hat{x}]$를 얻고, $\hat{x}$와 $j$에서 tile ID $t$와 cluster ID $k$를 찾는다. 이후 $(t,k)$에 해당하는 Gaussian list의 메모리 범위 $[S_{t,k}, E_{t,k})$를 읽는다.

각 Gaussian $G_i$에 대해서는 Section 4.2에서 설명한 투영 속성을 이용해 기여도를 계산한다. 즉 mean은 시점별 값 $\mu^{2D}_{i,j}$를 쓰고, covariance와 color는 cluster 공유 값 $\Sigma^{2D}_{i,k}$와 $c_{i,k}$를 쓴다. 최종 subpixel intensity (서브픽셀 밝기)는 다음과 같이 누적된다.

$$
C(\hat{x}) =
\sum_{i\in \mathcal{N}}
c^{(u)}_{i,k}\alpha_i
\prod_{p=1}^{i-1}(1-\alpha_p)
$$

$$
\alpha_i =
o_i\cdot
\exp\left(
-\frac{1}{2}
(\hat{x}-\mu^{2D}_{i,j})^\top
(\Sigma^{2D}_{i,k})^{-1}
(\hat{x}-\mu^{2D}_{i,j})
\right)
$$

여기서 $\mathcal{N}$은 해당 Gaussian list 안의 정렬된 Gaussian 집합이고, $c^{(u)}_{i,k}$는 RGB channel $u$에 해당하는 color component (색상 성분)이며, $o_i$는 학습된 opacity다. 핵심은 viewpoint index $j$는 mean $\mu^{2D}_{i,j}$를 고르는 데 쓰이고, cluster ID $k$는 재사용되는 covariance $\Sigma^{2D}_{i,k}$와 color $c_{i,k}$를 고르는 데 쓰인다는 점이다.

Blending은 누적 opacity가 충분히 커지거나 list를 모두 소진하면 종료된다. 계산된 intensity $C(\hat{x})$는 light-field image $I_{LF}$의 해당 subpixel 위치에 바로 기록된다. 따라서 별도의 후처리 없이 interlaced output (인터레이스드 출력)이 만들어진다.

## 4장 핵심 요약

4장의 구조는 다음처럼 볼 수 있다.

```text
기존 단순 subpixel 3DGS:
모든 시점별 projection -> 시점별 Gaussian-tile key -> tile/view별 list -> raster-order thread blending

CoherentRaster:
시점별 mean + cluster 공유 covariance/depth/color
-> tile/cluster key
-> tile/cluster Gaussian list
-> viewpoint-sorted thread remapping으로 blending
```

가장 중요한 변화는 두 가지다.

1. 시점이 조금 달라도 비슷한 속성은 cluster 단위로 재사용해 projection/key/sort 비용을 줄인다.
2. Alpha blending thread를 공간 순서가 아니라 viewpoint index가 가까운 순서로 배치해 warp가 같은 또는 가까운 Gaussian list를 읽도록 만든다.
