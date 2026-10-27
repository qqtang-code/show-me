# SGLang prefix cache 的三类注意力：Linear · SWA · Sparse

> 源码基线：一份含 Unified Radix Cache 的 SGLang checkout（commit `16eb87c0`）。路径相对仓库根。
> 本文由读源码产出，**未做运行时实测**；带数字的结论都给了推导所依据的那几行。
>
> **术语约定**：保留源码英文标识（`ComponentType`、`component_data`、`tombstone`、`branching_seqlen`、
> `checkpoint`、`indexer`、`top-k`）。函数名、字段名、flag 名原样，便于直接 grep。

## 0. 速览

1. 不是三套缓存，是同一棵树上的三类组件。`UnifiedTreeNode` 每个节点带一份 `component_data[ct]`，
   `ct ∈ {FULL, SWA, MAMBA}`。线性注意力 → `MAMBA`；滑动窗口 → `SWA`；
   **稀疏注意力没有组件身份**，它的 indexer K 寄生在 `FULL` 的同一批 token slot 上。
2. 同一个 `.value` 字段，三种粒度：`FULL`/`SWA` 存 per-token 池索引（`len(value)` = 覆盖 token 数）；
   `MAMBA` 存**一个 slot id**（`len(value)==1`），因为状态里没有 token 轴。
3. 命中是**合取**：`_all_valid(validators, node)` 要所有组件同时通过，边界才前进。
   但 `full_kv_hit_length` 独立于校验器先行累加，所以「KV 命中、辅助组件没命中」会变成
   `branching_seqlen` 交回调度器，而不是直接浪费。
4. SWA 的复用是尾窗**全有或全无**：一条已解码的 5000-token 缓存序列，查询前缀长度 `L < 4999` 时
   可复用 **0** 个 token，只有 `L ≥ 4999` 才拿到完整的 `L`。
5. 稀疏注意力缓存的是 **indexer K**，不是 top-k 模式。模式每步重算，所以「上下文/长度/配置不同」
   在结构上都不可能让旧模式失效 —— 它从没被留下来。

## 1. 共同底座：一棵树，三类组件

| 层 | 职责 | 位置 |
|---|---|---|
| `UnifiedRadixCache` | controller：所有 pool / host I/O，排空树的延迟动作。「tree decides, cache executes」 | `mem_cache/unified_radix_cache.py:603` |
| `UnifiedTreeCore` | 树本体：`root_node`、`{ComponentType → TreeComponent}`、每组件一条 LRU 与尺寸计数 | `unified_cache/unified_tree_core.py:416` |
| `TreeComponent` | 每类注意力一个，实现 match / insert-overlap / commit / split / evict / lock 六组 hook | `unified_cache/components/{full,swa,mamba}.py` |

```python
# mem_cache/registry.py:165-169
tree_components = [ComponentType.FULL]
if ctx.is_hybrid_swa:
    tree_components.append(ComponentType.SWA)
if ctx.is_hybrid_ssm:
    tree_components.append(ComponentType.MAMBA)
params.tree_components = tuple(tree_components)
```

**节点形状**：`UnifiedTreeNode.component_data` 是按 `ComponentType`（int 枚举）下标索引的 list，
`ComponentData{value, lock_ref, metadata, host_value, host_lock_ref, session_ref, session_ids}`。

**key 里有什么**：`RadixKey` 只有 `token_ids / extra_key / cache_salt / is_bigram / limit`。
**没有注意力配置** —— 没有 `sliding_window_size`、没有 `index_topk`、没有稀疏模式。
匹配长度一律 `page_aligned`。

### 命中判定：合取 + 两个独立进度

```python
# unified_cache/unified_tree_core.py:932-970（重排为逻辑序）
def _all_valid(validators, node):
    return all([v(node) for v in validators])       # 合取

def _update_best_if_valid(node):
    if _all_valid(validators, node):
        best_match_node = node                      # 边界只在这里前进

while key_offset < len(key) and child_key in node.children:
    child = node.children[child_key]
    if child.evicted and not child.backuped:
        break                                       # FULL 缺失：整条匹配直接断
    prefix_len = child.key.match_at(key, key_offset, page_size=self.page_size)
    full_kv_hit_length += prefix_len                # 先累加，早于任何校验器
    if prefix_len < len(child.key):
        node, action = self._split_node(child.key, child, prefix_len)
        _update_best_if_valid(node); break
    node = child
    _update_best_if_valid(node)
    key_offset += prefix_len
```

- `best_match_node` → 决定辅助组件复用到哪（LRU 提升、加锁、COW 源）。
- `full_kv_hit_length` → 纯 token 比较进度，可深于前者。
- 于是恒有 `full_kv_hit_length ≥ device_prefix_len`，差额被包装成 `swa_branching_seqlen` /
  `mamba_branching_seqlen` —— 语义是「这里本来可以命中，请你补一个快照点」，不是复用许可。

### 锁与逐出优先级

| 组件 | 加锁 | internal 逐出优先级 | 级联 |
|---|---|---|---|
| `FULL` | 路径锁 root→node | 2（最后死） | — |
| `SWA` | 窗口锁，到窗口边界并盖 uuid | 1 | 级联 MAMBA |
| `MAMBA` | 单点锁（`best_match_node`） | 0（最先死） | 不级联 |

leaf 节点三者优先级一律 0，逐出即整叶删除。`_cascade_evict` 规则：同一节点上 priority ≤ 触发者的组件一起走。

## 2. Linear Attention：一个 checkpoint = 一个节点的状态

所有线性注意力家族（GDN / KDA / Mamba2 / glm5_next / nemotron_h …）都经
`uses_ssm_state()` → `registry.py:166-169` → `ComponentType.MAMBA`。
家族之间只差状态形状参数，**缓存层一行不用改**。

### 2.1 存什么

状态**不在 KV 池**，挂在 `HybridReqToTokenPool` 的 `MambaPool` 上：

| 张量 | 形状 | 是什么 |
|---|---|---|
| `conv` | `[L, size+1, dim, K-1]` | 短卷积滑动窗口状态（KDA/GDN 把 K 放另一轴） |
| `temporal` | `[L, size+1, H, d_v, d_k]` | recurrent / SSM 状态本体 |
| slot 0 | 保留 | padding token 的写入目标 |
| `MambaCheckpointPool` | int8 + per-(head,k) scale | 可选 radix 侧副本池 |

请求持有 `req.kv.mamba_pool_idx`，shape `(1)`。`mamba_cache_per_req` 公式里**没有长度项** ⇒
成本 **O(1)/序列**，树成本 O(#节点)。

树上存的就是这个 id：`component_data[MAMBA].value` 是 shape `(1)` 的张量；
尺寸账目记的是「几个状态」不是「几个 token」（`mamba.py:237-239`）。

### 2.2 复用判定

```python
# components/mamba.py:149-160（完整函数）
def create_match_validator(self, match_device_only: bool = False):
    ct = self.component_type
    if match_device_only:
        return lambda node: node.component_data[ct].value is not None
    return lambda node: (
        node.component_data[ct].value is not None
        or node.component_data[ct].host_value is not None
    )
```

没有任何长度运算。「这个节点有状态」在构造上等于「这个深度是合法状态边界」，靠三条约束维持：

1. insert 的 key 被截断到状态的 seqlen（`unified_radix_cache.py:1152-1161` 对组件返回值取 `min()`）。
2. 发布深度必须落在 `mamba_checkpoint_grid = lcm(mamba_cache_chunk_size(), page_size)` 上
   （`mamba.py:80-84`、`runtime_context.py:1938-1951`）。
3. kernel 只在 chunk 对齐位置 materialize（`hybrid_linear_attn_backend.py:445-449`）。

另有硬约束：关掉 extra buffer 时 `page_size` 必须是 1（`mamba.py:75-78`）。

两个方向都被显式命名：

- `mamba_last_track_seqlen` —— 往回截。状态可能超出要发布的 key 长度，此时回退到 ping-pong 的另一个
  slot；两个都超就返回 `None`（docstring：`None means donate nothing`）。
- `mamba_branching_seqlen` —— 往前补。取 `full_kv_hit_length` 内最后一个网格对齐点，
  由 `schedule_batch.py:3210-3225` 在本次前向强制补一个 track 点。

### 2.3 更新

```text
① decode：状态在 ping-pong 两 slot 间轮转，只在 track 间隔上换
    batch_result_processor.py:1394-1396 / :1634 / :1559
② 结束 → donate（捐 slot，不拷）；未完（chunk 化）→ fork（copy_from）
    mamba.py:609-662
③ 命中 → COW：只记录 mamba_cow_src_index，在前向流上执行
    mamba.py:194-223 · model_runner.py:1912-1957
```

插入重叠时什么都不消耗（`update_component_on_insert_overlap` 未覆写）；插入落到已有且带状态的节点时
树保留自己的状态并置 `result.mamba_exist = True`；节点分裂时父节点拿 `None`。
逐出打 tombstone、优先级 0、不级联；`--mamba-max-states-per-path` 剪掉浅层线性中间态（soft limit）。

> README 与代码注释写「mamba = leaf-only data」，但 `commit_insert_component_data` 对任何 insert 尾节点
> 都会挂状态，`_evict_excess_path_states` 还会沿**祖先**找持有者。准确说法是「状态属于**深度命名它**的
> 那个节点」，不是「只能在叶子上」。

### 2.4 场景

| 场景 | 行为 |
|---|---|
| 完全相同前缀 | 只触达 `best_match_node` 一个节点的状态；请求拿 COW slot |
| 部分重叠 | 停在最深带状态节点，更深处重算 + 给 branching 建议 |
| 分叉 | 状态按节点存，两个兄弟就是两个状态；soft cap 保留分叉 |
| 并发 | N 请求 = N 个 COW slot（状态会被原地改写） |
| chunk 化 | 未完成 insert 走 fork，track seqlen 归 `None` |
| chunk < 网格 | mask 为假 → 完全不产生快照（启动 warning） |

**为什么命中长度不能等于 token 数**：状态没有 token 轴、是单个 slot id、且被 decode 原地改写
（必须 COW）。所以「能省多少 token」与「从哪个状态起步」是两个数，组件两个都保留。

## 3. Sliding Window Attention：窗口之外全是 tombstone

三条互斥路径，复用语义完全不同：

| 路径 | 判定 | 实现 |
|---|---|---|
| hybrid full+SWA | `is_hybrid_swa and full_tokens_per_layer > 0` | `UnifiedRadixCache` + `SWAComponent` |
| pure SWA | `is_hybrid_swa and full_tokens_per_layer == 0` | `PureSWARadixCache`，无组件、无 tombstone |
| per-request ring | `is_swa_req_ring(...)` / `request_window` | `swa_gates_prefix_match=False`，SWA 不 gate |

### 3.1 存什么

`SWAKVPool` 不是一个池切两半，是**两个独立池 + 一张翻译表**：

```python
# mem_cache/swa_memory_pool.py:96-107
self.full_kv_pool = full_kv_pool_class(size=size, layer_num=self.full_layer_nums, ...)
self.swa_kv_pool  = swa_kv_pool_class(size=size_swa, layer_num=self.swa_layer_nums, ...)
```

- `size_swa = full_tokens × swa_full_tokens_ratio`（默认 0.8，**按 token 打折，与 swa:full 层数无关**）。
- `full_to_swa_index_mapping`：与 full 池等长的 int64 表，未映射 = 0 = padding 哨兵。
- `ComponentData[SWA].value` 存的是**翻译后的 SWA 池索引** —— 这是它和 `FULL` 在数据层最实际的区别。
- 窗口锁边界不是长度，而是盖在边界节点上的 uuid，存在 `metadata["uuid"]` / `["host_uuid"]`
  （README 叫它 `component_uuid`，代码键名是 `"uuid"`）。

**tombstone**：逐出 SWA 数据时不清节点，`cd.value` 置 `None`。节点是否继续存在只由 FULL 决定
（`_iteratively_delete_tombstone_leaf`）。于是会出现「路径完整、Full KV 齐全、SWA 值缺失」的结构。

### 3.2 复用判定

```python
# components/swa.py:365-375
def validator(node: UnifiedTreeNode) -> bool:
    cd = node.component_data[ct]
    if cd.value is None and (match_device_only or cd.host_value is None):
        state["len"] = 0
        if not gates_prefix_match and (node.backuped or not node.evicted):
            return True
        return False
    state["len"] += len(node.key)
    return state["len"] >= sliding_window_size
```

- `state` 每次 match 新建，累加贯穿整条走查，**只在 tombstone 处归零**。
- 累加的是 `len(node.key)` 而不是 `len(cd.value)` —— 一个覆盖 5000 token 的节点，其 SWA 值可能只有
  最后 129 个位置，但校验器按 5000 计。这是「Full 全命中 + SWA 只有尾窗」能同时成立的原因。
- 累计 `< W` 时返回 `False`，边界不前进 ⇒ **0 复用**（走查继续，但边界没动）。
- ring 模式反向：`gates_prefix_match=False` 时 tombstone 判 `True`。

判据写成一行：设 `B` = 边界上溯最近的 device tombstone 右端，则**边界 `L` 合法 ⟺ `L − B ≥ W`**。

单测钉死了短前缀行为（`test_rust_tree_core_integration.py:3743-3746`）：`window=8`、插入 12 token、
SWA 游标 8，match 前 8 token → `full_kv_hit_length == 8` 但分页 SWA 下 `last_device_node` 停在 root，
**Full 命中 8 个、复用 0 个**。

SWA 还产出 `swa_host_hit_length` 与 `swa_branching_seqlen`；**不覆写
`finalize_match_result_in_cache`**，所以 `device_prefix_len` 不会被二次裁剪。

### 3.3 更新

```python
# mem_cache/common.py:111-120
evict_threshold = pre_len - ceil_align(
    max(sliding_window_size, page_size), page_size
)
```

这个阈值是枢轴：一边决定释放哪些 slot，一边决定 insert 时把节点切在哪。
记 `Wp = ceil_align(max(W,P), P)`，稳态游标 `B = pre_len − Wp`。
insert 时 `_clip_to_window` 先断言游标 page 对齐，再在 `window_start` 处 `_split_node`
（`swa.py:434-447`），结果是「父 tombstone、子 live」。新叶还会被
`_maybe_split_leaf_for_swa_lock` 把 SWA 值上限压到一个 page 对齐窗口，经 `SWARebuild` 由 controller 重建。

tombstone 不是永久的。`update_component_on_insert_overlap`（`swa.py:454-513`）三种结局：

| 分支 | 条件 | 动作 |
|---|---|---|
| 全恢复 | 整段在窗口内 | `consumed_from = 0`，树的 Full 值**换成新请求的 KV 索引**，旧值释放，再 `SWARebuild` |
| 部分恢复 | 只有窗口内那部分 | 返回 `live_start − total_prefix_len`，树按返回值释放未消费的重复索引 |
| Full 被锁 | `full_cd.lock_ref > 0` | 发 `RecoverSWAWithLockedFull`，不动已锁的 Full 值 |

设计取向：tombstone 恢复不是重算，而是**去重** —— 把新请求刚算出的 KV 收编进树，再从它重建 SWA。
逐出上内部节点 tombstone、叶子整叶删除，优先级 1，级联 MAMBA 不动 FULL。

### 3.4 定量：一条已解码的 5000-token 缓存序列

`C = 5000`、`W = 128`、`P = 1` ⇒ `Wp = 128`、`B = 4999 − 128 = 4871`，live 段 `[4871, 5000)` 共 129 token。

| 查询前缀长度 L | FULL 可复用 | hybrid SWA 可复用 | 原因 |
|---|---|---|---|
| `L ≤ 4871` | L | **0** | 落点在 tombstone 内，累计归零后永远 < 128 |
| `4871 < L < 4999` | L | **0** | `L − B < W`，切点处 live 段不够一窗 |
| `L = 4999` | 4999 | 4999 | `L − B = 128 ≥ W` |
| `L = 5000` | 5000 | 5000 | 整条命中，SWA 只需尾窗 live |

通式 `max_reusable = 0 if L < B + W else L`，`B + W = C − 1 − (Wp − W)`，
阈值 `T = C − 1 − (Wp − W) ∈ [C−P, C−1]`：**能复用的前缀只可能落在最后一个 page 之内**。

> 「SWA 只保留最后 128 个」说的是 SWA **层**的池子只留尾窗（`Wp+1 ≈ 129`），Full **层**留着全部 C。
> 而复用边界是整条链上的一个数 `device_prefix_len`，不能一半一半 —— 所以短前缀的结局是**一个都不复用**。

另一种形态：若树里先前插过一个 1000-token 请求，它在 `[872,1000)` 留下 live 窗，则 5000-token 查询的走查是
`[0,872)` 归零 → `[872,1000)` 累计 128 有效、边界到 1000 → `[1000,4871)` 又归零但边界不回退，
结果 `device_prefix_len = 1000`。这就是「停在更深的那个窗口沿」。

## 4. Sparse Attention：缓存的不是稀疏模式

这类注意力**在树里没有身份**：没有 `ComponentType.SPARSE`，没有稀疏专用校验器，
`tree_components` 里也找不到它。原因是 **SGLang 缓存 indexer K，从不缓存 top-k 选择**。

### 4.1 存什么

```python
# mem_cache/index_key_cache.py:21-47（节选）
class IndexKeyCache:
    def __init__(self, pool: DSATokenToKVPool, index_buf_size: int):
        num_pages = (index_buf_size + pool.page_size + 1) // pool.page_size
        self.buffer = [ torch.zeros(self._buffer_shape(...),
                        dtype=pool.index_k_with_scale_buffer_dtype, ...)
                        for i in range(pool.layer_num) ]
```

| 项 | 值 |
|---|---|
| 张量 | 每层一个二维 **uint8** 张量：行 = 物理 page，列 = `index_page_size × (K + scale)` |
| 每 token 每层 | `index_head_dim(128) + 128/128×4 = 132` 字节 |
| 粒度 | 按 page 分配、按 token loc 寻址；`index_kpool` 是池化因子 |
| 地址空间 | **与主 KV 共用**：`index_buf_size` 由 KV `size` 推导，`move_kv_cache` 同步搬 |
| HiCache 声明 | `indices_from_pool=PoolName.KV`、`layout_source=PoolName.KV` |
| skip-topk 层 | 复用上层 top-k，拿 0 行占位缓冲，无 host 层 |

「寄生」的后果：树上记住 KV 的 token 索引就顺带记住了 index-K 的位置，一次插入/逐出/搬迁同时覆盖两者；
代价是树上**没有任何字段或断言**记录「这一页的 index-K 已写过」，一致性靠结构不变量。

### 4.2 复用判定

复用 `FULL` 的校验器（`full.py:107-118`），**不检查 index-K 是否存在**。另有两处佐证：

- key 不含稀疏配置：树哈希只吃 token ids（+ `extra_key`/`cache_salt`），`index_topk` /
  `index_kpool` / `index_topk_pattern` 一律不进 key（`utils.py:116-123`、`166-174`）。
- 唯一真按页校验 index 存在性的是 L3：`PoolHitPolicy.ALL_PAGES`，可用前缀 clamp 成
  `min(KV pages, indexer pages)`（`hicache_storage.py:95`、`665-690`）。

**为什么可以这么松**：indexer K 是每个 token 自己的投影，与上下文无关；top-k 是
`query × 全部历史 key` 的函数，天然随上下文变。缓存前者、现算后者，
就把「失效」从设计里删掉了。

### 4.3 更新

| 时机 | 发生什么 |
|---|---|
| 新 token | 同一次 forward、同一批 `out_cache_loc` 写完 KV 再写 index-K；无独立提交点、无 dirty 标记 |
| 复用前缀 | index-K 恢复，top-k 用新 query **重算**（`dsa_indexer.py:1536`） |
| 短上下文 | `seq_len ≤ index_topk` 直接跳过 logits 与 top-k，合法状态 |
| 淘汰 | `IndexKeyCache` **没有 evict**；页随主 KV loc 释放而失效 |
| HiCache 卸载 | `cpu_copy` 显式同步搬 index-K |
| 布局切换 | fp4/fp8/page8 **不写进 key**，只能切布局前清库 |

`index_topk_pattern` / `index_topk_freq` / `index_skip_topk_offset` 决定的是**哪些层复用上一层的 top-k**
（从而连 indexer 都不构造），判定顺序在 `dsa_layer_skips_topk()`（`model_config.py:410-443`）。
全是部署级配置，改它们不触碰缓存内容。

### 4.4 场景

| 场景 | 结果 |
|---|---|
| 完全相同前缀 | index-K 逐字节复用；top-k 在新上下文重算 |
| 部分重叠 | 标准 radix 匹配，命中段内 index-K 一并复用 |
| 前缀长度变了 | 无位置漂移问题 —— 旧模式没被读过 |
| 稀疏配置不同 | 不影响缓存：不进 key/树/存储键 |
| skip-topk 层 | 无 index 缓冲也无所谓，top-k 来自同一步 anchor 层 |
| CP prefill | `dsa_cache_layer_split.py` 是层切分路由，对树透明 |
| HiCache 部分命中 | clamp 到 `min(KV, indexer)` |

六个稀疏家族里只有 DSA 这一支真正接进 radix 复用路径；唯一有组件身份的是 DeepSeek-V4 的
`C128` sidecar（`registry.py:171-181`），它有独立的 value / validator / 逐出优先级。

## 5. 三类机制的主要区别

| 问题 | Linear（MAMBA） | SWA（SWA） | Sparse（无组件） |
|---|---|---|---|
| ① 存什么 | 一个 slot id，指向 `conv` + `temporal` 两块状态；O(1)/序列 | per-token 的 SWA 池索引（需翻译）；窗口外是 tombstone | indexer K + scale，与主 KV 同 loc 空间；树上无独立字段 |
| ② 怎么判命中 | `value is not None`；合法边界由 `lcm(chunk,page)` 网格保证 | 累加 `len(node.key)`，遇 tombstone 归零；合法 ⟺ 连续 live 段 ≥ W | 复用 `FULL` 的校验器；不检查 index-K 存在性 |
| ③ 怎么更新 | ping-pong 轮转；donate / fork；命中 COW | 滑窗推进游标 → insert 在游标处切节点 → 新叶上限一窗；tombstone 可恢复 | 同 forward 同 loc 写入；top-k 重算；无独立淘汰 |
| ④ 场景差异 | 相同前缀 1 次 COW；部分重叠停在最深带状态节点 | 相同前缀整段可复用；严格前缀 0；短于一窗恒 0 | 所有场景都安全 |
| 加锁 | 单点锁 | 窗口锁 + uuid | 随 FULL 的路径锁 |
| 逐出优先级 | 0，不级联 | 1，级联 MAMBA | 跟 FULL（2）同生共死 |
| 粒度单位 | 状态（slot） | token（page 对齐） | token（page 对齐，随 KV） |

三条最本质的差异：

1. **负载有没有 token 轴** —— KV 与 index-K 有，任何 token 下标都是合法切点；循环状态没有，
   只有网格对齐位置合法，于是引入 `branching_seqlen` 这种「建议补快照」机制。
2. **负载是可变还是只增** —— KV append-only，可只读共享；状态被原地改写，必须 COW，
   并发成本 O(并发请求数)。
3. **「值」是连续还是离散** —— SWA 的值会随窗口滑动大片消失，需要 tombstone 表达「节点在、值没了」；
   稀疏的值与 KV 同生共死，不需要第三种状态，所以也不需要组件身份。

## 6. 以 GLM5 系列的推理路径为例

两个前提先纠正：

- **名字**：`GLM5.3flash` 这个字符串在 checkout 里**不存在**。对应架构是
  `Glm5NextForConditionalGeneration`。下文用架构名。
- **这个模型没有 SWA**：它不在 hybrid-SWA 白名单（`model_config.py:2548-2575`），
  glm5 代码里也没有 `sliding_window`；注册的组件是 **`(FULL, MAMBA)` 两个**。

### 6.1 层划分

```python
# configs/glm5_next.py:198-224（节选）
if linear_attn_config is None:
    layer_types = kwargs.get("layer_types")
    if layer_types is None:
        kda_layers = [i for i in range(num_hidden_layers) if i % 4 != 3]
    else:
        kda_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
```

默认 `num_hidden_layers = 45` ⇒ **34 个线性注意力层 + 11 个 full-attention 层**。
这只是兜底推导；真实切分来自 checkpoint 的 `layer_types` / `linear_attn_config`，
本 checkout 没有该 config，**逐层数量记为未确认**。

之后唯一真源是 `is_kda_layer()`（`configs/glm5_next.py:243-247`）：

```python
# models/glm5_next.py:707-720
self.is_linear_attn = config.is_kda_layer(layer_id)
if self.is_linear_attn:
    self.self_attn = Glm5NextLinearAttention(...)     # 走 MAMBA 组件
else:
    self.self_attn = DeepseekV2AttentionMLA(...)      # 走 FULL；带 indexer 时即 DSA 稀疏
```

> **命名陷阱**：`Glm5NextDecoderLayer._is_layer_sparse()`（`models/glm5_next.py:947-952`）看的是
> `n_routed_experts` 与 `moe_layer_freq` —— 它是 **MoE 与 dense MLP 的选择器**，与稀疏注意力无关。
> 真正的稀疏开关是 `is_deepseek_dsa(config)`（`model_config.py:190-204`）。

| 层种类 | 缓存什么 | 在哪 | 谁复用 |
|---|---|---|---|
| KDA 线性层（多数） | 卷积状态 + recurrent 状态 | `MambaPool` | 组件 `MAMBA`：单点锁 + COW |
| full-attention 层 | MLA latent KV | 主 KV 池 | 组件 `FULL`：路径锁 |
| full 层的 indexer | indexer K + scale（uint8） | 同主 KV 池的 page、同一批 loc | **组件 `FULL`（无独立身份）** |

### 6.2 一个请求的旅程

```text
Req.init_next_round_input(tree_cache)                       schedule_batch.py:1707
└─ match_kv_cache(..., cow_mamba=tree_cache.supports_mamba())
   └─ UnifiedRadixCache.match_prefix                       unified_radix_cache.py:603
      └─ UnifiedTreeCore.match_prefix                      unified_tree_core.py:867
         ├─ 每组件 create_match_validator() → 2 个：FULL, MAMBA
         ├─ 走查：_all_valid(...)；MAMBA 缺只停边界，FULL 缺直接 break
         ├─ finalize_match_result_in_tree_core → mamba_branching_seqlen
         └─ _match_post_processor → LRU 提升 / device_prefix_len
      └─ finalize_match_result_in_cache → MAMBA 分配私有 slot，记 COW 源
   └─ schedule_policy.py:1625  req.lock = tree_cache.lock(req.last_node)

（batch 构建）收集 COW/clear 列表 → model_runner.py:1912 在前向流上执行
（逐层前向）KDA 层走 linear-attn backend；full 层走 DSA MLA，稀疏层每步重算 top-k
（每 chunk）checkpoint_kv_cache → checkpoint_into_tree：prepare → min() → page_align → insert → 重 match → 转移锁
（结束）release_kv_cache → checkpoint(up_to=owned_len) → 解锁 + 释放
```

**最关键的不对称**：

- **MAMBA 缺 → 只让边界停住**。校验器返回 `False`，走查继续，`full_kv_hit_length` 继续累加，
  差额变成 `mamba_branching_seqlen`；KV 前缀照样复用，状态在这次 extend 里补一个快照点。
- **FULL 缺 → 整条匹配断**（`unified_tree_core.py:954-956` 的 `break`）。没有 KV 就无法前向，
  所以 FULL 的逐出优先级最高（2），它一死整条路径都不必谈复用。

### 6.3 四个场景

| 场景 | 发生什么 |
|---|---|
| 共享 system prompt | FULL 路径复用 + MAMBA 每请求 COW + index-K 顺带复用且 top-k 重算，三件事同时成立 |
| 多轮续写 | KV 侧几乎全中；KDA 侧停在最深网格对齐状态，尾巴重算，差额由 `mamba_branching_seqlen` 在本次 extend 补成新 track 点 |
| 「窗口命中但状态过期」 | 本模型没有 SWA，该情形不存在。等价真实情形是 `full_kv_hit_length > mamba_boundary_len` 与 MAMBA 单独被逐出 |
| 并发共享前缀 | FULL 路径锁叠加，MAMBA 各自 COW；池不足时先加锁、`evict_for_alloc(EvictParams(num_tokens=0, mamba_num=1))`、再分配、再解锁 |

想在这个模型上理解 SWA，唯一沾边的是 KDA 自身的短卷积窗口（`linear_conv_kernel_dim = 4`）——
那是状态内部的一个 4 步滑动窗口，不是一个可以被前缀缓存单独命中的 KV 区域。
真正的 SWA 前缀缓存只出现在 `DeepseekV4*` / `Gemma4*` / `Step3p5*` / `Inkling*` 这类
`is_hybrid_swa` 架构上。

## 7. 常见误解

| 误解 | 纠正 |
|---|---|
| 「SWA 的 prefix cache 就是复用最后 128 个 token」 | 链上的复用边界是一个整数 `device_prefix_len`，不能一半一半。5000-token 序列上长度 4000 的前缀请求拿到 0，不是 128 |
| 「稀疏注意力会缓存 top-k 选择，命中时要检查模式兼容」 | 不存在这个检查，因为模式从没被保存。缓存的是 indexer K，top-k 每步重算 |
| 「线性注意力的状态也占 KV 池槽位，命中长度就是 token 数」 | 状态挂在 `MambaPool` 上，`value` 是 shape `(1)` 的 slot id |
| 「三类机制各有一套独立的缓存树」 | 只有一棵树，`component_data` 是按 `ComponentType` 下标索引的 list；稀疏连这条线都不占 |
| 「命中失败就等于这部分完全白算」 | `full_kv_hit_length` 早于校验器累加，会变成 `branching_seqlen`，让本次前向顺手补一个快照，下一次能命中 |

## 8. 自测题

1. hybrid full+SWA，`W=256`、`page_size=16`，已解码的 8000-token 序列，3000-token 严格前缀能复用多少？
   → **0**。`Wp=256`，`B = 7999 − 256 = 7743`，`3000 − 7743 < 0`；阈值 `T = 7999`。
2. 同上，若 SWA 不 gate（ring 模式）？ → **3000**，回落到 Full 语义。
3. 为什么 `prepare_for_caching_req` 的返回值要取 `min()`？
   → 它决定树能发布到多长；MAMBA 返回状态实际加密的 seqlen，取 `min()` 保证「节点深度」与
   「状态 seqlen」是同一个数。不这么做，节点会声称自己在某个深度有状态而实际上没有。
4. index-K 布局从 fp8 换成 fp4，同一份 L3 缓存能命中吗？
   → 不能，且代码不会拦住你：布局不写进 key，只能靠切布局前清库。
5. 为什么 MAMBA 内部节点优先级是 0 且不级联？
   → 状态只在匹配边界节点被消费，内部节点上的状态对任何路径没贡献；而 FULL 一死整条路读不了，所以 2。
6. FULL 与 SWA 都命中、MAMBA 只到一半深会发生什么？
   → 合取 + 差额建议：边界停在 MAMBA 能命中的深度，`full_kv_hit_length` 记到更深，
   `mamba_branching_seqlen` 让本次前向补 track 点。

---

**未确认项**：GLM5 的逐层 KDA/full 切分；若干数值型 `index_*` 取值；
`swa_branching_seqlen` 在非 PD 路径是否触发额外重算；从未 decode 就结束的请求其 cursor=0 时的窗口上限；
MAMBA 状态能否出现在内部节点（本页按代码行为描述，README 用词与之不符）。
