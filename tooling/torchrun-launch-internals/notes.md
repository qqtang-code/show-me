# torchrun 启动流程与进程管理（文字版）

对应图解页：[index.html](index.html)。本文件是纯文字版，便于检索与复制。

> **术语约定**：翻译成中文会产生歧义的词一律保留英文原词 —— `rank`、`group`、`store`、`rendezvous`、`agent`、
> `worker group`、`elastic`、`quorum`、`complete`、`heartbeat`、`communicator`、`watchdog`、`checkpoint`。
> 变量名、函数名、参数名保持原样，以便与源码和报错信息逐字对照。

## 1. 启动流程

1. **参数解析 + `PET_*` 兜底**：每个 `--flag` 都注册了 `action=env`，命令行未提供时读取 `PET_<DEST>`（大写、下划线）。
   例如 `PET_NNODES=2 torchrun train.py` 等价于 `--nnodes=2`。
2. **默认值兜底**（`run()`）：
   - `--standalone` → 强制 `rdzv_backend=c10d`、`rdzv_endpoint=localhost:0`、`rdzv_id=uuid4()`；
   - 否则若 `rdzv_backend=static` 且未提供 endpoint 且 `--nnodes` 最大值为 1 → 同样改为 `c10d / localhost:0`；
   - `master_port is None` → 填 `29500`。
3. **算规模**：`parse_min_max_nnodes("1:4") -> (1,4)`；`determine_local_world_size` 支持 `auto/cpu/gpu/xpu/int`。
   `nproc_per_node > 1` 且未设置 `OMP_NUM_THREADS` 时，torchrun 会将其设为 `1` 并输出警告。
4. **拼命令**：`cmd = os.getenv("PYTHON_EXEC", sys.executable)`，`cmd_args = ["-u", train.py, *剩余参数]`。
   默认路径**不注入 rank 参数**；仅 `use_env=False` 的老路径才追加 `--local-rank=<macro>`。
5. **构造 handler 与 agent**：`get_rendezvous_handler(RendezvousParameters)` → `WorkerSpec` → `LocalElasticAgent`。
6. **rendezvous**：agent 取回 `GROUP_RANK` / `GROUP_WORLD_SIZE`。
7. **分配 RANK**：`RANK = GROUP_RANK * LOCAL_WORLD_SIZE + LOCAL_RANK`（默认走 store 的 slow path）。
8. **生成 worker 的环境变量字典**（见下）。
9. **`Popen`**：entrypoint 为字符串 → `SubprocessContext` → 每个 rank 一个
   `subprocess.Popen(args, env=os.environ.copy() | worker_env, start_new_session=True)`。

> `--start-method` 对 torchrun 的命令行路径不生效：只有 entrypoint 为**函数**时才走 `torch.multiprocessing`。

## 2. 注入的环境变量（`_start_workers` 的 `worker_env`）

`RANK`、`GROUP_RANK`、`ROLE_RANK`、`ROLE_NAME`、`LOCAL_WORLD_SIZE`、`WORLD_SIZE`、`GROUP_WORLD_SIZE`、
`ROLE_WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT`、`LOCAL_RANK`（由 `_set_local_rank_env` 填）、
`TORCHELASTIC_RESTART_COUNT`、`TORCHELASTIC_MAX_RESTARTS`、`TORCHELASTIC_RUN_ID`、
`TORCHELASTIC_USE_AGENT_STORE`、`TORCH_NCCL_ASYNC_ERROR_HANDLING`。

例外：`OMP_NUM_THREADS`（父环境已设置则透传）、`TORCHELASTIC_TIMER_FILE`（仅 `TORCHELASTIC_ENABLE_FILE_TIMER=1` 时）、
`CUDA_VISIBLE_DEVICES`（传统模式原样继承；仅 `--virtual-local-rank` 会切割并令 `LOCAL_RANK=0`）。

## 3. rendezvous

| backend | 裁判 | MASTER_ADDR/PORT 来源 |
|---|---|---|
| `static`（默认） | 由你指定：`--node-rank` 作为 rank，`--master-addr/--master-port` 作为 endpoint，rank 0 创建 TCPStore | 直接使用这两个参数 |
| `c10d` | rendezvous store 自身（TCPStore/FileStore），支持 `min:max` elastic | rank 0 把 `MASTER_ADDR/MASTER_PORT` 写入 store，所有 rank 读到同一份（`RendezvousStoreInfo.build`） |
| `etcd` / `etcd-v2` | 外部 etcd（需开 v2 API） | 同上 |

`static` 之外，`--master-addr / --master-port / --node-rank` 会被忽略并输出警告。

## 4. 容错与重启（`SimpleElasticAgent._invoke_run`）

```
_initialize_workers()            # rendezvous + env + Popen
while True:
    sleep(monitor_interval)      # 默认 0.1s
    state = _monitor_workers().state
    SUCCEEDED  -> _exit_barrier() 后返回
    FAILED/UNHEALTHY -> 有余额则 _restart_workers()（停全组 → 重新 rendezvous → 全组重开）
                        无余额则 _stop_workers() 并返回失败
    HEALTHY -> 若 rdzv_handler.num_nodes_waiting() > 0，主动 _restart_workers()（成员变更不计次）
```

- 终止流程：`killpg(pid, SIGTERM)` → 等待 `shutdown_timeout`（默认 30s）→ `SIGKILL`。
- root cause：子进程把异常写入 `TORCHELASTIC_ERROR_FILE`（需要 `@record`），agent 按时间戳取最早的一个作为根因（root cause）。
- **RANK 不稳定**：重启或成员变更后可能被分到别的 rank 区间，`WORLD_SIZE` 也可能变化。

## 5. 脚本侧最小适配

```python
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record

def setup():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",
                            device_id=torch.device("cuda", local_rank))
    return dist.get_rank(), dist.get_world_size()

@record
def main():
    rank, world_size = setup()
    model = DDP(MyModel().to(rank), device_ids=[rank])
    sampler = DistributedSampler(MyDataset(), num_replicas=world_size, rank=rank)
    loader = DataLoader(MyDataset(), batch_size=32, sampler=sampler)
    for epoch in range(N):
        sampler.set_epoch(epoch)          # 漏了这句，每个 epoch 的分片顺序都相同
        ...
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

`init_process_group` 不传 `init_method` 时默认使用 `env://`，从 `RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT`
四个环境变量创建 store（缺任意一个都会抛 `environment variable X expected, but not set`）。

## 6. 常见报错与排查

| 症状 | 优先排查 |
|---|---|
| `Address already in use` / TCPStore bind 失败 | 区分 `MASTER_PORT`(29500) 与 `--rdzv-endpoint`(29400)；用 `lsof -i :29500` 检查；清理残留的 `torchrun` 进程 |
| `Timed out after Ns waiting for clients. k/N clients joined` | 通常与端口无关，而是节点间网络 / 防火墙；用 `nc -vz <master> <port>` 验证 |
| `RendezvousTimeoutError` | 各节点参数是否一致（`--rdzv-id`/`--nnodes`）；rank 0 地址是否可达；NTP 时钟同步；`--rdzv-conf=timeout=` |
| `environment variable RANK expected, but not set` | 脚本不是由 torchrun 启动；或自定义 launcher 未透传环境变量 |
| 重启后 checkpoint 错乱 | RANK 不稳定 + world_size 会变化；checkpoint 中记录 shard 方案；使用原子写 |
| 通信 hang 住且无报错 | 集合通信次数不一致（依赖数据的控制流）；`NCCL_SOCKET_IFNAME` 指定错误网卡；`TORCH_NCCL_*` 超时设置 |
| 8 个进程都使用 GPU 0 | 漏了 `torch.cuda.set_device(LOCAL_RANK)` |
| `unrecognized arguments: --local-rank=N` | argparse 只注册了 `--local_rank`；应同时注册两种拼写 |

## 7. 参考

- 源码：`torch/distributed/run.py`、`launcher/api.py`、`elastic/agent/server/{api,local_elastic_agent}.py`、
  `elastic/rendezvous/{static_tcp_rendezvous,c10d_rendezvous_backend,dynamic_rendezvous}.py`、
  `elastic/multiprocessing/__init__.py`、`distributed_c10d.py`、`rendezvous.py`、`csrc/.../TCPStore.cpp`
- 官方文档：<https://docs.pytorch.org/docs/stable/elastic/run.html>
