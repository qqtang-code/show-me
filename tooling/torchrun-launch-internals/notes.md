# torchrun 启动流程与进程管理（文字版）

对应图解页：[index.html](index.html)。这份 notes 是纯文字版，便于检索与复制。

## 1. 启动流程

1. **参数解析 + `PET_*` 兜底**：每个 `--flag` 都注册了 `action=env`，命令行没给就读 `PET_<DEST>`（下划线、大写）。
   例如 `PET_NNODES=2 torchrun train.py` 等价于 `--nnodes=2`。
2. **默认值兜底**（`run()`）：
   - `--standalone` → 强制 `rdzv_backend=c10d`、`rdzv_endpoint=localhost:0`、`rdzv_id=uuid4()`；
   - 否则若 `rdzv_backend=static` 且无 endpoint 且 `--nnodes` 最大值为 1 → 同样改成 `c10d / localhost:0`；
   - `master_port is None` → 填 `29500`。
3. **算规模**：`parse_min_max_nnodes("1:4") -> (1,4)`；`determine_local_world_size` 支持 `auto/cpu/gpu/xpu/int`。
   `nproc_per_node > 1` 且未设 `OMP_NUM_THREADS` 时，torchrun 会设成 `1` 并告警。
4. **拼命令**：`cmd = os.getenv("PYTHON_EXEC", sys.executable)`，`cmd_args = ["-u", train.py, *剩余参数]`。
   默认路径**不注入 rank 参数**；仅 `use_env=False` 的老路径才追加 `--local-rank=<macro>`。
5. **构造 handler 与 agent**：`get_rendezvous_handler(RendezvousParameters)` → `WorkerSpec` → `LocalElasticAgent`。
6. **rendezvous**：agent 拿 `group_rank` / `group_world_size`。
7. **分配 rank**：`global_rank = group_rank * local_world_size + local_rank`（默认走 store 慢路径）。
8. **生成 worker 环境变量字典**（见下）。
9. **`Popen`**：entrypoint 是字符串 → `SubprocessContext` → 每 rank 一个
   `subprocess.Popen(args, env=os.environ.copy() | worker_env, start_new_session=True)`。

> `--start-method` 对 torchrun 命令行路径无效：只有 entrypoint 是**函数**时才走 `torch.multiprocessing`。

## 2. 注入的环境变量（`_start_workers` 的 `worker_env`）

`RANK`、`GROUP_RANK`、`ROLE_RANK`、`ROLE_NAME`、`LOCAL_WORLD_SIZE`、`WORLD_SIZE`、`GROUP_WORLD_SIZE`、
`ROLE_WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT`、`LOCAL_RANK`（由 `_set_local_rank_env` 填）、
`TORCHELASTIC_RESTART_COUNT`、`TORCHELASTIC_MAX_RESTARTS`、`TORCHELASTIC_RUN_ID`、
`TORCHELASTIC_USE_AGENT_STORE`、`TORCH_NCCL_ASYNC_ERROR_HANDLING`。

例外：`OMP_NUM_THREADS`（父环境已设则透传）、`TORCHELASTIC_TIMER_FILE`（仅 `TORCHELASTIC_ENABLE_FILE_TIMER=1` 时）、
`CUDA_VISIBLE_DEVICES`（传统模式原样继承；`--virtual-local-rank` 才切割并令 `LOCAL_RANK=0`）。

## 3. rendezvous

| backend | 裁判 | MASTER_ADDR/PORT 来源 |
|---|---|---|
| `static`（默认） | 你自己：`--node-rank` 作 rank，`--master-addr/--master-port` 作 endpoint，rank0 起 TCPStore | 直接用这两个参数 |
| `c10d` | rendezvous store 自身（TCPStore/FileStore），支持 `min:max` 弹性 | rank0 在 store 里写 `MASTER_ADDR/MASTER_PORT`，所有 rank 读同一份（`RendezvousStoreInfo.build`） |
| `etcd` / `etcd-v2` | 外部 etcd（需开 v2 API） | 同上 |

`static` 之外，`--master-addr / --master-port / --node-rank` 会被忽略并告警。

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

- 关闭：`killpg(pid, SIGTERM)` → 等 `shutdown_timeout`（默认 30s）→ `SIGKILL`。
- root cause：子进程把异常写进 `TORCHELASTIC_ERROR_FILE`（需 `@record`），agent 按时间戳取最早的一个当根因。
- **rank 不稳定**：重启/成员变更后可能被分到别的 rank 区间，`WORLD_SIZE` 也可能变。

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
        sampler.set_epoch(epoch)          # 忘了这句每个 epoch 分片顺序相同
        ...
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

`init_process_group` 不传 `init_method` 时默认 `env://`，从 `RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT`
四个环境变量建 store（缺一个就抛 `environment variable X expected, but not set`）。

## 6. 常见报错与排查

| 症状 | 优先排查 |
|---|---|
| `Address already in use` / TCPStore bind 失败 | 区分 `MASTER_PORT`(29500) 与 `--rdzv-endpoint`(29400)；`lsof -i :29500`；残留 `torchrun` 进程 |
| `Timed out after Ns waiting for clients. k/N clients joined` | 通常不是端口，是节点间网络/防火墙；`nc -vz <master> <port>` |
| `RendezvousTimeoutError` | 各节点参数是否一致（`--rdzv-id`/`--nnodes`）；rank0 地址是否可达；NTP 时钟同步；`--rdzv-conf=timeout=` |
| `environment variable RANK expected, but not set` | 脚本不是被 torchrun 启动的；或自定义 launcher 没透传环境 |
| 重启后 checkpoint 错乱 | rank 不稳定 + world_size 会变；checkpoint 记录 shard 方案；原子写 |
| 通信卡死无报错 | 集合通信次数不一致（数据相关控制流）；`NCCL_SOCKET_IFNAME` 指错网卡；`TORCH_NCCL_*` 超时设置 |
| 8 个进程都挤在 GPU0 | 漏了 `torch.cuda.set_device(LOCAL_RANK)` |
| `unrecognized arguments: --local-rank=N` | argparse 只注册了 `--local_rank`；应同时注册两种拼写 |

## 7. 参考

- 源码：`torch/distributed/run.py`、`launcher/api.py`、`elastic/agent/server/{api,local_elastic_agent}.py`、
  `elastic/rendezvous/{static_tcp_rendezvous,c10d_rendezvous_backend,dynamic_rendezvous}.py`、
  `elastic/multiprocessing/__init__.py`、`distributed_c10d.py`、`rendezvous.py`、`csrc/.../TCPStore.cpp`
- 官方文档：<https://docs.pytorch.org/docs/stable/elastic/run.html>
