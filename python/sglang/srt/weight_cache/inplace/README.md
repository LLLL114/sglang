# Same-fleet Qwen3 MoE weight transfer

`bind_weights` is an explicit stopped-epoch helper for the same eight existing
NVIDIA GPU workers. It does not implement production scheduler reconfiguration,
admission control, distributed rollback, or a second target fleet. The existing
immutable daemon transfer path is unchanged.

The source is a loaded BF16, unquantized `Qwen3MoeForCausalLM` using ordinary
gate-first Triton experts, without EPLB, redundant experts or fused shared
experts. Supported `(attention DP, expert EP)` layouts are `(1,1)`, `(1,2)`,
`(1,4)`, `(1,8)`, `(2,8)` and `(8,8)` with global TP eight. Pipeline, attention
context and MoE data parallel sizes must be one. DP LM-head mode must remain
unchanged. The helper checks a single hostname, eight distinct CUDA device UUIDs
and global-rank-ordered workers. Source parameters must share the current device,
be nonempty, contiguous and individually fit the 1 GiB arena cap. Tied/overlapping storage
and unsupported kernel layouts are rejected by the shared layout accounting.
Both source and target resolved arguments must retain trivial expert placement,
disabled EPLB, zero redundant experts and no elastic EP backend/join mode. Missing
source-history fields are rejected. Native trivial expert-location metadata is
supported only when the module's complete layer mapping is an identity with equal
logical and physical expert counts. Local dispatch projections may belong to the
target EP context; they do not replace this full-map check for the old model.

## Ownership and call sequence

Before calling on every worker, the application must stop admission, drain all
requests and weight transfers, drop KV/request state, release graphs and stale
parameter views, and stop new runtime-manifest snapshots. All previously issued
snapshots and allocation leases must finish and be released. Invalidating a lease
does not complete already submitted DMA. If supplying an engine, first drain its
users and close old registrations and cached peer imports collectively.

Retain the old model, rebuild the target process groups and construct the target
model with meta parameters. Target buffers must be corresponding meta buffers or
the same local source objects. No target GPU weights are allocated during normal
serving. Both models must retain the correct native module group metadata.
Module types, configuration, kernel backends and process-group metadata must stay
fixed throughout the call. The helper only replaces registered tensors and packs
their bytes; the caller must prevent concurrent metadata changes or hooks that
change these semantics.

```python
import os

from sglang.srt.weight_cache.inplace import bind_weights

os.environ["MC_INTRANODE_NVLINK"] = "1"
result = bind_weights(
    old_model,
    target_meta_model,
    old_server_args,
    target_server_args,
    global_rank,
    group=cpu_world_group,
)
```

Arguments are the current SGLang `ServerArgs`; the helper reads their resolved
attention-DP and EP settings. The CPU group contains exactly the eight workers in
global-rank order. All workers must enter the same transition and collective
sequence. A local error, including preflight failure, requires the caller to abort
participating peers; this helper supplies no distributed fault-tolerance protocol.
The default engine uses `nvlink_intra` and requires the companion Mooncake APIs
`InplaceWeightResharder`, Reader executor injection and
`reset_reshard_peers_quiescent`.

After success, install the target model, rebuild execution/KV/graph state and the
authoritative runtime-manifest manager, validate inference and collectively reopen
admission and snapshot access. Restoring old source bytes does not validate old
graphs or old physical bindings.

## Layout, memory and completion

Canonical checkpoint identities come from `Qwen3MoeWeightAdapter` and
`LayerLayoutBuilder`; model-specific semantics live in SGLang. Meta layout plans
describe shapes only. Physical runtime-manifest export continues to reject meta
storage. `WeightLayoutPlan.views` also provides checkpoint keys, global slices and
parameter byte offsets for a streaming checkpoint reference loader.

Private wire manifests derive grouped expert axes from those validated views;
they use the existing runtime-manifest schema and Mooncake adapter. Per-expert
canonical manifests are never gathered. Local expert groups and pair-specific
`w2`/attention `o_proj` blocks are compacted before the one inventory gather.
Exact local aliases reuse source storage. Sources needed by any changed target
remain available until the global completion fence. Stable worker participant
identities are enabled only for this explicit same-fleet call.

Changed source parameters move into final source arenas one bounded chunk at a
time. Each chunk synchronizes, verifies the old allocator blocks are inactive,
and clears the CUDA cache before allocating the next chunk. Target arenas are
final parameter storage and are allocated after source compaction. Packing uses
one cloned parameter at a time, at most 1 GiB under the parameter cap; targets and
source byte order are restored after transfers finish. No full host checkpoint
or host weight copy is retained. Logical canonical metadata is local; the gather
contains compact wire fragments and declarations only.

For subsequent epochs, a source storage can contain both changed weights and
retained aliases. Its registered parameter members form an indivisible release
unit; retained members also move when necessary to free that old allocation.
Changed and retained members receive separate backing arenas, with their combined
payload bounded by the same 1 GiB cap. This prevents target aliases from retaining
unused changed-weight bytes. Oversized registered units, invalid ranges,
overlapping members, empty/null storage and multiple source devices are rejected
before mutation. Unused storage capacity and legal parameter offsets are allowed;
the cap bounds copied registered payload, not the old allocation's capacity.
Fresh independent source allocations retain their original chunk order.

The caller must still drop external hidden views and finish all readers. The
helper cannot enumerate arbitrary external owners; an old active allocation after
replacement and the existing GC fallback remains a fatal release failure.
`source_arena_payload_bytes` reports total copied payload, including necessary
alias relocation; `source_changed_payload_bytes` and
`source_relocated_alias_payload_bytes` separate those costs. Arena count/maximum
describe actual new arenas; `source_release_unit_max_bytes` bounds the combined
payload of a release unit before its verified flush.

The first native canonical layouts validate supported layer semantics and decide
aliases. They are lowered to small grouped wire layout plans, then detached to
plain meta tensors without copying loaders, module references or tensor
attributes. After packing and installation, those same-epoch plans are rebound to
current registered storage through the original byte-range, alias, physical
overlap, complete-coverage and binding checks. Padding initializations, local
buffers and derived-state declarations remain accounted for. Physical adapter
prerequisites are checked again; this is not a second native layer-semantics
validation or a persistent plan cache. Alias fragments are excluded by their
parameter ownership only after complete physical validation.

Every submitted group uses eight XOR endpoint phases with a world barrier after
each phase, including empty phases. Mixed empty/nonempty local target groups are
rejected rather than entering unequal collective sequences. The collective
scope pins models, arenas, engine and registrations until successful completion,
unregistration and peer reset. From the first mutation, any failure parks these
owners and refuses later calls: restart the entire participating fleet. There is
no checkpoint fallback after mutation.

`MC_RESHARD_PLAN_CACHE_ENABLED=1` enables the bounded process-local address-free
logical plan cache (default capacity two). Cache settings must match all workers
and cannot change within a process. Hits still acquire fresh physical bindings
and allocation guards. The result includes actual cache info, compact inventory
counts, group presence, arena release checks, phase completion, bytes and physical
operation counts. These counters are not end-to-end serving latency.

## Validation

With the companion Mooncake Python package available:

```sh
PYTHONPATH=python python -m pytest -q test/manual/weight_cache/inplace
```

For a CPU-only dependency environment, the manual leaf launcher avoids frontend,
distributed-package and utility-package GPU initializers:

```sh
python test/manual/weight_cache/run_cpu_tests.py -q test/manual/weight_cache
```

This launcher installs namespace packages only; it executes the real leaf
implementations. It does not test those package initializers, native model
construction, CUDA allocation or the complete `bind_weights` collective. It is
an explicit manual tool and does not register a CI job.

The CPU tests compare complete target bytes through the actual Mooncake
planner/binder/Reader for all thirty directed transitions between supported
layouts with both the default Reader executor and the phased endpoint executor,
plus explicit meta-export rejection and cache reuse with fresh storage. Separate
tests exercise allocator-release/flush gates and eight real Gloo processes with
symmetric barriers even when a worker has no transfer batches.
They use CPU canonical fixtures; they do not qualify CUDA IPC, native model
construction, online scheduling, generation readiness or 30B performance. Run the
existing native layer-loader golden tests and a matched eight-GPU checkpoint vs
reshard benchmark on the final paired revisions before making those claims.
These manual tests add no remote CI registration.
