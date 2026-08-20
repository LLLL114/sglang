"""Runtime prefill<->decode role switching for PD disaggregation.

The token KV pool is role-independent and never reallocated; only the
role-specific disaggregation structures are torn down and rebuilt on a flip.
The prefix (tree) cache is role-dependent: when both roles select the same
cache recipe it is merely reset, otherwise it is destroyed (host pools,
storage daemon threads included) and rebuilt for the new role.
Kept out of scheduler.py to avoid growing it further.
"""

from __future__ import annotations

import atexit
import logging
from typing import TYPE_CHECKING, Callable, Optional

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import PdRoleSwitchReqInput, PdRoleSwitchReqOutput
from sglang.srt.runtime_context import get_context, get_memory

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class PdRoleSwitchRestart(Exception):
    """Break out of the current role's event loop after a successful switch."""


def run_event_loop_supervisor(
    scheduler: Scheduler, dispatch_once: Callable[[Scheduler], None]
) -> None:
    """Re-dispatch the scheduler event loop after each runtime role switch."""
    while True:
        try:
            return dispatch_once(scheduler)
        except PdRoleSwitchRestart:
            logger.info(
                "Re-dispatching event loop after PD role switch -> %s",
                scheduler.disaggregation_mode.value,
            )


def handle_pd_role_switch(
    scheduler: Scheduler, recv_req: PdRoleSwitchReqInput
) -> PdRoleSwitchReqOutput:
    """Flip the scheduler's disaggregation role at runtime. The instance must be
    idle; rebuild failure is fatal to the instance (no in-place rollback)."""
    old_role = scheduler.disaggregation_mode.value
    new_role = (recv_req.new_role or "").lower()

    def _fail(msg: str) -> PdRoleSwitchReqOutput:
        logger.warning(
            "PD role switch rejected (%s -> %s): %s", old_role, new_role, msg
        )
        return PdRoleSwitchReqOutput(
            success=False, message=msg, old_role=old_role, new_role=new_role
        )

    reason = _reject_reason(scheduler, new_role)
    if reason is not None:
        return _fail(reason)
    if new_role == old_role:
        return PdRoleSwitchReqOutput(
            success=True,
            message="already in target role",
            old_role=old_role,
            new_role=new_role,
        )
    if not scheduler.is_fully_idle():
        return _fail("instance is not idle; drain all requests before switching")

    scheduler._pd_role_switch_in_progress = True
    try:
        # Teardown + role flip + rebuild are one logical atomic step. If any of
        # them raises, the instance is left half-torn-down (old role released,
        # new role not up) and isn't safe to serve, so mark it unhealthy. There
        # is no in-place rollback.
        try:
            teardown_disaggregation(scheduler)
            get_context().override("role_switch.flip", disaggregation_mode=new_role)
            switch_tree_cache(scheduler, old_role, new_role)
            scheduler.init_disaggregation()
            scheduler._sync_disaggregation_mode_to_subcomponents()
        except Exception as e:
            scheduler._pd_role_switch_unhealthy = True
            logger.critical(
                "PD role switch (%s -> %s) failed during teardown/rebuild; "
                "instance unhealthy: %s",
                old_role,
                new_role,
                e,
            )
            return _fail(
                f"role switch failed; instance unhealthy, restart required: {e}"
            )

        if new_role == "decode":
            # Best-effort deferred capture; a failure only degrades to eager.
            try:
                scheduler.tp_worker.ensure_decode_cuda_graphs(
                    recv_req.decode_cuda_graph_bs
                )
            except Exception:
                logger.exception("Decode CUDA graph capture on role switch failed")

        # Break out of the old-role event loop so the supervisor re-dispatches.
        scheduler._event_loop_should_restart = True
        logger.info("PD role switch succeeded: %s -> %s", old_role, new_role)
        return PdRoleSwitchReqOutput(
            success=True, message="ok", old_role=old_role, new_role=new_role
        )
    except Exception as e:
        logger.exception("PD role switch failed")
        return _fail(f"role switch raised: {e}")
    finally:
        scheduler._pd_role_switch_in_progress = False


def _reject_reason(scheduler: Scheduler, new_role: str) -> Optional[str]:
    """Why the switch must be rejected before draining, or None to proceed.

    Table-driven: the first failing precondition's message is returned.
    """
    sa = scheduler.server_args
    km = _current_kv_manager(scheduler)
    # (failed?, lazy message). Messages are callables so only the selected one
    # is built (avoids touching fields irrelevant to the failing check).
    checks = (
        (
            not sa.enable_pd_role_switch,
            lambda: "--enable-pd-role-switch is not set on this instance",
        ),
        (
            scheduler._pd_role_switch_unhealthy,
            lambda: "instance is unhealthy after a failed role switch; restart required",
        ),
        (
            scheduler._pd_role_switch_in_progress,
            lambda: "another role switch is already in progress",
        ),
        (
            new_role not in ("prefill", "decode"),
            lambda: f"invalid new_role={new_role!r}",
        ),
        (
            scheduler.disaggregation_mode == DisaggregationMode.NULL,
            lambda: "instance is not running in PD disaggregation mode",
        ),
        (
            km is not None and not km.supports_role_switch,
            lambda: f"transfer backend {sa.disaggregation_transfer_backend!r} "
            "does not support runtime role switch",
        ),
    )
    return next((msg() for failed, msg in checks if failed), None)


def _current_kv_manager(scheduler: Scheduler):
    """The KV manager of the current role's disaggregation queue, or None."""
    if scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
        q = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
    elif scheduler.disaggregation_mode == DisaggregationMode.DECODE:
        q = getattr(scheduler, "disagg_decode_prealloc_queue", None)
    else:
        q = None
    return getattr(q, "kv_manager", None) if q is not None else None


def teardown_disaggregation(scheduler: Scheduler) -> None:
    """Release the current role's disaggregation structures (queues, metadata
    buffers, KV transfer manager) so the other role can be rebuilt."""
    mode = scheduler.disaggregation_mode
    if mode == DisaggregationMode.PREFILL:
        q = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
        if q is not None:
            km = getattr(q, "kv_manager", None)
            if km is not None:
                km.teardown()
            scheduler.disagg_prefill_bootstrap_queue = None
        scheduler.disagg_prefill_inflight_queue = []
    elif mode == DisaggregationMode.DECODE:
        q = getattr(scheduler, "disagg_decode_prealloc_queue", None)
        if q is not None:
            km = getattr(q, "kv_manager", None)
            if km is not None:
                km.teardown()
            scheduler.disagg_decode_prealloc_queue = None
        scheduler.disagg_decode_transfer_queue = None
    scheduler.disagg_metadata_buffers = None
    scheduler.req_to_metadata_buffer_idx_allocator = None


def switch_tree_cache(scheduler: Scheduler, old_role: str, new_role: str) -> None:
    """Reset or rebuild the prefix cache for the flipped role.

    The cache *type* is a function of role-derived configuration (see
    _tree_cache_recipe_for_role). When both roles resolve to the same recipe
    the historical reset path is enough; when they differ, the old cache is
    fully destroyed (pinned host pools, storage daemon threads, shared-pool
    hooks) and a new one is built through the regular kv_cache_builder path.
    Must run after the disaggregation-mode flip so re-resolution against the
    config bags sees the new role.
    """
    old_recipe = _tree_cache_recipe_for_role(scheduler, old_role)
    new_recipe = _tree_cache_recipe_for_role(scheduler, new_role)
    if new_recipe == old_recipe:
        _release_prefix_cache_for_role_switch(scheduler)
        return
    logger.info(
        "PD role switch %s -> %s changes the tree cache recipe "
        "(disable_radix_cache, retraction_backup): %s -> %s; rebuilding %s",
        old_role,
        new_role,
        old_recipe,
        new_recipe,
        type(scheduler.tree_cache).__name__,
    )
    _destroy_tree_cache_for_role_switch(scheduler)
    _rebuild_tree_cache_for_role_switch(scheduler, new_role)


def _role_disable_radix_cache(scheduler: Scheduler, role: str) -> bool:
    """The disable_radix_cache value `role` selects its tree cache with."""
    sa = scheduler.server_args
    stash = getattr(sa, "_pd_role_disable_radix_cache", None)
    if stash is None:
        # No per-role intent recorded (e.g. schedulers built directly in
        # tests): keep the currently resolved flag for both roles.
        return get_memory().disable_radix_cache
    # A model-specific resolution pass after the PD hook may force radix off
    # for reasons that apply to both roles. Detect it on the pristine startup
    # record: the started role's recorded intent was radix ON, yet the
    # resolved flag ended up OFF.
    later_forced = sa.disable_radix_cache and not stash[sa.disaggregation_mode]
    return stash[role] or later_forced


def _role_retraction_backup(scheduler: Scheduler, role: str) -> str:
    """The retraction-backup backend `role` selects its tree cache with."""
    explicit = scheduler.server_args.disaggregation_decode_retraction_backup
    if explicit is not None:
        return explicit
    from sglang.srt.mem_cache.kv_cache_builder import (
        compute_auto_decode_retraction_backup,
    )

    return compute_auto_decode_retraction_backup(
        tp_worker=scheduler.tp_worker, mode=role
    )


def _tree_cache_recipe_for_role(scheduler: Scheduler, role: str) -> tuple[bool, str]:
    """The role-derived inputs that drive tree cache selection.

    Everything else feeding default_radix_cache_factory (model hybridness,
    env flags, hierarchical-cache enablement, ...) is role-independent, so
    two roles with equal recipes build the same cache type.
    """
    return (
        _role_disable_radix_cache(scheduler, role),
        _role_retraction_backup(scheduler, role),
    )


def _destroy_tree_cache_for_role_switch(scheduler: Scheduler) -> None:
    """Fully release the old role's prefix cache before rebuilding.

    reset() alone is not enough across cache types: the HiCache variants own
    pinned host pools, storage prefetch/backup daemon threads, a
    layer-transfer hook on the shared device pool, and an atexit auto-detach
    holding a strong reference — all of which would leak once the scheduler
    drops its reference. The instance is fully idle (checked before
    teardown), so nothing races the destruction.
    """
    tree_cache = scheduler.tree_cache
    # Best-effort L3 clear while the bookkeeping is still intact: stale pages
    # must not be matched by the rebuilt cache of the other role.
    clear_storage = getattr(tree_cache, "clear_storage_backend", None)
    if callable(clear_storage):
        try:
            clear_storage()
        except Exception:
            logger.exception("hicache storage clear on role switch failed")
    # Stop the prefetch/backup daemon threads and release the L3 backend.
    detach_storage = getattr(tree_cache, "detach_storage_backend", None)
    if callable(detach_storage):
        try:
            detach_storage()
        except Exception:
            logger.exception("hicache storage detach on role switch failed")
    # Unlock and drop every cached prefix (frees the shared device KV slots).
    tree_cache.reset()
    # Free the pinned host pools (HiCache L2); idempotent no-op otherwise.
    tree_cache.release_host_resources()
    # Unhook the dead cache's controller from the shared device pool and the
    # tp_worker; a stale LayerDoneCounter would gate KV reads forever.
    kv_cache = scheduler.token_to_kv_pool_allocator.get_kvcache()
    if hasattr(kv_cache, "register_layer_transfer_counter"):
        kv_cache.register_layer_transfer_counter(None)
    scheduler.tp_worker.register_hicache_layer_transfer_counter(None)
    # The HiCache variants atexit-register their shutdown; unregister so the
    # dead cache is collectable and no stale detach runs at process exit.
    shutdown = getattr(tree_cache, "shutdown", None)
    if callable(shutdown):
        atexit.unregister(shutdown)
    scheduler.req_to_token_pool.clear()
    scheduler.token_to_kv_pool_allocator.clear()


def _rebuild_tree_cache_for_role_switch(scheduler: Scheduler, new_role: str) -> None:
    """Re-resolve the role-derived cache config and rebuild the tree cache."""
    from sglang.srt.mem_cache import kv_cache_builder

    sa = scheduler.server_args
    fields = {"disable_radix_cache": _role_disable_radix_cache(scheduler, new_role)}
    if sa.disaggregation_decode_retraction_backup is None:
        # Startup auto-resolved the backend for the old role; clear it so
        # build_kv_cache re-resolves against the flipped mode.
        fields["disaggregation_decode_retraction_backup"] = None
    if not getattr(sa, "_hicache_ratio_user_set", True):
        # Re-default per role, mirroring startup: decode leaves the ratio to
        # the retraction resolver (1.0 for host_pool), prefill takes the
        # static default from _handle_hicache_ratio_default.
        fields["hicache_ratio"] = (
            None
            if new_role == "decode"
            else (1.2 if sa.hicache_host_memory_mode == "buffer_only" else 2.0)
        )
    get_context().override("role_switch.tree_cache", **fields)

    result = kv_cache_builder.build_kv_cache(
        **scheduler._kv_cache_build_kwargs, for_pd_role_switch=True
    )
    scheduler.disable_radix_cache = result.disable_radix_cache
    scheduler._rebind_tree_cache(result.tree_cache)
    logger.info(
        "PD role switch rebuilt tree cache for %s role: %s",
        new_role,
        type(result.tree_cache).__name__,
    )


def _release_prefix_cache_for_role_switch(scheduler: Scheduler) -> None:
    """Release the prefix (radix/hicache) cache so a flip works with radix ON.

    With radix disabled (ChunkCache) the flip needs nothing here: ChunkCache
    keeps no persistent prefixes and, since the instance is idle before the
    switch, the allocator is already empty. This is the historical
    ``--disable-radix-cache`` path, left untouched by the guard below.

    With radix (or hicache) enabled, finished prefixes stay in the tree and keep
    their KV-pool slots *locked* even while idle. Carried across a role switch
    that means (a) the new role would match against stale prefixes whose KV no
    longer means what it did (corruption) and (b) those locked slots would leak
    on every flip. Reset mirrors ``Scheduler.flush_cache``'s cache-release block
    (the instance is already fully idle, checked before teardown) and, for
    hicache, best-effort clears the storage backend so it is released completely.
    """
    if scheduler.disable_radix_cache:
        return
    tree_cache = scheduler.tree_cache
    if tree_cache is not None:
        clear_storage = getattr(tree_cache, "clear_storage_backend", None)
        if callable(clear_storage):
            try:
                clear_storage()
            except Exception:
                logger.exception("hicache storage release on role switch failed")
        tree_cache.reset()
    scheduler.req_to_token_pool.clear()
    scheduler.token_to_kv_pool_allocator.clear()
