"""Drain the PP wire protocol before changing a scheduler's PD role.

Control votes run off the scheduler thread: a stage must keep forwarding the
old protocol while later stages are still receiving the control request.

Ingress is received but not dispatched during a switch, including aborts and
generation health checks. Drain externally and allow health-probe deadlines
to cover this window. Resource phases use --watchdog-timeout (at least 30s)
as their collective budget; configure it above the longest expected rebuild
or first graph capture. Progress between phases does not extend a single
action's watchdog budget.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta

import torch.distributed as dist

from sglang.srt.disaggregation import role_switch
from sglang.srt.managers.io_struct import PdRoleSwitchReqOutput
from sglang.srt.runtime_context import get_context, get_device, get_parallel
from sglang.srt.utils.watchdog import WatchdogRaw

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PPRoleSwitchMarker:
    sequence: int
    phase: str


class PPRoleSwitch:
    def __init__(self, scheduler, group=None, marker_group=None, phase_group=None):
        self.scheduler = scheduler
        self.group = group
        self.owns_group = group is None
        self.marker_group = marker_group
        self.phase_group = phase_group if phase_group is not None else group
        if self.owns_group:
            self.group = dist.new_group(
                ranks=get_parallel().world_group.ranks,
                backend="gloo",
                timeout=timedelta(seconds=30),
            )
            # PP0 must decide on one iteration across TP ranks. Keep that
            # control broadcast off the inference group's long timeout too.
            self.marker_group = dist.new_group(
                ranks=get_parallel().world_group.ranks[: get_parallel().tp_size],
                backend="gloo",
                timeout=timedelta(seconds=30),
            )
            # A finished rank may wait for a peer's entire resource rebuild or
            # first graph capture. Keep this budget separate from control votes.
            self.phase_group = dist.new_group(
                ranks=get_parallel().world_group.ranks,
                backend="gloo",
                timeout=timedelta(
                    seconds=max(30, get_device().watchdog_timeout or 300)
                ),
            )
        self.rank = dist.get_rank(self.group)
        self.size = dist.get_world_size(self.group)
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pd-pp-vote"
        )
        self.sequence = 0
        self.progress = 0
        self.request = None
        self.phase = None
        self.vote = None
        self.result = None
        if self.owns_group and get_device().watchdog_timeout is not None:
            # The inference watchdog is inactive on an idle pipeline. A stuck
            # old-protocol recv must not prevent an in-progress switch from
            # reaching the same existing fatal-watchdog path.
            self.watchdog = WatchdogRaw(
                debug_name="PP PD role switch",
                get_counter=lambda: self.progress,
                is_active=lambda: self.request is not None,
                watchdog_timeout=get_device().watchdog_timeout,
                dump_info=lambda: f"sequence={self.sequence}, phase={self.phase}",
            )

    @property
    def draining(self):
        return self.phase == "drain"

    def enqueue(self, request):
        if self.request is not None:
            raise RuntimeError(
                "Concurrent PP role-switch requests bypassed serialization"
            )
        self.sequence += 1
        self.request = request
        self._advance_phase("prepare")
        if self.rank == 0:
            logger.info(
                "PP role switch %d requested: target=%s; scheduler ingress, "
                "including aborts and generation health checks, is deferred "
                "until the control response",
                self.sequence,
                request.new_role,
            )

    def _advance_phase(self, phase):
        self.phase = phase
        self.progress += 1

    def _agree(self, phase, local):
        values = [None] * self.size
        group = (
            self.phase_group
            if phase in ("teardown", "rebuild", "decode_graphs", "restarted")
            else self.group
        )
        dist.all_gather_object(values, (self.sequence, phase, local), group=group)
        if any(v[:2] != (self.sequence, phase) for v in values):
            raise RuntimeError("PP role-switch control sequence mismatch")
        return [v[2] for v in values]

    def _check(self, initial):
        s = self.scheduler
        target = (self.request.new_role or "").lower()
        old = s.disaggregation_mode.value
        rejection = role_switch._reject_reason(s, target) if initial else None
        if rejection is None and target != old:
            rejection = role_switch._preflight_reason(s, self.request, target)
        return {
            "old_role": old,
            "new_role": target,
            "rejection": rejection,
            "graphs": list(s.tp_worker.get_decode_cuda_graph_bs()),
            "idle": s.is_fully_idle(),
        }

    def _merge_preflight(self, values):
        old = values[0]["old_role"]
        target = values[0]["new_role"]
        if any((v["old_role"], v["new_role"]) != (old, target) for v in values):
            raise RuntimeError("PP ranks disagree on the current or target role")
        errors = [
            (rank, v["rejection"])
            for rank, v in enumerate(values)
            if v["rejection"] is not None
        ]
        if (
            not errors
            and target == "decode"
            and target != old
            and any(v["graphs"] != values[0]["graphs"] for v in values)
        ):
            errors = [(0, ("PP ranks have inconsistent decode CUDA graphs", True))]
        return PdRoleSwitchReqOutput(
            success=not errors,
            message="; ".join(f"rank {rank}: {error[0]}" for rank, error in errors)
            or ("already in target role" if old == target else "ready"),
            old_role=old,
            new_role=target,
            safe_to_restore=all(error[1] for _, error in errors),
        )

    def poll_inputs(self, requests):
        if self.phase == "prepare" and self.vote is None:
            # Check after the entire input batch was dispatched, including any
            # work that arrived alongside the control request.
            local = self._check(initial=True)
            self.scheduler._pd_role_switch_in_progress = True
            self.vote = self.executor.submit(self._agree, "prepare", local)
        if get_parallel().pp_rank == 0 and self.request is not None:
            marker = None
            if (
                self.rank == 0
                and self.vote is not None
                and self.vote.done()
                and self.phase == "prepare"
            ):
                self.result = self._merge_preflight(self.vote.result())
                finish = (
                    not self.result.success
                    or self.request.prepare_only
                    or self.result.old_role == self.result.new_role
                )
                marker = PPRoleSwitchMarker(
                    self.sequence, "finish" if finish else "drain"
                )
            elif (
                self.rank == 0
                and self.vote is not None
                and self.vote.done()
                and self.phase == "drain"
            ):
                self.vote.result()
                marker = PPRoleSwitchMarker(self.sequence, "commit")
            # Future completion is not simultaneous across TP ranks. Broadcast
            # PP0's decision so they change the protocol on the same iteration.
            markers = [marker]
            tp = get_parallel().tp_group
            dist.broadcast_object_list(
                markers, src=tp.ranks[0], group=self.marker_group
            )
            if markers[0] is not None:
                requests.append(markers[0])
                self.handle_marker(markers[0])
        return self.phase == "commit"

    def handle_marker(self, marker):
        if marker.sequence != self.sequence or self.vote is None:
            raise RuntimeError("Unexpected PP role-switch marker")
        values = self.vote.result()
        if marker.phase in ("drain", "finish"):
            self.result = self._merge_preflight(values)
        self.vote = None
        self._advance_phase(marker.phase)
        if marker.phase == "finish":
            self._respond()

    def after_iteration(self, slots, tails):
        if self.draining and self.vote is None:
            if any(value is not None for slot in slots for value in slot):
                return
            if any(value is not None for value in tails):
                return
            self.vote = self.executor.submit(self._agree, "drained", None)

    def _run_phase(self, phase, action):
        self._advance_phase(phase)
        error = None
        try:
            action()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.progress += 1
        errors = self._agree(phase, error)
        self.progress += 1
        if any(errors):
            self.scheduler._pd_role_switch_unhealthy = True
            raise RuntimeError(f"PP role switch failed during {phase}: {errors}")

    def commit(self):
        s = self.scheduler
        # The input contract requires a drained ingress. Check again before
        # releasing anything, after all PP protocol messages have been consumed.
        values = self._agree("recheck", self._check(False))
        if any(not value["idle"] for value in values):
            s._pd_role_switch_unhealthy = True
            raise RuntimeError("PP role switch encountered new work after draining")
        self.result = self._merge_preflight(values)
        if not self.result.success:
            # The wire has drained, so restart the same role's PP loop before
            # answering the safe rejection; do not discard any role resources.
            self._advance_phase("restarting")
            return
        self._run_phase("teardown", lambda: role_switch.teardown_disaggregation(s))

        def rebuild():
            get_context().override(
                "role_switch.flip", disaggregation_mode=self.result.new_role
            )
            s.init_disaggregation()
            s._sync_disaggregation_mode_to_subcomponents()

        self._run_phase("rebuild", rebuild)
        if self.result.new_role == "decode":
            self._run_phase(
                "decode_graphs",
                lambda: s.tp_worker.ensure_decode_cuda_graphs(
                    self.request.decode_cuda_graph_bs
                ),
            )
        self.result.message = "ok"
        self.result.safe_to_restore = False
        self._advance_phase("restarting")

    def on_loop_started(self):
        if self.phase == "restarting":
            self._agree("restarted", None)
            self.progress += 1
            self._respond()

    def _respond(self):
        self.scheduler.send_control_output(self.result, self.request)
        if self.rank == 0:
            logger.info(
                "PP role switch %d response: success=%s, message=%s; "
                "deferred scheduler ingress can resume",
                self.sequence,
                self.result.success,
                self.result.message,
            )
        self.scheduler._pd_role_switch_in_progress = False
        self.request = self.result = self.vote = self.phase = None

    def close(self):
        self.executor.shutdown(wait=True)
        self.request = None
        if self.owns_group:
            if self.rank < get_parallel().tp_size:
                dist.destroy_process_group(self.marker_group)
            dist.destroy_process_group(self.phase_group)
            dist.destroy_process_group(self.group)
