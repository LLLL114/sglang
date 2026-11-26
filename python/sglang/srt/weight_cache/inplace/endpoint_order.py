"""Bounded endpoint rotation inside the existing pinned native submission."""

from mooncake.reshard.transfer_engine.executor import MooncakeTransferEngineExecutor


class BalancedEndpointExecutor(MooncakeTransferEngineExecutor):
    def configure(
        self, *, rank, target_attention_tp, local_endpoint, endpoints, phase_barrier
    ):
        if self.pending_transfer_ids():
            raise RuntimeError(
                "cannot change endpoint policy with pending native transfers"
            )
        if rank < 0 or target_attention_tp not in (1, 4, 8):
            raise ValueError("invalid endpoint rotation topology")
        if (
            len(endpoints) != 8
            or len(set(endpoints)) != 8
            or not 0 <= rank < 8
            or endpoints[rank] != local_endpoint
            or not callable(phase_barrier)
        ):
            raise ValueError(
                "phased experiment requires eight rank-ordered endpoints and a barrier"
            )
        self.endpoints, self.phase_barrier = tuple(endpoints), phase_barrier
        self.rank, self.target_attention_tp, self.local_endpoint = (
            rank,
            target_attention_tp,
            local_endpoint,
        )
        self.profiles = []

    def _execute_batches(self, batches, direction, *, max_inflight_batches, submission):
        if max_inflight_batches != 1:
            raise ValueError(
                "endpoint candidate requires original max_inflight_batches=1"
            )
        # Fixed eight-GPU experiment metadata cap; core limits stay unchanged.
        groups = {}
        original_batches, original_records, indices = [], [], {}
        count = operations = nbytes = 0
        for batch in batches:
            count += 1
            operations += batch.operation_count
            nbytes += batch.nbytes
            if count > 128 or operations > 131072:
                raise ValueError(
                    "endpoint scheduling metadata exceeds 128 batches / 131072 operations"
                )
            if id(batch) in indices:
                raise ValueError("endpoint scheduler requires unique batch objects")
            indices[id(batch)] = count - 1
            original_records.append(
                dict(
                    index=count - 1,
                    endpoint=batch.endpoint,
                    operations=batch.operation_count,
                    nbytes=batch.nbytes,
                )
            )
            original_batches.append(batch.endpoint)
            groups.setdefault(batch.endpoint, []).append(batch)
        endpoints = list(groups)
        if any(endpoint not in self.endpoints for endpoint in endpoints):
            raise ValueError("batch endpoint absent from the fixed eight-rank group")
        phase_endpoints = [self.endpoints[self.rank ^ phase] for phase in range(8)]
        order = [endpoint for endpoint in phase_endpoints if endpoint in groups]
        profile = dict(
            policy="eight-phase-xor-world-barrier",
            rank=self.rank,
            target_attention_tp=self.target_attention_tp,
            local_endpoint=self.local_endpoint,
            original_endpoint_order=endpoints,
            original_batch_endpoints=original_batches,
            original_batch_records=original_records,
            planned_batch_indices=[
                indices[id(batch)] for endpoint in order for batch in groups[endpoint]
            ],
            endpoint_order=order,
            batches=count,
            physical_operations=operations,
            nbytes=nbytes,
            rank_ordered_endpoints=list(self.endpoints),
            phase_endpoint_order=phase_endpoints,
            phases=[],
            phase_barrier_count=0,
            actual_batch_endpoints=[],
            actual_batch_indices=[],
            actual_batch_operations=[],
            actual_batch_nbytes=[],
            actual_physical_operations=0,
            actual_nbytes=0,
            completed=False,
        )
        self.profiles.append(profile)
        self._active_indices = indices
        self._active_profile = profile
        try:
            receipts = []
            for phase, endpoint in enumerate(phase_endpoints):
                phase_batches = groups.get(endpoint, ())
                phase_record = dict(
                    phase=phase,
                    source_rank=self.rank ^ phase,
                    endpoint=endpoint,
                    batch_indices=[indices[id(batch)] for batch in phase_batches],
                    physical_operations=sum(
                        batch.operation_count for batch in phase_batches
                    ),
                    nbytes=sum(batch.nbytes for batch in phase_batches),
                    actual_batch_indices=[],
                    actual_batch_endpoints=[],
                    actual_physical_operations=0,
                    actual_nbytes=0,
                    barrier_before_completed=True,
                    barrier_before_source=(
                        "registration-start-fence"
                        if phase == 0
                        else "previous-phase-after"
                    ),
                    native_complete=False,
                    barrier_after_completed=False,
                    barrier_complete=False,
                    completed=False,
                )
                profile["phases"].append(phase_record)
                self._active_phase = phase_record
                receipts.extend(
                    super()._execute_batches(
                        iter(phase_batches),
                        direction,
                        max_inflight_batches=max_inflight_batches,
                        submission=submission,
                    )
                )
                phase_record["native_complete"] = True
                self.phase_barrier()
                profile["phase_barrier_count"] += 1
                phase_record["barrier_complete"] = True
                phase_record["barrier_after_completed"] = True
                phase_record["completed"] = True
            result = tuple(receipts)
            profile["completed"] = True
            return result
        finally:
            self._active_profile = None
            self._active_indices = None
            self._active_phase = None

    def _execute_batch(self, batch, direction, submission):
        profile = getattr(self, "_active_profile", None)
        if profile is not None:
            phase = self._active_phase
            phase["actual_batch_indices"].append(self._active_indices[id(batch)])
            phase["actual_batch_endpoints"].append(batch.endpoint)
            phase["actual_physical_operations"] += batch.operation_count
            phase["actual_nbytes"] += batch.nbytes
            profile["actual_batch_indices"].append(self._active_indices[id(batch)])
            profile["actual_batch_operations"].append(batch.operation_count)
            profile["actual_batch_nbytes"].append(batch.nbytes)
            profile["actual_batch_endpoints"].append(batch.endpoint)
            profile["actual_physical_operations"] += batch.operation_count
            profile["actual_nbytes"] += batch.nbytes
        return super()._execute_batch(batch, direction, submission)


def make_executor(
    engine, *, rank, target_attention_tp, local_endpoint, endpoints, phase_barrier
):
    """Build the example scheduler before constructing the public Reader."""
    executor = BalancedEndpointExecutor(engine)
    executor.configure(
        rank=rank,
        target_attention_tp=target_attention_tp,
        local_endpoint=local_endpoint,
        endpoints=endpoints,
        phase_barrier=phase_barrier,
    )
    return executor
