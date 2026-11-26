"""Pair-specific down-weight blocks with one layer of packing scratch."""


def parts(args, peer_args):
    if args.tp_size != peer_args.tp_size:
        raise ValueError("same fleet required")
    return max(1, args.ep_size // peer_args.ep_size)


def _blocked_parameters(parameters, blocks, attention_blocks):
    for name, value in parameters.items():
        count = (
            blocks
            if name.endswith("experts.w2_weight")
            else (attention_blocks if "self_attn.o_proj." in name else 1)
        )
        if count == 1:
            continue
        if value.shape[-1] % count:
            raise ValueError("down width must divide wire blocks")
        yield value, count


def validate(parameters, blocks, *, attention_blocks=1):
    for _ in _blocked_parameters(parameters, blocks, attention_blocks):
        pass


def convert(parameters, blocks, *, pack, attention_blocks=1):
    peak = total = 0
    for value, count in _blocked_parameters(parameters, blocks, attention_blocks):
        value = value.detach()
        value = value.view(1, *value.shape) if len(value.shape) == 2 else value
        experts, hidden, width = value.shape
        chunk = width // count
        scratch = value.clone()
        size = scratch.numel() * scratch.element_size()
        peak, total = max(peak, size), total + size
        if pack:
            wire = value.view(count, experts, hidden, chunk)
            for block in range(count):
                wire[block].copy_(scratch[:, :, block * chunk : (block + 1) * chunk])
        else:
            wire = scratch.view(count, experts, hidden, chunk)
            for block in range(count):
                value[:, :, block * chunk : (block + 1) * chunk].copy_(wire[block])
        del wire, scratch
    return dict(scratch_peak_bytes=peak, scratch_total_bytes=total)
