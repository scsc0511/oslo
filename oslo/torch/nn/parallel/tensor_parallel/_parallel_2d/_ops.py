from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.distributed._ops import all_gather, all_reduce, reduce_scatter


def matmul_2d(
    a,
    b,
    summa_dim,
    out_shape,
    parallel_context: ParallelContext,
    row_rank=None,
    col_rank=None,
    row_parallel_mode=ParallelMode.TENSOR_2D_ROW,
    col_parallel_mode=ParallelMode.TENSOR_2D_COL,
):
    if row_rank is None:
        row_rank = parallel_context.get_local_rank(row_parallel_mode)
    if col_rank is None:
        col_rank = parallel_context.get_local_rank(col_parallel_mode)

    data_parallel_rank = (
        0
        if not parallel_context.is_initialized(ParallelMode.DATA)
        else parallel_context.get_local_rank(ParallelMode.DATA)
    )
    pipeline_parallel_rank = (
        0
        if not parallel_context.is_initialized(ParallelMode.PIPELINE)
        else parallel_context.get_local_rank(ParallelMode.PIPELINE)
    )
    pipeline_parallel_size = (
        1
        if not parallel_context.is_initialized(ParallelMode.PIPELINE)
        else parallel_context.get_world_size(ParallelMode.PIPELINE)
    )
    tensor_parallel_size = summa_dim**2

    return Matmul_AB_2D(
        a,
        b,
        summa_dim,
        parallel_context,
        out_shape,
        row_rank,
        col_rank,
        row_parallel_mode,
        col_parallel_mode,
        data_parallel_rank,
        pipeline_parallel_rank,
        pipeline_parallel_size,
        tensor_parallel_size,
    )


def add_bias_2d(
    input_: Tensor,
    bias: Tensor,
    output_size_per_partition: int,
    row_rank: int,
    col_rank: int,
    parallel_context: ParallelContext,
    row_parallel_mode: ParallelMode,
    col_parallel_mode: ParallelMode,
    skip_bias_add: bool,
    data_parallel_rank: int,
    pipeline_parallel_rank: int,
    pipeline_parallel_size: int,
    tensor_parallel_size: int,
) -> Tensor:
    return _Add_Bias_2D.apply(
        input_,
        bias,
        output_size_per_partition,
        row_rank,
        col_rank,
        parallel_context,
        row_parallel_mode,
        col_parallel_mode,
        skip_bias_add,
        data_parallel_rank,
        pipeline_parallel_rank,
        pipeline_parallel_size,
        tensor_parallel_size,
    )


def layernorm_2d(
    input_: Tensor,
    E_x: Tensor,
    Var_x: Tensor,
    hidden_size: int,
    parallel_context: ParallelContext,
    row_parallel_mode: ParallelMode,
    col_parallel_mode: ParallelMode,
) -> Tensor:

    return _Layernorm_2D.apply(
        input_,
        E_x,
        Var_x,
        hidden_size,
        parallel_context,
        row_parallel_mode,
        col_parallel_mode,
    )


def all_gather_tensor_2d(
    tensor: Tensor,
    dim: int,
    parallel_mode: ParallelMode,
    parallel_context: ParallelContext,
) -> Tensor:
    return _AllGatherTensor2D.apply(
        tensor,
        dim,
        parallel_mode,
        parallel_context,
    )


def split_batch_2d(
    input_: Tensor,
    parallel_context: ParallelContext,
    dim: int = 0,
) -> Tensor:
    dim_size = input_.size(dim)
    world_size = parallel_context.get_world_size(ParallelMode.TENSOR_2D_COL)

    if world_size <= 1:
        return input_

    assert (
        dim_size % world_size == 0
    ), f"The batch size ({dim_size}) is not a multiple of 2D size ({world_size})."

    return torch.chunk(
        input_, parallel_context.get_world_size(ParallelMode.TENSOR_2D_COL), dim=dim
    )[parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)].contiguous()


def reduce_tensor_2d(
    input_: Tensor,
    parallel_mode: ParallelMode,
    parallel_context: ParallelContext,
) -> Tensor:
    return _ReduceTensor2D.apply(
        input_,
        parallel_mode,
        parallel_context,
    )


def reduce_scatter_tensor_2d(
    tensor: Tensor,
    dim: int,
    parallel_mode: ParallelMode,
    parallel_context: ParallelContext,
) -> Tensor:
    dim_size = tensor.size(dim)
    world_size = parallel_context.get_world_size(parallel_mode)
    assert (
        dim_size % world_size == 0
    ), f"The batch size ({dim_size}) is not a multiple of 2D size ({world_size})."

    return _ReduceScatterTensor2D.apply(
        tensor,
        dim,
        parallel_mode,
        parallel_context,
    )


def reduce_by_batch_2d(
    input_,
    parallel_context: ParallelContext,
    reduce_mean: bool = False,
) -> Tensor:
    return _ReduceByBatch2D.apply(
        input_,
        parallel_context,
        reduce_mean,
    )


class Matmul_AB_2D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        summa_dim: int,
        parallel_context: ParallelContext,
        out_shape: Tuple[int, ...],
        row_rank: int,
        col_rank: int,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
    ) -> Tensor:
        # A: [b / q, s, h / q] -> [(b * s) / q, h / q]
        # B: [h / q, s / q]
        # C: [b / q, s, s / q] -> [(b * s) / q, s / q]

        assert A.shape[-1] == B.shape[-2], "Invalid shapes: A={}, B={} for AB.".format(
            A.shape, B.shape
        )

        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[0], B.shape[-1])
        C = torch.zeros(C_shape, dtype=A.dtype, device=torch.cuda.current_device())

        # use circular buffer to store the communication tensor
        # 2 is enough for all cases
        A_list = [torch.empty_like(A) for _ in range(2)]
        B_list = [torch.empty_like(B) for _ in range(2)]

        row_group = parallel_context.get_group(row_parallel_mode)
        col_group = parallel_context.get_group(col_parallel_mode)

        src_a = (
            summa_dim * col_rank
            + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size
            + pipeline_parallel_rank * tensor_parallel_size
        )
        src_b = (
            row_rank
            + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size
            + pipeline_parallel_rank * tensor_parallel_size
        )

        opa = [None] * 2
        opb = [None] * 2

        A_list[0].copy_(A)
        B_list[0].copy_(B)
        opa[0] = dist.broadcast(A_list[0], src=src_a, group=row_group, async_op=True)
        opb[0] = dist.broadcast(B_list[0], src=src_b, group=col_group, async_op=True)
        cur = 0

        for i in range(summa_dim):
            if i != summa_dim - 1:
                A_list[1 - cur].copy_(A)
                opa[1 - cur] = dist.broadcast(
                    A_list[1 - cur], src=src_a + 1, group=row_group, async_op=True
                )
                B_list[1 - cur].copy_(B)
                opb[1 - cur] = dist.broadcast(
                    B_list[1 - cur],
                    src=src_b + summa_dim,
                    group=col_group,
                    async_op=True,
                )

            if opa[cur] is not None:
                opa[cur].wait()
            if opb[cur] is not None:
                opb[cur].wait()

            torch.addmm(C, A_list[cur], B_list[cur], out=C)
            cur = 1 - cur
            src_a += 1
            src_b += summa_dim

        out = C.reshape(out_shape)

        if ctx:
            ctx.summa_dim = summa_dim
            ctx.parallel_context = parallel_context
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors
        with torch.no_grad():
            A_grad = Matmul_ABT_2D.apply(
                output_grad,
                B,
                ctx.summa_dim,
                ctx.parallel_context,
                ctx.A_shape,
                ctx.row_rank,
                ctx.col_rank,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size,
            )
            B_grad = Matmul_ATB_2D.apply(
                A,
                output_grad,
                ctx.summa_dim,
                ctx.parallel_context,
                ctx.B_shape,
                ctx.row_rank,
                ctx.col_rank,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size,
            )
        return (
            A_grad,
            B_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class Matmul_ABT_2D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        summa_dim: int,
        parallel_context: ParallelContext,
        out_shape: Tuple[int, ...],
        row_rank: int,
        col_rank: int,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
    ) -> Tensor:

        assert A.shape[-1] == B.shape[-1], "Invalid shapes: A={}, B={} for ABT.".format(
            A.shape, B.shape
        )

        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[0], B.shape[0])
        C = torch.empty(C_shape, dtype=A.dtype, device=torch.cuda.current_device())

        # use circular buffer to store the communication tensor
        # 2 is enough for all cases
        B_list = [torch.empty_like(B) for _ in range(2)]
        C_list = [torch.empty_like(C) for _ in range(2)]

        row_group = parallel_context.get_group(row_parallel_mode)
        col_group = parallel_context.get_group(col_parallel_mode)

        src_b = (
            row_rank
            + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size
            + pipeline_parallel_rank * tensor_parallel_size
        )
        src_c = (
            summa_dim * col_rank
            + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size
            + pipeline_parallel_rank * tensor_parallel_size
        )

        opb = [None] * 2
        opr = [None] * 2

        B_list[0].copy_(B)
        opb[0] = dist.broadcast(B_list[0], src=src_b, group=col_group, async_op=True)
        cur = 0

        for i in range(summa_dim):
            if i != summa_dim - 1:
                B_list[1 - cur].copy_(B)
                opb[1 - cur] = dist.broadcast(
                    B_list[1 - cur],
                    src=src_b + summa_dim,
                    group=col_group,
                    async_op=True,
                )

            if opr[cur] is not None:
                opr[cur].wait()
                if i - 2 == row_rank:
                    C.copy_(C_list[cur])

            if opb[cur] is not None:
                opb[cur].wait()

            torch.matmul(A, B_list[cur].transpose(0, 1), out=C_list[cur])
            opr[cur] = dist.reduce(
                C_list[cur], dst=src_c, group=row_group, async_op=True
            )
            cur = 1 - cur
            src_b += summa_dim
            src_c += 1

        for op in opr:
            op.wait()

        if summa_dim - 2 == row_rank:
            C.copy_(C_list[cur])
        if summa_dim - 1 == row_rank:
            C.copy_(C_list[1 - cur])
        out = C.reshape(out_shape)

        if ctx:
            ctx.summa_dim = summa_dim
            ctx.parallel_context = parallel_context
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors

        with torch.no_grad():
            A_grad = Matmul_AB_2D.apply(
                output_grad,
                B,
                ctx.summa_dim,
                ctx.parallel_context,
                ctx.A_shape,
                ctx.row_rank,
                ctx.col_rank,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size,
            )
            B_grad = Matmul_ATB_2D.apply(
                output_grad,
                A,
                ctx.summa_dim,
                ctx.parallel_context,
                ctx.B_shape,
                ctx.row_rank,
                ctx.col_rank,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size,
            )
        return (
            A_grad,
            B_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class Matmul_ATB_2D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        summa_dim: int,
        parallel_context: ParallelContext,
        out_shape: Tuple[int, ...],
        row_rank: int,
        col_rank: int,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
    ) -> Tensor:

        assert A.shape[-2] == B.shape[-2], "Invalid shapes: A={}, B={} for ATB.".format(
            A.shape, B.shape
        )

        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[-1], B.shape[-1])
        C = torch.empty(C_shape, dtype=A.dtype, device=torch.cuda.current_device())

        # use circular buffer to store the communication tensor
        # 2 is enough for all cases
        A_list = [torch.empty_like(A) for _ in range(2)]
        C_list = [torch.empty_like(C) for _ in range(2)]

        row_group = parallel_context.get_group(row_parallel_mode)
        col_group = parallel_context.get_group(col_parallel_mode)

        src_a = (
            summa_dim * col_rank
            + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size
            + pipeline_parallel_rank * tensor_parallel_size
        )
        src_c = (
            row_rank
            + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size
            + pipeline_parallel_rank * tensor_parallel_size
        )

        opa = [None] * 2
        opr = [None] * 2

        A_list[0].copy_(A)
        opa[0] = dist.broadcast(A_list[0], src=src_a, group=row_group, async_op=True)
        cur = 0

        for i in range(summa_dim):
            if i != summa_dim - 1:
                A_list[1 - cur].copy_(A)
                opa[1 - cur] = dist.broadcast(
                    A_list[1 - cur], src=src_a + 1, group=row_group, async_op=True
                )

            if opr[cur] is not None:
                opr[cur].wait()
                if i - 2 == col_rank:
                    C.copy_(C_list[cur])

            if opa[cur] is not None:
                opa[cur].wait()

            torch.matmul(A_list[cur].transpose(0, 1), B, out=C_list[cur])
            opr[cur] = dist.reduce(
                C_list[cur], dst=src_c, group=col_group, async_op=True
            )
            cur = 1 - cur
            src_a += 1
            src_c += summa_dim

        for op in opr:
            op.wait()

        if summa_dim - 2 == col_rank:
            C.copy_(C_list[cur])
        if summa_dim - 1 == col_rank:
            C.copy_(C_list[1 - cur])
        out = C.reshape(out_shape)

        if ctx:
            ctx.summa_dim = summa_dim
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors

        with torch.no_grad():
            A_grad = Matmul_ABT_2D.apply(
                B,
                output_grad,
                ctx.summa_dim,
                ctx.A_shape,
                ctx.row_rank,
                ctx.col_rank,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size,
            )
            B_grad = Matmul_AB_2D.apply(
                A,
                output_grad,
                ctx.summa_dim,
                ctx.B_shape,
                ctx.row_rank,
                ctx.col_rank,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size,
            )
        return (
            A_grad,
            B_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _Add_Bias_2D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        input_: Tensor,
        bias: Tensor,
        output_size_per_partition: int,
        row_rank: int,
        col_rank: int,
        parallel_context: ParallelContext,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
        skip_bias_add: bool,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
    ) -> Tensor:
        bias_temp = all_gather(
            bias,
            -1,
            parallel_mode=col_parallel_mode,
            parallel_context=parallel_context,
        )

        ctx.row_rank = row_rank
        ctx.col_rank = col_rank
        ctx.parallel_context = parallel_context
        ctx.row_parallel_mode = row_parallel_mode
        ctx.col_parallel_mode = col_parallel_mode
        ctx.bias = skip_bias_add
        ctx.data_parallel_rank = data_parallel_rank
        ctx.pipeline_parallel_rank = pipeline_parallel_rank
        ctx.pipeline_parallel_size = pipeline_parallel_size
        ctx.tensor_parallel_size = tensor_parallel_size

        if skip_bias_add:
            return bias_temp
        else:
            output = input_ + bias_temp
            return output

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        col_parallel_mode = ctx.col_parallel_mode
        parallel_context = ctx.parallel_context

        if ctx.bias:
            grad = reduce_scatter(
                output_grad,
                -1,
                parallel_mode=col_parallel_mode,
                parallel_context=parallel_context,
            )
            return (
                None,
                grad,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        else:
            reduce_dim = tuple(range(output_grad.ndim - 1))
            reduce = torch.sum(output_grad, dim=reduce_dim)
            grad = reduce_scatter(
                reduce,
                -1,
                parallel_mode=col_parallel_mode,
                parallel_context=parallel_context,
            )
            return (
                output_grad,
                grad,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )


class _Layernorm_2D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx: Any,
        input_: Tensor,
        E_x: Tensor,
        Var_x: Tensor,
        hidden_size: int,
        parallel_context: ParallelContext,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
    ) -> Tensor:
        input_ = input_ - E_x
        # in here, input = x - E[x], Var_x = 1 / sqrt(Var[x] + eps)
        ctx.normalized_shape = hidden_size
        ctx.parallel_context = parallel_context
        output = input_ * Var_x
        ctx.save_for_backward(output, Var_x)
        ctx.row_parallel_mode = row_parallel_mode
        ctx.col_parallel_mode = col_parallel_mode
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        parallel_context = ctx.parallel_context
        row_parallel_mode = ctx.row_parallel_mode

        x, Var_x = ctx.saved_tensors
        # in here, Var_x = 1 / sqrt(Var[x] + eps), x = (x - E[x]) * Var_x
        output_grad_sum = torch.sum(output_grad, dim=-1, keepdim=True)
        torch.distributed.all_reduce(
            output_grad_sum, group=parallel_context.get_group(row_parallel_mode)
        )
        output_grad_sum /= ctx.normalized_shape

        output_grad_mul_x_sum = torch.sum(output_grad * x, dim=-1, keepdim=True)
        torch.distributed.all_reduce(
            output_grad_mul_x_sum, group=parallel_context.get_group(row_parallel_mode)
        )
        output_grad_mul_x_sum /= ctx.normalized_shape

        input_grad = output_grad.clone()
        input_grad -= x * output_grad_mul_x_sum
        input_grad -= output_grad_sum
        input_grad *= Var_x

        return input_grad, None, None, None, None, None, None


class _AllGatherTensor2D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        inputs: Tensor,
        dim: int,
        parallel_mode: ParallelMode,
        parallel_context: ParallelContext,
    ) -> Tensor:
        ctx.dim = dim
        ctx.parallel_mode = parallel_mode
        ctx.parallel_context = parallel_context

        outputs = all_gather(
            inputs,
            dim,
            parallel_mode=parallel_mode,
            parallel_context=parallel_context,
        )
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        grad = reduce_scatter(
            output_grad,
            ctx.dim,
            parallel_mode=ctx.parallel_mode,
            parallel_context=ctx.parallel_context,
        )
        return grad.contiguous(), None, None, None


class _ReduceTensor2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, parallel_mode, parallel_context):
        return all_reduce(
            input_,
            parallel_mode=parallel_mode,
            parallel_context=parallel_context,
        )

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad, None, None


class _ReduceScatterTensor2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, parallel_mode, parallel_context):
        ctx.dim = dim
        ctx.parallel_mode = parallel_mode
        ctx.parallel_context = parallel_context
        return reduce_scatter(
            input_,
            dim,
            parallel_mode=parallel_mode,
            parallel_context=parallel_context,
        )

    @staticmethod
    def backward(ctx, output_grad):
        return (
            all_gather(
                output_grad,
                ctx.dim,
                parallel_mode=ctx.parallel_mode,
                parallel_context=ctx.parallel_context,
            ),
            None,
            None,
            None,
        )


class _ReduceByBatch2D(torch.autograd.Function):
    @staticmethod
    def symbolic(
        graph, input_, parallel_context: ParallelContext, reduce_mean: bool = False
    ):
        output = all_reduce(
            input_,
            parallel_mode=ParallelMode.TENSOR_2D_COL,
            parallel_context=parallel_context,
        )
        if reduce_mean:
            reduce_size = parallel_context.get_world_size(ParallelMode.TENSOR_2D_COL)
            return output / reduce_size
        return output

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx, input_, parallel_context: ParallelContext, reduce_mean: bool = False
    ):
        output = all_reduce(
            input_,
            parallel_mode=ParallelMode.TENSOR_2D_COL,
            parallel_context=parallel_context,
        )
        ctx.reduce_mean = reduce_mean
        if reduce_mean:
            reduce_size = parallel_context.get_world_size(ParallelMode.TENSOR_2D_COL)
            ctx.reduce_size = reduce_size
            return output.clone() / reduce_size
        return output.clone()

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):
        if ctx.reduce_mean:
            return output_grad / ctx.reduce_size, None, None
        else:
            return output_grad, None, None
