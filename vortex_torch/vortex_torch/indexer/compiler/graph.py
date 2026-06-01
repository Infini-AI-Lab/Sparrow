import torch
from typing import List, Dict, Set, DefaultDict, Tuple, Optional, Callable
from collections import defaultdict, deque
from ..context import Context
from ...abs import vTensor, vOp
from ...utils import Schedule

class UnionFind:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def add(self, x: int):
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class Graph:
    def __init__(
        self,
        tensor_list: List[vTensor],
        op_list: List[vOp],
        output_tensor_to_op_list: List[Optional[int]],
        op_to_input_tensor_list: List[List[int]],
        op_to_output_tensor_list: List[int],
        input_tensor_ids: List[int],
        output_tensor_ids: List[int],
        global_input_tensor_ids: List[int],
        global_output_tensor_ids: List[int],
    ):
        # All ids in these structural fields are local to this graph.
        self.tensor_list: List[vTensor] = tensor_list
        self.op_list: List[vOp] = op_list
        self.output_tensor_to_op_list: List[Optional[int]] = output_tensor_to_op_list
        self.op_to_input_tensor_list: List[List[int]] = op_to_input_tensor_list
        self.op_to_output_tensor_list: List[int] = op_to_output_tensor_list

        # Boundary tensor ids in local graph space.
        self.input_tensor_ids: List[int] = input_tensor_ids
        self.output_tensor_ids: List[int] = output_tensor_ids

        # Boundary tensor ids in original global graph space.
        self.global_input_tensor_ids: List[int] = global_input_tensor_ids
        self.global_output_tensor_ids: List[int] = global_output_tensor_ids
        self.schedule = op_list[0].schedule if (op_list and op_list[0] is not None) else None
        self.forward: Callable = None  # to be filled in later with the generated implementation
    def __repr__(self) -> str:
        return (
            f"Graph("
            f"num_tensors={len(self.tensor_list)}, "
            f"num_ops={len(self.op_list)}, "
            f"input_tensor_ids={self.input_tensor_ids}, "
            f"output_tensor_ids={self.output_tensor_ids}, "
            f"global_input_tensor_ids={self.global_input_tensor_ids}, "
            f"global_output_tensor_ids={self.global_output_tensor_ids})"
        )


def _as_tensor_id_list(x) -> List[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _build_local_graph(
    global_tensor_list: List[vTensor],
    global_op_list: List[vOp],
    global_output_tensor_to_op_list,
    global_op_to_input_tensor_list: List[List[int]],
    global_op_to_output_tensor_list,
    selected_global_op_ids: List[int],
    global_input_tensor_ids: List[int],
    global_output_tensor_ids: List[int],
) -> Graph:
    """
    Build a self-contained local Graph from a subset of global op ids.

    All structural ids inside the returned Graph are local ids.
    Boundary tensors are stored in both local ids and global ids.
    """
    global_op_to_output_tensor_ids: List[List[int]] = [
        _as_tensor_id_list(x) for x in global_op_to_output_tensor_list
    ]

    selected_global_op_set: Set[int] = set(selected_global_op_ids)
    selected_global_tensor_ids: Set[int] = set()

    for global_op_id in selected_global_op_ids:
        for global_tensor_id in global_op_to_input_tensor_list[global_op_id]:
            selected_global_tensor_ids.add(global_tensor_id)

        for global_tensor_id in global_op_to_output_tensor_ids[global_op_id]:
            selected_global_tensor_ids.add(global_tensor_id)

    sorted_global_tensor_ids: List[int] = sorted(selected_global_tensor_ids)

    global_op_id_to_local: Dict[int, int] = {
        global_op_id: local_op_id
        for local_op_id, global_op_id in enumerate(selected_global_op_ids)
    }
    global_tensor_id_to_local: Dict[int, int] = {
        global_tensor_id: local_tensor_id
        for local_tensor_id, global_tensor_id in enumerate(sorted_global_tensor_ids)
    }

    local_tensor_list: List[vTensor] = [
        global_tensor_list[global_tensor_id]
        for global_tensor_id in sorted_global_tensor_ids
    ]
    local_op_list: List[vOp] = [
        global_op_list[global_op_id]
        for global_op_id in selected_global_op_ids
    ]

    local_output_tensor_to_op_list: List[Optional[int]] = []
    for global_tensor_id in sorted_global_tensor_ids:
        global_producer_op_id = global_output_tensor_to_op_list[global_tensor_id]

        if global_producer_op_id is None:
            local_output_tensor_to_op_list.append(None)
        elif global_producer_op_id in selected_global_op_set:
            local_output_tensor_to_op_list.append(
                global_op_id_to_local[global_producer_op_id]
            )
        else:
            local_output_tensor_to_op_list.append(None)

    local_op_to_input_tensor_list: List[List[int]] = []
    for global_op_id in selected_global_op_ids:
        local_inputs = [
            global_tensor_id_to_local[global_tensor_id]
            for global_tensor_id in global_op_to_input_tensor_list[global_op_id]
        ]
        local_op_to_input_tensor_list.append(local_inputs)

    local_op_to_output_tensor_list: List[int] = []
    for global_op_id in selected_global_op_ids:
        global_outputs = global_op_to_output_tensor_ids[global_op_id]
        if len(global_outputs) != 1:
            raise RuntimeError(
                f"Graph.op_to_output_tensor_list expects one output tensor per op, "
                f"but op {global_op_id} has outputs {global_outputs}"
            )
        local_op_to_output_tensor_list.append(
            global_tensor_id_to_local[global_outputs[0]]
        )

    local_input_tensor_ids = [
        global_tensor_id_to_local[global_tensor_id]
        for global_tensor_id in global_input_tensor_ids
    ]
    local_output_tensor_ids = [
        global_tensor_id_to_local[global_tensor_id]
        for global_tensor_id in global_output_tensor_ids
    ]

    return Graph(
        tensor_list=local_tensor_list,
        op_list=local_op_list,
        output_tensor_to_op_list=local_output_tensor_to_op_list,
        op_to_input_tensor_list=local_op_to_input_tensor_list,
        op_to_output_tensor_list=local_op_to_output_tensor_list,
        input_tensor_ids=local_input_tensor_ids,
        output_tensor_ids=local_output_tensor_ids,
        global_input_tensor_ids=list(global_input_tensor_ids),
        global_output_tensor_ids=list(global_output_tensor_ids),
    )


def contruct_graph(ctx: Context) -> Tuple[Graph, List[Graph]]:

    tensor_list: List[vTensor] = ctx.tensor_list
    op_list: List[vOp] = ctx.op_list
    output_tensor_to_op_list = ctx.output_tensor_to_op_list
    op_to_input_tensor_list: List[List[int]] = ctx.op_to_input_tensor_list
    op_to_output_tensor_list = ctx.op_to_output_tensor_list
    for i, op in enumerate(op_list):
        print(f"Op {i}: {op}, inputs={op_to_input_tensor_list[i]}, outputs={op_to_output_tensor_list[i]}")
    
    final_output_tensor_id = 1

    # =========================================================
    # 1. Normalize global op outputs
    # =========================================================
    op_to_output_tensor_ids: List[List[int]] = [
        _as_tensor_id_list(x) for x in op_to_output_tensor_list
    ]

    # =========================================================
    # 2. Build global producer/consumer mappings
    # =========================================================
    tensor_to_consumers: DefaultDict[int, List[int]] = defaultdict(list)

    for consumer_op_id, input_tensor_ids in enumerate(op_to_input_tensor_list):
        for tensor_id in input_tensor_ids:
            tensor_to_consumers[tensor_id].append(consumer_op_id)

    # =========================================================
    # 3. Reverse search from final output tensor
    # Post-order DFS gives op-level topo order.
    # =========================================================
    final_producer_op_id = output_tensor_to_op_list[final_output_tensor_id]
    if final_producer_op_id is None:
        raise RuntimeError(
            f"final_output_tensor_id={final_output_tensor_id} has no producer op"
        )

    reachable_ops: Set[int] = set()
    visited_ops: Set[int] = set()
    topo_op_ids: List[int] = []

    def dfs_op(global_op_id: int):
        if global_op_id in visited_ops:
            return
        visited_ops.add(global_op_id)

        for global_input_tensor_id in op_to_input_tensor_list[global_op_id]:
            global_producer_op_id = output_tensor_to_op_list[global_input_tensor_id]
            if global_producer_op_id is not None:
                dfs_op(global_producer_op_id)

        reachable_ops.add(global_op_id)
        topo_op_ids.append(global_op_id)

    dfs_op(final_producer_op_id)

    # =========================================================
    # 4. Merge reachable W ops into subgraphs
    # S ops stay standalone.
    # =========================================================
    uf = UnionFind()

    for global_op_id in topo_op_ids:
        if op_list[global_op_id].schedule == Schedule.W:
            uf.add(global_op_id)

    for consumer_op_id in topo_op_ids:
        if op_list[consumer_op_id].schedule != Schedule.W:
            continue

        for input_tensor_id in op_to_input_tensor_list[consumer_op_id]:
            producer_op_id = output_tensor_to_op_list[input_tensor_id]
            if producer_op_id is None:
                continue
            if producer_op_id not in reachable_ops:
                continue
            if op_list[producer_op_id].schedule != Schedule.W:
                continue

            uf.union(producer_op_id, consumer_op_id)

    # =========================================================
    # 5. Assign each reachable op to a subgraph key
    # =========================================================
    op_to_sg_key: Dict[int, tuple] = {}

    for global_op_id in topo_op_ids:
        if op_list[global_op_id].schedule == Schedule.S:
            op_to_sg_key[global_op_id] = ("S", global_op_id)
        else:
            op_to_sg_key[global_op_id] = ("W", uf.find(global_op_id))

    # =========================================================
    # 6. Collect subgraph op ids in op-topological order
    # =========================================================
    sg_key_to_op_ids: DefaultDict[tuple, List[int]] = defaultdict(list)
    sg_key_order: List[tuple] = []
    seen_sg_keys: Set[tuple] = set()

    for global_op_id in topo_op_ids:
        sg_key = op_to_sg_key[global_op_id]
        if sg_key not in seen_sg_keys:
            seen_sg_keys.add(sg_key)
            sg_key_order.append(sg_key)
        sg_key_to_op_ids[sg_key].append(global_op_id)

    # =========================================================
    # 7. Build subgraph DAG and topo sort subgraphs
    # =========================================================
    sg_key_to_id: Dict[tuple, int] = {
        sg_key: i for i, sg_key in enumerate(sg_key_order)
    }

    sg_graph: DefaultDict[int, Set[int]] = defaultdict(set)
    sg_indegree: Dict[int, int] = {i: 0 for i in range(len(sg_key_order))}

    for consumer_op_id in topo_op_ids:
        consumer_sg_id = sg_key_to_id[op_to_sg_key[consumer_op_id]]

        for input_tensor_id in op_to_input_tensor_list[consumer_op_id]:
            producer_op_id = output_tensor_to_op_list[input_tensor_id]
            if producer_op_id is None:
                continue
            if producer_op_id not in reachable_ops:
                continue

            producer_sg_id = sg_key_to_id[op_to_sg_key[producer_op_id]]
            if producer_sg_id == consumer_sg_id:
                continue

            if consumer_sg_id not in sg_graph[producer_sg_id]:
                sg_graph[producer_sg_id].add(consumer_sg_id)
                sg_indegree[consumer_sg_id] += 1

    queue = deque([sg_id for sg_id, deg in sg_indegree.items() if deg == 0])
    topo_sg_ids: List[int] = []

    while queue:
        sg_id = queue.popleft()

        topo_sg_ids.append(sg_id)
        for next_sg_id in sg_graph[sg_id]:
            sg_indegree[next_sg_id] -= 1
            if sg_indegree[next_sg_id] == 0:
                queue.append(next_sg_id)

    if len(topo_sg_ids) != len(sg_key_order):
        raise RuntimeError("Cycle detected in subgraph DAG")

    # =========================================================
    # 8. Compute full-graph boundary tensors in global ids
    # =========================================================
    full_global_input_tensor_set: Set[int] = set()
    for global_op_id in topo_op_ids:
        for global_tensor_id in op_to_input_tensor_list[global_op_id]:
            global_producer_op_id = output_tensor_to_op_list[global_tensor_id]
            if global_producer_op_id is None or global_producer_op_id not in reachable_ops:
                full_global_input_tensor_set.add(global_tensor_id)

    full_global_input_tensor_ids = sorted(full_global_input_tensor_set)
    full_global_output_tensor_ids = [final_output_tensor_id]

    # =========================================================
    # 9. Build self-contained full graph
    # =========================================================
    full_graph = _build_local_graph(
        global_tensor_list=tensor_list,
        global_op_list=op_list,
        global_output_tensor_to_op_list=output_tensor_to_op_list,
        global_op_to_input_tensor_list=op_to_input_tensor_list,
        global_op_to_output_tensor_list=op_to_output_tensor_list,
        selected_global_op_ids=topo_op_ids,
        global_input_tensor_ids=full_global_input_tensor_ids,
        global_output_tensor_ids=full_global_output_tensor_ids,
    )

    # =========================================================
    # 10. Build self-contained subgraphs
    # =========================================================
    op_to_subgraph_id: Dict[int, int] = {}
    for new_sg_id, old_sg_id in enumerate(topo_sg_ids):
        sg_key = sg_key_order[old_sg_id]
        for global_op_id in sg_key_to_op_ids[sg_key]:
            op_to_subgraph_id[global_op_id] = new_sg_id

    subgraphs: List[Graph] = []

    for new_sg_id, old_sg_id in enumerate(topo_sg_ids):
        sg_key = sg_key_order[old_sg_id]
        sg_global_op_ids = sg_key_to_op_ids[sg_key]

        global_input_tensor_set: Set[int] = set()
        global_output_tensor_set: Set[int] = set()

        for global_op_id in sg_global_op_ids:
            for global_tensor_id in op_to_input_tensor_list[global_op_id]:
                global_producer_op_id = output_tensor_to_op_list[global_tensor_id]
                if global_producer_op_id is None:
                    global_input_tensor_set.add(global_tensor_id)
                elif global_producer_op_id in op_to_subgraph_id and op_to_subgraph_id[global_producer_op_id] != new_sg_id:
                    global_input_tensor_set.add(global_tensor_id)

        for global_op_id in sg_global_op_ids:
            for global_out_tensor_id in op_to_output_tensor_ids[global_op_id]:
                consumers = tensor_to_consumers.get(global_out_tensor_id, [])

                cross_subgraph_used = False
                for consumer_op_id in consumers:
                    if consumer_op_id not in op_to_subgraph_id:
                        continue
                    if op_to_subgraph_id[consumer_op_id] != new_sg_id:
                        cross_subgraph_used = True
                        break

                if cross_subgraph_used or global_out_tensor_id == final_output_tensor_id:
                    global_output_tensor_set.add(global_out_tensor_id)

        subgraph = _build_local_graph(
            global_tensor_list=tensor_list,
            global_op_list=op_list,
            global_output_tensor_to_op_list=output_tensor_to_op_list,
            global_op_to_input_tensor_list=op_to_input_tensor_list,
            global_op_to_output_tensor_list=op_to_output_tensor_list,
            selected_global_op_ids=sg_global_op_ids,
            global_input_tensor_ids=sorted(global_input_tensor_set),
            global_output_tensor_ids=sorted(global_output_tensor_set),
        )
        subgraphs.append(subgraph)

    for i, subgraph in enumerate(subgraphs):
        print(f"Subgraph {i}: {subgraph}")
    return full_graph, subgraphs