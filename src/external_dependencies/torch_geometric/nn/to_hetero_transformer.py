import copy
import warnings
from collections import defaultdict, deque
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.nn import Module

from torch_geometric.nn.dense.linear import is_uninitialized_parameter
from torch_geometric.nn.fx import Transformer, get_submodule
from torch_geometric.typing import EdgeType, Metadata, NodeType
from torch_geometric.utils.hetero import (
    check_add_self_loops,
    get_unused_node_types,
)

try:
    from torch.fx import Graph, GraphModule, Node
except (ImportError, ModuleNotFoundError, AttributeError):
    GraphModule, Graph, Node = 'GraphModule', 'Graph', 'Node'


def get_dict(mapping: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return mapping if mapping is not None else {}


def to_hetero(module: Module, metadata: Metadata, aggr: str = "sum",
              input_map: Optional[Dict[str, str]] = None,
              debug: bool = False) -> GraphModule:
    r"""Converts a homogeneous GNN model into its heterogeneous equivalent in
    which node representations are learned for each node type in
    :obj:`metadata[0]`, and messages are exchanged between each edge type in
    :obj:`metadata[1]`, as denoted in the `"Modeling Relational Data with Graph
    Convolutional Networks" <https://arxiv.org/abs/1703.06103>`_ paper:

    .. code-block:: python

        import torch
        from torch_geometric.nn import SAGEConv, to_hetero

        class GNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = SAGEConv((-1, -1), 32)
                self.conv2 = SAGEConv((32, 32), 32)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index).relu()
                x = self.conv2(x, edge_index).relu()
                return x

        model = GNN()

        node_types = ['paper', 'author']
        edge_types = [
            ('paper', 'cites', 'paper'),
            ('paper', 'written_by', 'author'),
            ('author', 'writes', 'paper'),
        ]
        metadata = (node_types, edge_types)

        model = to_hetero(model, metadata)
        model(x_dict, edge_index_dict)

    where :obj:`x_dict` and :obj:`edge_index_dict` denote dictionaries that
    hold node features and edge connectivity information for each node type and
    edge type, respectively.

    The below illustration shows the original computation graph of the
    homogeneous model on the left, and the newly obtained computation graph of
    the heterogeneous model on the right:

    .. figure:: ../_figures/to_hetero.svg
      :align: center
      :width: 90%

      Transforming a model via :func:`to_hetero`.

    Here, each :class:`~torch_geometric.nn.conv.MessagePassing` instance
    :math:`f_{\theta}^{(\ell)}` is duplicated and stored in a set
    :math:`\{ f_{\theta}^{(\ell, r)} : r \in \mathcal{R} \}` (one instance for
    each relation in :math:`\mathcal{R}`), and message passing in layer
    :math:`\ell` is performed via

    .. math::

        \mathbf{h}^{(\ell)}_v = \bigoplus_{r \in \mathcal{R}}
        f_{\theta}^{(\ell, r)} ( \mathbf{h}^{(\ell - 1)}_v, \{
        \mathbf{h}^{(\ell - 1)}_w : w \in \mathcal{N}^{(r)}(v) \}),

    where :math:`\mathcal{N}^{(r)}(v)` denotes the neighborhood of :math:`v \in
    \mathcal{V}` under relation :math:`r \in \mathcal{R}`, and
    :math:`\bigoplus` denotes the aggregation scheme :attr:`aggr` to use for
    grouping node embeddings generated by different relations
    (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"` or :obj:`"mul"`).

    Args:
        module (torch.nn.Module): The homogeneous model to transform.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        aggr (str, optional): The aggregation scheme to use for grouping node
            embeddings generated by different relations
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"sum"`)
        input_map (Dict[str, str], optional): A dictionary holding information
            about the type of input arguments of :obj:`module.forward`.
            For example, in case :obj:`arg` is a node-level argument, then
            :obj:`input_map['arg'] = 'node'`, and
            :obj:`input_map['arg'] = 'edge'` otherwise.
            In case :obj:`input_map` is not further specified, will try to
            automatically determine the correct type of input arguments.
            (default: :obj:`None`)
        debug (bool, optional): If set to :obj:`True`, will perform
            transformation in debug mode. (default: :obj:`False`)
    """
    transformer = ToHeteroTransformer(module, metadata, aggr, input_map, debug)
    return transformer.transform()


class ToHeteroTransformer(Transformer):

    aggrs = {
        'sum': torch.add,
        # For 'mean' aggregation, we first sum up all feature matrices, and
        # divide by the number of matrices in a later step.
        'mean': torch.add,
        'max': torch.max,
        'min': torch.min,
        'mul': torch.mul,
    }

    def __init__(
        self,
        module: Module,
        metadata: Metadata,
        aggr: str = 'sum',
        input_map: Optional[Dict[str, str]] = None,
        debug: bool = False,
    ):
        super().__init__(module, input_map, debug)

        self.metadata = metadata
        self.aggr = aggr
        assert len(metadata) == 2
        assert len(metadata[0]) > 0 and len(metadata[1]) > 0
        assert aggr in self.aggrs.keys()

        self.validate()

    def validate(self):
        unused_node_types = get_unused_node_types(*self.metadata)
        if len(unused_node_types) > 0:
            warnings.warn(
                f"There exist node types ({unused_node_types}) whose "
                f"representations do not get updated during message passing "
                f"as they do not occur as destination type in any edge type. "
                f"This may lead to unexpected behavior.")

        names = self.metadata[0] + [rel for _, rel, _ in self.metadata[1]]
        for name in names:
            if not name.isidentifier():
                warnings.warn(
                    f"The type '{name}' contains invalid characters which "
                    f"may lead to unexpected behavior. To avoid any issues, "
                    f"ensure that your types only contain letters, numbers "
                    f"and underscores.")

    def placeholder(self, node: Node, target: Any, name: str):
        # Adds a `get` call to the input dictionary for every node-type or
        # edge-type.
        if node.type is not None:
            Type = EdgeType if self.is_edge_level(node) else NodeType
            node.type = Dict[Type, node.type]

        self.graph.inserting_after(node)

        dict_node = self.graph.create_node('call_function', target=get_dict,
                                           args=(node, ), name=f'{name}_dict')
        self.graph.inserting_after(dict_node)

        for key in self.metadata[int(self.is_edge_level(node))]:
            out = self.graph.create_node('call_method', target='get',
                                         args=(dict_node, key, None),
                                         name=f'{name}__{key2str(key)}')
            self.graph.inserting_after(out)

    def get_attr(self, node: Node, target: Any, name: str):
        raise NotImplementedError

    def call_message_passing_module(self, node: Node, target: Any, name: str):
        # Add calls to edge type-wise `MessagePassing` modules and aggregate
        # the outputs to node type-wise embeddings afterwards.

        module = get_submodule(self.module, target)
        check_add_self_loops(module, self.metadata[1])

        # Group edge-wise keys per destination:
        key_name, keys_per_dst = {}, defaultdict(list)
        for key in self.metadata[1]:
            keys_per_dst[key[-1]].append(key)
            key_name[key] = f'{name}__{key[-1]}{len(keys_per_dst[key[-1]])}'

        for dst, keys in dict(keys_per_dst).items():
            # In case there is only a single edge-wise connection, there is no
            # need for any destination-wise aggregation, and we can already set
            # the intermediate variable name to the final output name.
            if len(keys) == 1:
                key_name[keys[0]] = f'{name}__{dst}'
                del keys_per_dst[dst]

        self.graph.inserting_after(node)
        for key in self.metadata[1]:
            args, kwargs = self.map_args_kwargs(node, key)
            out = self.graph.create_node('call_module',
                                         target=f'{target}.{key2str(key)}',
                                         args=args, kwargs=kwargs,
                                         name=key_name[key])
            self.graph.inserting_after(out)

        # Perform destination-wise aggregation.
        # Here, we aggregate in pairs, popping the first two elements of
        # `keys_per_dst` and append the result to the list.
        for dst, keys in keys_per_dst.items():
            queue = deque([key_name[key] for key in keys])
            i = 1
            while len(queue) >= 2:
                key1, key2 = queue.popleft(), queue.popleft()
                args = (self.find_by_name(key1), self.find_by_name(key2))

                new_name = f'{name}__{dst}'
                if self.aggr == 'mean' or len(queue) > 0:
                    new_name = f'{new_name}_{i}'

                out = self.graph.create_node('call_function',
                                             target=self.aggrs[self.aggr],
                                             args=args, name=new_name)
                self.graph.inserting_after(out)
                queue.append(new_name)
                i += 1

            if self.aggr == 'mean':
                key = queue.popleft()
                out = self.graph.create_node(
                    'call_function', target=torch.div,
                    args=(self.find_by_name(key), len(keys_per_dst[dst])),
                    name=f'{name}__{dst}')
                self.graph.inserting_after(out)

    def call_global_pooling_module(self, node: Node, target: Any, name: str):
        # Add calls to node type-wise `GlobalPooling` modules and aggregate
        # the outputs to graph type-wise embeddings afterwards.
        self.graph.inserting_after(node)
        for key in self.metadata[0]:
            args, kwargs = self.map_args_kwargs(node, key)
            out = self.graph.create_node('call_module',
                                         target=f'{target}.{key2str(key)}',
                                         args=args, kwargs=kwargs,
                                         name=f'{node.name}__{key2str(key)}')
            self.graph.inserting_after(out)

        # Perform node-wise aggregation.
        queue = deque(
            [f'{node.name}__{key2str(key)}' for key in self.metadata[0]])
        i = 1
        while len(queue) >= 2:
            key1, key2 = queue.popleft(), queue.popleft()
            args = (self.find_by_name(key1), self.find_by_name(key2))
            out = self.graph.create_node('call_function',
                                         target=self.aggrs[self.aggr],
                                         args=args, name=f'{name}_{i}')
            self.graph.inserting_after(out)
            queue.append(f'{name}_{i}')
            i += 1

        if self.aggr == 'mean':
            key = queue.popleft()
            out = self.graph.create_node(
                'call_function', target=torch.div,
                args=(self.find_by_name(key), len(self.metadata[0])),
                name=f'{name}_{i}')
            self.graph.inserting_after(out)
        self.replace_all_uses_with(node, out)

    def call_module(self, node: Node, target: Any, name: str):
        if self.is_graph_level(node):
            return

        # Add calls to node type-wise or edge type-wise modules.
        self.graph.inserting_after(node)
        for key in self.metadata[int(self.is_edge_level(node))]:
            args, kwargs = self.map_args_kwargs(node, key)
            out = self.graph.create_node('call_module',
                                         target=f'{target}.{key2str(key)}',
                                         args=args, kwargs=kwargs,
                                         name=f'{name}__{key2str(key)}')
            self.graph.inserting_after(out)

    def call_method(self, node: Node, target: Any, name: str):
        if self.is_graph_level(node):
            return

        # Add calls to node type-wise or edge type-wise methods.
        self.graph.inserting_after(node)
        for key in self.metadata[int(self.is_edge_level(node))]:
            args, kwargs = self.map_args_kwargs(node, key)
            out = self.graph.create_node('call_method', target=target,
                                         args=args, kwargs=kwargs,
                                         name=f'{name}__{key2str(key)}')
            self.graph.inserting_after(out)

    def call_function(self, node: Node, target: Any, name: str):
        if self.is_graph_level(node):
            return

        # Add calls to node type-wise or edge type-wise functions.
        self.graph.inserting_after(node)
        for key in self.metadata[int(self.is_edge_level(node))]:
            args, kwargs = self.map_args_kwargs(node, key)
            out = self.graph.create_node('call_function', target=target,
                                         args=args, kwargs=kwargs,
                                         name=f'{name}__{key2str(key)}')
            self.graph.inserting_after(out)

    def output(self, node: Node, target: Any, name: str):
        # Replace the output by dictionaries, holding either node type-wise or
        # edge type-wise data.
        def _recurse(value: Any) -> Any:
            if isinstance(value, Node):
                if self.is_graph_level(value):
                    return value
                return {
                    key: self.find_by_name(f'{value.name}__{key2str(key)}')
                    for key in self.metadata[int(self.is_edge_level(value))]
                }
            elif isinstance(value, dict):
                return {k: _recurse(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_recurse(v) for v in value]
            elif isinstance(value, tuple):
                return tuple(_recurse(v) for v in value)
            else:
                return value

        if node.type is not None and isinstance(node.args[0], Node):
            output = node.args[0]
            if self.is_node_level(output):
                node.type = Dict[NodeType, node.type]
            elif self.is_edge_level(output):
                node.type = Dict[EdgeType, node.type]
        else:
            node.type = None

        node.args = (_recurse(node.args[0]), )

    def init_submodule(self, module: Module, target: str) -> Module:
        # Replicate each module for each node type or edge type.
        has_node_level_target = bool(
            self.find_by_target(f'{target}.{key2str(self.metadata[0][0])}'))
        has_edge_level_target = bool(
            self.find_by_target(f'{target}.{key2str(self.metadata[1][0])}'))

        if not has_node_level_target and not has_edge_level_target:
            return module

        module_dict = torch.nn.ModuleDict()
        for key in self.metadata[int(has_edge_level_target)]:
            module_dict[key2str(key)] = copy.deepcopy(module)
            if len(self.metadata[int(has_edge_level_target)]) <= 1:
                continue
            if hasattr(module, 'reset_parameters'):
                module_dict[key2str(key)].reset_parameters()
            elif sum([
                    is_uninitialized_parameter(p) or p.numel()
                    for p in module.parameters()
            ]) > 0:
                warnings.warn(
                    f"'{target}' will be duplicated, but its parameters "
                    f"cannot be reset. To suppress this warning, add a "
                    f"'reset_parameters()' method to '{target}'")

        return module_dict

    # Helper methods ##########################################################

    def map_args_kwargs(self, node: Node,
                        key: Union[NodeType, EdgeType]) -> Tuple[Tuple, Dict]:
        def _recurse(value: Any) -> Any:
            if isinstance(value, Node):
                out = self.find_by_name(f'{value.name}__{key2str(key)}')
                if out is not None:
                    return out
                elif isinstance(key, tuple) and key[0] == key[-1]:
                    name = f'{value.name}__{key2str(key[0])}'
                    return self.find_by_name(name)
                elif isinstance(key, tuple) and key[0] != key[-1]:
                    return (
                        self.find_by_name(f'{value.name}__{key2str(key[0])}'),
                        self.find_by_name(f'{value.name}__{key2str(key[-1])}'),
                    )
                else:
                    raise NotImplementedError
            elif isinstance(value, dict):
                return {k: _recurse(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_recurse(v) for v in value]
            elif isinstance(value, tuple):
                return tuple(_recurse(v) for v in value)
            else:
                return value

        args = tuple(_recurse(v) for v in node.args)
        kwargs = {k: _recurse(v) for k, v in node.kwargs.items()}
        return args, kwargs


def key2str(key: Union[NodeType, EdgeType]) -> str:
    key = '__'.join(key) if isinstance(key, tuple) else key
    return key.replace(' ', '_').replace('-', '_').replace(':', '_')