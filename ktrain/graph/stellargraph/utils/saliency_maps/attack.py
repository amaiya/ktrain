# -*- coding: utf-8 -*-
#
# Copyright 2018 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .....imports import *

import numpy as np


def remove_topk_nodes(adj, node_importance_rank, topk):

    """
    Remove top k important nodes by removing all the links connecting to the nodes.

    Args:
        adj: The adjacency matrix for the graph.
        node_importance_rank: The sorted indices of nodes following the node importance (descending) order.
        topk: The max number of nodes to remove.

    Returns:
        The modified adjacency matrix.
    """
    for idx in node_importance_rank[:topk]:
        for i in range(adj.shape[0]):
            adj[i, idx] = 0
            adj[idx, i] = 0

    return adj

def perturb_topk_edges(adj, edge_importance_rank, topk, remove = True):
    """
    Remove topk important edges from the graph.

    Note that removing edges is different from removing nodes. The node importance is computed in a way that only nodes
    existed in the ego graph have non-zero importance. Consequently, a node to remove always exists. However, an edge
    to remove may not exist in the graph. In other words, the corresponding entry in adj was 0. Therefore, we ignore those
    non-existing edges as they are not able to be removed.

    Args:
        adj: The adjacency matrix for the graph.
        edge_importance_rank: The sorted indices of edges following the edge importance (descending) order.
        topk: The max number of edges to remove.
        remove (bool): Setting it to False for adding edges.

    Returns:
        The modified adjacency matrix.
    """
    removed_cnt = 0
    perturb_target = int(remove)
    for edge in edge_importance_rank:
        if adj[edge[0], edge[1]] != perturb_target:
            if remove:
                print('remove edge {} -> {}'.format(edge[0], edge[1]))
            else:
                print('add edge {} -> {}'.format(edge[0], edge[1]))

            adj[edge[0], edge[1]] = perturb_target
            removed_cnt += 1
        if removed_cnt == topk:
            break
    return adj

