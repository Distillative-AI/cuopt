/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct gub_node_t {
  i_t var_idx;
  i_t cstr_idx;
};

// this is the GUB constraint implementation from Conflict graphs in solving integer programming
// problems (Atamturk et.al.) this is a four-way linked list, vertical direction keeps the GUB
// constraint that a variable takes part horizontal direction keeps all the vars in the current GUB
// constraint the directions are sorted by the index to make the search easier
template <typename i_t, typename f_t>
struct gub_linked_list_t {
  view_t view() { return view_t{nodes.data(), right.data(), left.data(), up.data(), down.data()}; }

  struct view_t {
    raft::device_span<gub_node_t<i_t, f_t>> nodes;
    raft::device_span<i_t> right;
    raft::device_span<i_t> left;
    raft::device_span<i_t> up;
    raft::device_span<i_t> down;
  };
  rmm::device_uvector<gub_node_t<i_t, f_t>> nodes;
  // the vectors keep the indices to the nodes above
  rmm::device_uvector<i_t> right;
  rmm::device_uvector<i_t> left;
  rmm::device_uvector<i_t> up;
  rmm::device_uvector<i_t> down;
};

}  // namespace cuopt::linear_programming::detail

// Rounding Procedure:

// fix set of variables x_1, x_2, x_3,... in a bulk. Consider sorting according largest size GUB
// constraint(or some other criteria).

// compute new activities on changed constraints, given that x_1=v_1, x_2=v_2, x_3=v_3:

// 	if the current constraint is GUB

// 		if at least two binary vars(note that some can be full integer) are common: (needs
// binary_vars_in_bulk^2 number of checks)

// 			return infeasible

// 		else

// 			set L_r to 1.

// 	else(non-GUB constraints)

// 		greedy clique partitioning algorithm:

// 			set L_r = sum(all positive coefficients on binary vars) + sum(min_activity contribution on
// non-binary vars) # note that the paper doesn't contain this part, since it only deals with binary

// 			# iterate only on binary variables(i.e. vertices of B- and complements of B+)

// 			start with highest weight vertex (v) among unmarked and mark it

// 			find maximal clique among unmarked containing the vertex: (there are various algorithms to
// find maximal clique)

// 				max_clique = {v}

// 				L_r -= w_v

// 				# prioritization is on higher weight vertex when there are equivalent max cliques?
//                 # we could try BFS to search multiple greedy paths
// 				for each unmarked vertex(w):

// 					counter = 0

// 					for each vertex(k) in max_clique:

// 						if(check_if_pair_shares_an_edge(w,k))

// 							counter++

// 					if counter == max_clique.size()

// 						max_clique = max_clique U {w}

// 						mark w as marked

// 			if(L_r > UB) return infeasible

// remove all fixed variables(original and newly propagated) from the conflict graph. !!!!!! still a
// bit unclear how to remove it from the adjaceny list data structure since it only supports
// additions!!!!

// add newly discovered GUB constraints into dynamic adjacency list

// do double probing to infer new edges(we need a heuristic to choose which pairs to probe)

// check_if_pair_shares_an_edge(w,v):

// 	check GUB constraints by traversing the double linked list:

// 		on the column of variable w:

// 		for each row:

// 			if v is contained on the row

// 				return true

// 	check added edges on adjacency list:

// 		k <- last[w]

// 		while k != 0

// 			if(adj[k] == v)

// 				return true

// 			k <-next[k]

// 	return false
