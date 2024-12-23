import random
from collections import defaultdict
from ..network.Placeholder import Placeholder


class Graph():

    def based_on_feed_dict_create_graph(self, feed_dict):

        nodes = [n for n in feed_dict]  # know all the placeholder

        computing_graph = defaultdict(list)

        while nodes:
            n = nodes.pop(0)

            if isinstance(n, Placeholder):
                n.value = feed_dict[n]

            if n in computing_graph: continue

            for m in n.outputs:
                computing_graph[n].append(m)
                nodes.append(m)

        return computing_graph

    def toplogic(self, graph):
        sorted_node = []

        while len(graph) > 0:

            all_inputs = []
            all_outputs = []

            for n in graph:
                all_inputs += graph[n]
                all_outputs.append(n)

            all_inputs = set(all_inputs)
            all_outputs = set(all_outputs)

            need_remove = all_outputs - all_inputs  # which in all_inputs but not in all_outputs

            if len(need_remove) > 0:
                node = random.choice(list(need_remove))

                need_to_visited = [node]

                if len(graph) == 1: need_to_visited += graph[node]

                graph.pop(node)
                sorted_node += need_to_visited

                for _, links in graph.items():
                    if node in links: links.remove(node)
            else:  # have cycle
                break

        return sorted_node

    def node_compting_sort(self, feed_dict):
        graph = self.based_on_feed_dict_create_graph(feed_dict)

        return self.toplogic(graph)

graph = Graph()