#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

using namespace std;
using EdgeIndex = vector<pair<int, int>>;

void print_iterable(vector<int> data);

EdgeIndex loadEdgeIndexFromFile(const string& filename) {
    EdgeIndex edgeIndex;

    ifstream file(filename);
    if (file.is_open()) {
        string line;

        // Read the first line representing source nodes
        getline(file, line);
        istringstream iss_src(line);

        // Read the corresponding destination nodes from the second line
        getline(file, line);
        istringstream iss_dst(line);

        int source, destination;
        char comma;
        while (iss_src >> source >> comma && iss_dst >> destination >> comma) {
            edgeIndex.push_back(make_pair(source, destination));
        }
        iss_src >> source;
        iss_dst >> destination;
        edgeIndex.push_back(make_pair(source, destination));

        file.close();
    } else {
        cerr << "Failed to open file: " << filename << endl;
    }

    return edgeIndex;
}


vector<int> k_hop_subgraph(
    const vector<int>& node_idx, int num_hops, 
    const vector<pair<int, int>>& edgeIndex, 
    bool relabel_nodes = false, int num_nodes = -1, string flow = "source_to_target", bool directed = false) {
    vector<int> subgraph_nodes;
    unordered_set<int> visited;

    if (num_nodes == -1) {
        for (const auto& edge : edgeIndex) {
            num_nodes = max(num_nodes, max(edge.first, edge.second) + 1);
        }
    }

    for (const auto& node : node_idx) {
        visited.insert(node);
        subgraph_nodes.push_back(node);
    }

    for (int i = 0; i < num_hops; ++i) {
        unordered_set<int> new_nodes;
        for (const auto& edge : edgeIndex) {
            if (visited.count(edge.first) > 0) {
                new_nodes.insert(edge.second);
            } 
            // else if (!directed && visited.count(edge.second) > 0) {
            //     new_nodes.insert(edge.first);
            // }
        }

        for (const auto& node : new_nodes) {
            visited.insert(node);
            subgraph_nodes.push_back(node);
        }
    }

    if (relabel_nodes) {
        unordered_map<int, int> node_map;
        int new_label = 0;
        for (const auto& node : visited) {
            node_map[node] = new_label++;
        }
        for (auto& node : subgraph_nodes) {
            node = node_map[node];
        }
    }

    return subgraph_nodes;
}


vector<vector<int>> get_all_neighbor_subgraph(const int num_nodes, const vector<pair<int, int>>& edgeIndex, int num_hops) {
    vector<vector<int>> subgraph_list;
    for (int nodeIdx = 0; nodeIdx < num_nodes; nodeIdx ++) {
        subgraph_list.push_back(k_hop_subgraph({nodeIdx}, 1, edgeIndex));
        print_iterable(subgraph_list[nodeIdx]);
    }
    return subgraph_list;
}


vector<int> intersect(vector<int>& array1, vector<int>& array2) {
    vector<int> result;
    sort(array1.begin(), array1.end());
    sort(array2.begin(), array2.end());
    set_intersection(array1.begin(), array1.end(), array2.begin(), array2.end(), back_inserter(result));
    return result;
}


pair<int, int> subgraph(const vector<int>& nodes, const vector<pair<int, int>>& edge_index) {
    unordered_set<int> node_set(nodes.begin(), nodes.end());

    int num_nodes = 0;
    int num_edges = 0;

    for (const auto& [u, v] : edge_index) {
        if (node_set.count(u) > 0 and node_set.count(v) > 0) {
            num_edges++;
        }
    }
    num_nodes = node_set.size();
    return make_pair(num_nodes, num_edges / 2);
}


pair<int, int> get_overlap_subgraph(vector<int> u_subset, vector<int> v_subset, const vector<pair<int, int>>& edgeIndex) {
    vector<int> overlapSubset = intersect(u_subset, v_subset);
    return subgraph(overlapSubset, edgeIndex);
}


vector<vector<double>> get_sc_matrix(const int num_nodes, const vector<pair<int, int>>& edgeIndex, int lambda_) {
    vector<vector<int>> subgraph_list = get_all_neighbor_subgraph(num_nodes, edgeIndex, 1);
    vector<vector<double>> structural_coeff(num_nodes, vector<double>(num_nodes, 0.));
    for (const auto [u, v] : edgeIndex) {
        const auto [s_num_node, s_num_edge] = get_overlap_subgraph(subgraph_list[u], subgraph_list[v], edgeIndex);
        // structural_coeff[u][v] = s_num_edge * pow(s_num_node, lambda_) / (s_num_node * (s_num_node - 1));
        // printf("(%d, %d)\ts_num_node=%d\ts_num_edge=%d\n", u, v, s_num_node, s_num_edge);
        if (s_num_node <= 1) {
            continue;
        }
        double sc = (double)s_num_edge * s_num_node / (s_num_node * (s_num_node - 1));
        structural_coeff[u][v] = sc;
    }
    return structural_coeff;
}

void print_iterable(vector<int> data) {
    for (int x : data) {
        cout << x << " ";
    }
    cout << endl;
}


int main() {
    // [Demo] load cora edge index
    string filename = "../out/cora_edge_index.csv";
    EdgeIndex edgeIndex = loadEdgeIndexFromFile(filename);
    int num_nodes = 0;
    for (const auto& edge : edgeIndex) {
        num_nodes = max(num_nodes, max(edge.first, edge.second) + 1);
    }

    // [Test] G1
    // string filename = "../out/G1_edge_index.csv";
    // EdgeIndex edgeIndex = loadEdgeIndexFromFile(filename);
    // for (const auto [u, v] : edgeIndex) {
    //     printf("(%d, %d)\n", u, v);
    // }

    // int num_nodes = 0;
    // for (const auto& edge : edgeIndex) {
    //     num_nodes = max(num_nodes, max(edge.first, edge.second) + 1);
    // }
    // vector<vector<int>> subgraph_list = get_all_neighbor_subgraph(num_nodes, edgeIndex, 1);
    // for (int u = 0; u < num_nodes; u ++) {
    //     print_iterable(subgraph_list[u]);
    // }

    // [Demo] get_all_neighbor_subgraph  
    // clock_t start = clock();
    // vector<vector<int>> subgraph_list = get_all_neighbor_subgraph(num_nodes, edgeIndex, 1);
    // for (int u = 0; u < num_nodes; u ++) {
    //     print_iterable(subgraph_list[u]);
    // }
    // clock_t end = clock();
    // double duration = double(end - start) / CLOCKS_PER_SEC;
    // cout << "Execution time: " << duration << " seconds" << endl;

    // auto u_subset = k_hop_subgraph({2706}, 1, edgeIndex);
    // auto v_subset = k_hop_subgraph({2707}, 1, edgeIndex);
    // auto overlap_subset = intersect(u_subset, v_subset);
    // print_iterable(u_subset);
    // print_iterable(v_subset);

    // [Demo] get_sc_matrix
    clock_t start = clock();
    vector<vector<double>> structural_coeff = get_sc_matrix(num_nodes, edgeIndex, 1);
    clock_t end = clock();
    double duration = double(end - start) / CLOCKS_PER_SEC;
    cout << "Execution time: " << duration << " seconds" << endl;

    // int m = structural_coeff.size(), n = structural_coeff[0].size();
    // for (int r = 0; r < m; r ++) {
    //     cout << "row " << r << ": "; 
    //     for (int c = 0; c < n; c ++) {
    //         if (c == 0) {
    //             printf("%.2f", structural_coeff[r][c]);
    //         } else {
    //             printf(",%.2f", structural_coeff[r][c]);
    //         }
    //         // cout << structural_coeff[r][c] << " ";
    //     }
    //     cout << endl;
    // }
}
