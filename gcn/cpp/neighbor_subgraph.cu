#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;
const int MAX_DEGREE = 50;

/*
G.add_edges_from([
    (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 2), (2, 3), (3, 4)
])

vector<pair<int, int> edge_index {{0,1}, {0,2}, {0,3}, {0,4}, {1,2}, {2,3}, {3,4}};
vector<int> edge_index = {0,1, 0,2, 0,3, 0,4, 1,2, 2,3, 3,4};
*/
vector<int> loadEdgeIndexFromFile(const string& filename) {
    vector<int> edge_index;

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
            // edge_index.push_back(make_pair(source, destination));
            // edge_index[0].push_back(source);
            // edge_index[1].push_back(destination);
            edge_index.push_back(source);
            edge_index.push_back(destination);
        }
        iss_src >> source;
        iss_dst >> destination;
        // edge_index.push_back(make_pair(source, destination));
        // edge_index[0].push_back(source);
        // edge_index[1].push_back(destination);
        edge_index.push_back(source);
        edge_index.push_back(destination);
        file.close();
    } else {
        cerr << "Failed to open file: " << filename << endl;
    }

    return edge_index;
}


__global__
void one_hop_subgraph_kernel(
    int num_hops, int* d_edge_index, int num_nodes, int num_edges, int* d_neighbor_subgraph_nodes, int* d_neighbor_subgraph_node_count) {
    
    int node_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_idx < num_nodes) {
        // d_neighbor_subgraph_nodes[0] = node_idx;
        int idx = 0;
        d_neighbor_subgraph_nodes[node_idx * MAX_DEGREE + idx] = node_idx;
        d_neighbor_subgraph_node_count[node_idx] ++;
        idx ++;

        for (int i = 0; i < num_edges; i ++) {
            if (node_idx == d_edge_index[2 * i]) {
                d_neighbor_subgraph_nodes[node_idx * MAX_DEGREE + idx] = d_edge_index[2 * i + 1];
                d_neighbor_subgraph_node_count[node_idx] ++;
                idx ++;
            } 
            // else if (node_idx == d_edge_index[2 * i + 1]) {
            //     d_neighbor_subgraph_nodes[node_idx * MAX_DEGREE + idx] = d_edge_index[2 * i];
            //     d_neighbor_subgraph_node_count[node_idx] ++;
            //     idx ++;
            // }
        }
    }
}


vector<vector<int>> get_all_neighbor_subgraph(const int num_nodes, const int num_edges, const vector<int>& edge_index, int num_hops) {
    // cout << "num_nodes = " << num_nodes << endl;
    // cout << "num_edges = " << num_edges << endl;

    int* d_edge_index;
    int* d_neighbor_subgraph_nodes;
    int* d_neighbor_subgraph_node_count;

    cudaMalloc((void**)&d_edge_index, sizeof(int) * num_edges * 2);
    cudaMalloc((void**)&d_neighbor_subgraph_nodes, sizeof(int) * num_nodes * MAX_DEGREE);
    cudaMalloc((void**)&d_neighbor_subgraph_node_count, sizeof(int) * num_nodes);
    cudaMemcpy(d_edge_index, edge_index.data(), sizeof(int) * num_edges * 2, cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int gridSize = (num_nodes + blockSize - 1) / blockSize;

    one_hop_subgraph_kernel<<<gridSize, blockSize>>>(num_hops, d_edge_index, num_nodes, num_edges, d_neighbor_subgraph_nodes, d_neighbor_subgraph_node_count);

    
    vector<int> h_neighbor_subgraph_node_count(num_nodes, 0);
    cudaMemcpy(h_neighbor_subgraph_node_count.data(), d_neighbor_subgraph_node_count, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);
    
    vector<int> h_neighbor_subgraph_nodes(num_nodes * MAX_DEGREE);
    cudaMemcpy(h_neighbor_subgraph_nodes.data(), d_neighbor_subgraph_nodes, sizeof(int) * num_nodes * MAX_DEGREE, cudaMemcpyDeviceToHost);

    
    cudaFree(d_edge_index);
    cudaFree(d_neighbor_subgraph_nodes);
    cudaFree(d_neighbor_subgraph_node_count);

    vector<vector<int>> all_neighbor_subgraph(num_nodes);
    for (int i = 0; i < h_neighbor_subgraph_nodes.size(); i ++) {
        int central_node_idx = i / MAX_DEGREE;
        int neighbor_node_idx = i % MAX_DEGREE;
        if (neighbor_node_idx < h_neighbor_subgraph_node_count[central_node_idx]) {
            all_neighbor_subgraph[central_node_idx].push_back(h_neighbor_subgraph_nodes[i]);
        }
    }

    return all_neighbor_subgraph;
}


vector<int> intersect(vector<int>& array1, vector<int>& array2) {
    vector<int> result;
    sort(array1.begin(), array1.end());
    sort(array2.begin(), array2.end());
    set_intersection(array1.begin(), array1.end(), array2.begin(), array2.end(), back_inserter(result));
    return result;
}


// pair<int, int> subgraph(const vector<int>& nodes, const vector<pair<int, int>>& edge_index) {
//     unordered_set<int> node_set(nodes.begin(), nodes.end());

//     int num_nodes = 0;
//     int num_edges = 0;

//     for (const auto& [u, v] : edge_index) {
//         if (node_set.count(u) > 0 and node_set.count(v) > 0) {
//             num_edges++;
//         }
//     }
//     num_nodes = node_set.size();
//     return make_pair(num_nodes, num_edges / 2);
// }


pair<int, int> subgraph(const vector<int>& nodes, const vector<int>& edge_index) {
    unordered_set<int> node_set(nodes.begin(), nodes.end());

    int num_nodes = 0;
    int num_edges = 0;

    for (int i = 0; i < num_edges; i ++) {
        int u = edge_index[2 * i], v = edge_index[2 * i + 1];
        if (node_set.count(u) > 0 and node_set.count(v) > 0) {
            num_edges++;
        }
    }
    num_nodes = node_set.size();
    return make_pair(num_nodes, num_edges / 2);
}


pair<int, int> get_overlap_subgraph(vector<int> u_subset, vector<int> v_subset, const vector<int>& edge_index) {
    vector<int> overlapSubset = intersect(u_subset, v_subset);
    return subgraph(overlapSubset, edge_index);
}


// vector<vector<double>> get_sc_matrix(const int num_nodes, const vector<pair<int, int>>& edge_index, int lambda_) {
//     vector<vector<int>> all_neighbor_subgraph = get_all_neighbor_subgraph(num_nodes, edge_index, 1);
//     vector<vector<double>> structural_coeff(num_nodes, vector<double>(num_nodes, 0.));
//     for (const auto [u, v] : edge_index) {
//         const auto [s_num_node, s_num_edge] = get_overlap_subgraph(all_neighbor_subgraph[u], all_neighbor_subgraph[v], edge_index);
//         // structural_coeff[u][v] = s_num_edge * pow(s_num_node, lambda_) / (s_num_node * (s_num_node - 1));
//         // printf("(%d, %d)\ts_num_node=%d\ts_num_edge=%d\n", u, v, s_num_node, s_num_edge);
//         if (s_num_node <= 1) {
//             continue;
//         }
//         double sc = (double)s_num_edge * s_num_node / (s_num_node * (s_num_node - 1));
//         structural_coeff[u][v] = sc;
//     }
//     return structural_coeff;
// }

vector<vector<double>> get_sc_matrix(const int num_nodes, const int num_edges, const vector<int>& edge_index, int lambda_) {
    vector<vector<int>> all_neighbor_subgraph = get_all_neighbor_subgraph(num_nodes, num_edges, edge_index, 1);
    vector<vector<double>> structural_coeff(num_nodes, vector<double>(num_nodes, 0.));
    for (int i = 0; i < num_edges; i ++) {
        int u = edge_index[2 * i], v = edge_index[2 * i + 1];
        const auto [s_num_node, s_num_edge] = get_overlap_subgraph(all_neighbor_subgraph[u], all_neighbor_subgraph[v], edge_index);
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
    // for (int i = 0; i < data.size(); i ++) {
    //     cout << i << "\t" << data[i] << endl;
    // }
}


int main() {
    // [Demo] load cora edge index
    int num_nodes = 2708;
    string filename = "../out/cora_edge_index.csv";
    vector<int> edge_index = loadEdgeIndexFromFile(filename);
    int num_edges = edge_index.size() / 2;

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
    clock_t start = clock();
    vector<vector<int>> all_neighbor_subgraph = get_all_neighbor_subgraph(num_nodes, num_edges, edge_index, 1);
    clock_t end = clock();
    for (int u = 0; u < num_nodes; u ++) {
        print_iterable(all_neighbor_subgraph[u]);
    }
    
    // double duration = double(end - start) / CLOCKS_PER_SEC;
    // cout << "Execution time: " << duration << " seconds" << endl;

    // auto u_subset = k_hop_subgraph({2706}, 1, edgeIndex);
    // auto v_subset = k_hop_subgraph({2707}, 1, edgeIndex);
    // auto overlap_subset = intersect(u_subset, v_subset);
    // print_iterable(u_subset);
    // print_iterable(v_subset);

    // // [Demo] get_sc_matrix
    // clock_t start = clock();
    // vector<vector<double>> structural_coeff = get_sc_matrix(num_nodes, num_edges, edge_index, 1);
    // clock_t end = clock();
    // double duration = double(end - start) / CLOCKS_PER_SEC;
    // cout << "Execution time: " << duration << " seconds" << endl;

    // int m = structural_coeff.size(), n = structural_coeff[0].size();
    // for (int r = 0; r < m; r ++) {
    //     // cout << "row " << r << ": "; 
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
