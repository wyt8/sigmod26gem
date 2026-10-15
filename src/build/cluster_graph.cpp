#include "graph/index_impl.h"
#include "distance/kernels.h"
#include <cstring>
#include <limits>
#include <stdexcept>

namespace gem::detail {
Graph Graph::build(const EncodedCorpus& corpus, const BuildOptions& options) {
    if (options.m < 2 || options.m > 10000 || options.ef_construction < options.m)
        throw std::invalid_argument("invalid HNSW build options");
    auto impl = std::make_unique<Impl>(corpus);
    const auto centers = corpus.fine_centroids.size() / corpus.documents.dimension;
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (centers > limit / centers || centers * centers > limit / sizeof(float))
        throw std::overflow_error("workspace size overflow");
    if (centers * centers * sizeof(float) > options.distance_budget_bytes)
        throw std::invalid_argument("dense centroid matrix exceeds build.distance_budget_bytes");
    std::vector<float> distances(centers * centers);
    dot_table(corpus.fine_centroids.data(), centers, corpus.fine_centroids.data(), centers,
              corpus.documents.dimension, distances.data());
    impl->graph = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        impl->space.get(), impl->documents.size() + 1, options.m, options.ef_construction, options.seed);
    impl->graph->fstdistfuncClusterEMD = [centers](const vectorset* a, const vectorset* b, const float* table) {
        vectorset left(nullptr, nullptr, 0, 0), right(nullptr, nullptr, 0, 0);
        std::memcpy(&left, a, sizeof(left));
        std::memcpy(&right, b, sizeof(right));
        return qemd_distance(left.codes, left.vecnum, right.codes, right.vecnum, table, centers);
    };
    for (const auto& cluster : corpus.clusters) {
        if (cluster.empty()) continue;
        impl->graph->entry_map.clear();
        // The entry must be installed before workers begin: each later insert
        // routes from it.  Keep clusters serial because entry_map is reused
        // for each cluster, but parallelize independent document inserts.
        const auto entry = cluster.front();
        impl->graph->addClusterPointEntry(&impl->documents[entry], distances.data(), entry, entry);
        if (options.build_threads == 1 || cluster.size() == 1) {
            for (std::size_t i = 1; i < cluster.size(); ++i) {
                const auto id = cluster[i];
                impl->graph->addClusterPointEntry(&impl->documents[id], distances.data(), id, entry);
            }
            continue;
        }
        if (options.build_threads == 0) {
#pragma omp parallel for schedule(static)
            for (std::ptrdiff_t i = 1;
                 i < static_cast<std::ptrdiff_t>(cluster.size()); ++i) {
                const auto id = cluster[static_cast<std::size_t>(i)];
                impl->graph->addClusterPointEntry(&impl->documents[id], distances.data(), id, entry);
            }
            continue;
        }
#pragma omp parallel for schedule(static) num_threads(options.build_threads)
        for (std::ptrdiff_t i = 1;
             i < static_cast<std::ptrdiff_t>(cluster.size()); ++i) {
            const auto id = cluster[static_cast<std::size_t>(i)];
            impl->graph->addClusterPointEntry(&impl->documents[id], distances.data(), id, cluster.front());
        }
    }
    impl->prepare_search(corpus);
    return Graph(std::move(impl));
}
}  // namespace gem::detail
