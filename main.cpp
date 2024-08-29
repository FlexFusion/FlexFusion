#include <bits/stdc++.h>
#include <climits>

// We define these numbers as a constexpr since we are going to allocate arrays with size N
// You may use metaprogramming if you want to make it more flexible
constexpr int NUM_STAGES = 4; // The number of pipeline stages
constexpr int NUM_NODES = NUM_STAGES;
constexpr int NUM_MODELS = 2; // The number of models
constexpr int NUM_MBATCHS = 5; // The number of micro batches in each model. "mbatch" is micro batch

// The forward and backward time of a micro batch in each model
// We use constexpr to make it more efficient. This can be changed to a normal variable
constexpr int MODEL_FWD_TIMES[NUM_MODELS] = {2, 1};
constexpr int MODEL_BWD_TIMES[NUM_MODELS] = {4, 2};

// A mapping from a stage in a model to the index of a node
constexpr int STAGE_TO_NODE[NUM_MODELS][NUM_STAGES*2] = {
    {0, 1, 2, 3, 3, 2, 1, 0},
    {3, 2, 1, 0, 0, 1, 2, 3}
};

// constexpr int NUM_STAGES = 5;
// constexpr int NUM_NODES = NUM_STAGES;
// constexpr int NUM_MODELS = 1;
// constexpr int NUM_MBATCHS = 5;

// constexpr int MODEL_FWD_TIMES[NUM_MODELS] = {1};
// constexpr int MODEL_BWD_TIMES[NUM_MODELS] = {2};

// // A mapping from a stage in a model to the index of a node
// constexpr int STAGE_TO_NODE[NUM_MODELS][NUM_STAGES*2] = {
//     {0, 1, 2, 3, 4, 4, 3, 2, 1, 0},
// };

using hash_t = std::size_t;

struct State {
    int node_cur_model_id[NUM_NODES];
    int node_cur_mbatch_id[NUM_NODES];
    int node_task_time_remaining[NUM_NODES];
    int mbatch_nxt_stage[NUM_MODELS][NUM_MBATCHS]; // The next stage of a micro batch in each model

    inline hash_t get_hash() const {
        hash_t hash = 0;
        for (int i = 0; i < NUM_NODES; i++) {
            hash = hash * 133 + node_task_time_remaining[i];    // TODO Fix this hash
            hash = hash * NUM_MODELS + node_cur_model_id[i];
            hash = hash * NUM_MBATCHS + node_cur_mbatch_id[i];
        }
        for (int i = 0; i < NUM_MODELS; i++) {
            for (int j = 0; j < NUM_MBATCHS; j++) {
                hash = hash * (2*NUM_STAGES+1) + mbatch_nxt_stage[i][j];
            }
        }
        return hash;
    }
};

// An item in the memory, including the best answer as well as the selected model and micro batch for the next
struct MemoryItem {
    int smallest_time_usage;
    int selected_model; // -1 means that we wait for another node to finish its task
    int selected_mbatch_id;
};

std::unordered_map<hash_t, MemoryItem> memory;

int dfs(const State &state) {
    // Check whether all micro batches are finished
    bool all_finished = true;
    for (int i = 0; i < NUM_MODELS; i++) {
        for (int j = 0; j < NUM_MBATCHS; j++) {
            if (state.mbatch_nxt_stage[i][j] != 2*NUM_STAGES) {
                all_finished = false;
                break;
            }
        }
    }
    if (all_finished) {
        // Do not need to issue more micro batches
        int max_time_remaining = 0;
        for (int i = 0; i < NUM_NODES; i++) {
            max_time_remaining = std::max(max_time_remaining, state.node_task_time_remaining[i]);
        }
        return max_time_remaining;
    }
    // Try to retrieve the result from memory
    hash_t hash = state.get_hash();
    if (memory.find(hash) != memory.end()) {
        return memory[hash].smallest_time_usage;
    }
    int best_result = INT_MAX;
    int best_model_id = -1;
    int best_mbatch_id = -1;
    int min_schedulable_task_duration = INT_MAX;
    // Try to schedule a micro batch on a node
    // Enumerate the micro batch to be scheduled
    for (int model_id = 0; model_id < NUM_MODELS; ++model_id) {
        // We do this in reverse order to optimize performance: if mbatch #i is already finished,
        // everything before it will also be finished, so we can skip them
        for (int mbatch_id = NUM_MBATCHS-1; mbatch_id >= 0; --mbatch_id) {
            int next_stage_id = state.mbatch_nxt_stage[model_id][mbatch_id];
            if (next_stage_id == 2*NUM_STAGES) {
                // This micro batch is finished, continue
                break;
            }
            if (mbatch_id != 0 && next_stage_id == state.mbatch_nxt_stage[model_id][mbatch_id-1]) {
                // Progress will surpass the previous micro batch, infesible
                continue;
            }
            int target_node = STAGE_TO_NODE[model_id][next_stage_id];
            if (state.node_task_time_remaining[target_node] != 0) {
                // The node is occupied, infesible
                continue;
            }
            if (next_stage_id != 0) {
                int prev_node = STAGE_TO_NODE[model_id][next_stage_id-1];
                if (state.node_task_time_remaining[prev_node] != 0 && state.node_cur_model_id[prev_node] == model_id && state.node_cur_mbatch_id[prev_node] == mbatch_id) {
                    // The previous stage is not finished, infesible
                    continue;
                }
            }
            // Construct the new state
            State new_state = state;
            int cur_task_duration = (next_stage_id < NUM_STAGES) ? MODEL_FWD_TIMES[model_id] : MODEL_BWD_TIMES[model_id];
            min_schedulable_task_duration = std::min(min_schedulable_task_duration, cur_task_duration);
            new_state.node_task_time_remaining[target_node] = cur_task_duration;
            new_state.mbatch_nxt_stage[model_id][mbatch_id] = next_stage_id + 1;
            // Move the time forward
            int min_task_time_remaining = INT_MAX;
            for (int i = 0; i < NUM_NODES; i++) {
                min_task_time_remaining = std::min(min_task_time_remaining, new_state.node_task_time_remaining[i]);
            }
            for (int i = 0; i < NUM_NODES; i++) {
                new_state.node_task_time_remaining[i] -= min_task_time_remaining;
            }
            new_state.node_cur_model_id[target_node] = model_id;
            new_state.node_cur_mbatch_id[target_node] = mbatch_id;
            // Recur
            int cur_result = dfs(new_state);
            if (cur_result == INT_MAX) {
                // Infeasible
                continue;
            }
            cur_result += min_task_time_remaining;
            if (cur_result < best_result) {
                best_result = cur_result;
                best_model_id = model_id;
                best_mbatch_id = mbatch_id;
            }
        }
    }
    // Or, we can try to wait for a while (for another node to finish its task)
    {
        int min_task_time_remaining = INT_MAX;
        for (int i = 0; i < NUM_NODES; i++) {
            int cur_node_time_remaining = state.node_task_time_remaining[i];
            if (cur_node_time_remaining != 0) {
                min_task_time_remaining = std::min(min_task_time_remaining, cur_node_time_remaining);
            }
        }
        // The latter condition is a pruning optimization: If we can allocate something and that ends
        // before we just wait, we should allocate it
        if (min_task_time_remaining != INT_MAX && min_task_time_remaining < min_schedulable_task_duration) {
            State new_state = state;
            // Move the time forward
            for (int i = 0; i < NUM_NODES; i++) {
                if (new_state.node_task_time_remaining[i] != 0) {
                    new_state.node_task_time_remaining[i] -= min_task_time_remaining;
                }
            }
            // Recur
            int cur_result = dfs(new_state);
            if (cur_result != INT_MAX) {
                cur_result += min_task_time_remaining;
            }
            if (cur_result < best_result) {
                best_result = cur_result;
                best_model_id = -1;
                best_mbatch_id = -1;
            }
        }
    }
    // Save the result to memory
    memory[hash] = {best_result, best_model_id, best_mbatch_id};
    return best_result;
}

struct PathItem {
    int start_time;
    int duration;
    int selected_model;
    int selected_mbatch;
    int selected_stage;
};
void restore_memory(State state, int cur_time, std::vector<PathItem> &path) {
    hash_t hash = state.get_hash();
    if (memory.find(hash) == memory.end()) {
        return;
    }
    MemoryItem &item = memory[hash];

    int model_id = item.selected_model;
    int mbatch_id = item.selected_mbatch_id;

    if (model_id != -1) {
        int next_stage_id = state.mbatch_nxt_stage[model_id][mbatch_id];
        int target_node = STAGE_TO_NODE[model_id][next_stage_id];
        int duration = (next_stage_id < NUM_STAGES) ? MODEL_FWD_TIMES[model_id] : MODEL_BWD_TIMES[model_id];
        state.node_task_time_remaining[target_node] = duration;
        state.mbatch_nxt_stage[model_id][mbatch_id] = next_stage_id + 1;
        int min_task_time_remaining = INT_MAX;
        for (int i = 0; i < NUM_NODES; i++) {
            min_task_time_remaining = std::min(min_task_time_remaining, state.node_task_time_remaining[i]);
        }
        for (int i = 0; i < NUM_NODES; i++) {
            state.node_task_time_remaining[i] -= min_task_time_remaining;
        }
        state.node_cur_model_id[target_node] = model_id;
        state.node_cur_mbatch_id[target_node] = mbatch_id;
        path.push_back({cur_time, duration, model_id, mbatch_id, next_stage_id});
        cur_time += min_task_time_remaining;
    } else {
        int min_task_time_remaining = INT_MAX;
        for (int i = 0; i < NUM_NODES; i++) {
            int cur_node_time_remaining = state.node_task_time_remaining[i];
            if (cur_node_time_remaining != 0) {
                min_task_time_remaining = std::min(min_task_time_remaining, cur_node_time_remaining);
            }
        }
        assert(min_task_time_remaining != INT_MAX);
        for (int i = 0; i < NUM_NODES; i++) {
            if (state.node_task_time_remaining[i] != 0) {
                state.node_task_time_remaining[i] -= min_task_time_remaining;
            }
        }
        cur_time += min_task_time_remaining;
    }
    restore_memory(state, cur_time, path);
}

void print_path(int tot_time_usage, std::vector<PathItem> const& path) {
    using std::string;
    static string color_table[NUM_MODELS][2] = {
        {"\033[31m", "\033[32m"},
        {"\033[33m", "\033[34m"}
    };
    static string clear_color = "\033[0m";
    auto colorize = [&](const string &text, int model_id, bool is_bwd) -> string {
        return color_table[model_id][is_bwd] + text + clear_color;
    };

    struct Job {
        bool is_occupied = false;
        int model_id;
        int mbatch_id;
        bool is_bwd;
    };
    std::vector<Job> jobs[NUM_NODES];
    for (auto &jobs_on_node : jobs) {
        jobs_on_node.resize(tot_time_usage);
    }
    for (const PathItem& path_item : path) {
        auto [start_time, duration, model_id, mbatch_id, selected_stage] = path_item;
        int node_id = STAGE_TO_NODE[model_id][selected_stage];
        // printf("Start time: %d, Duration: %d, Model: %d, Micro batch: %d, Node: %d\n", start_time, duration, model_id, mbatch_id, node_id);
        bool is_bwd = selected_stage >= NUM_STAGES;
        for (int t = start_time; t < start_time + duration; ++t) {
            jobs[node_id][t] = {true, model_id, mbatch_id, is_bwd};
        }
    }

    for (int i = 0; i < NUM_NODES; ++i) {
        for (int t = 0; t < tot_time_usage; ++t) {
            if (jobs[i][t].is_occupied) {
                string text = "FB"[jobs[i][t].is_bwd] + std::to_string(jobs[i][t].mbatch_id);
                printf("%s", colorize(text, jobs[i][t].model_id, jobs[i][t].is_bwd).c_str());
            } else {
                printf("  ");
            }
        }
        printf("\n");
    }
}

int main() {
    State init_state;
    for (int i = 0; i < NUM_NODES; i++) {
        init_state.node_task_time_remaining[i] = 0;
    }
    for (int i = 0; i < NUM_MODELS; i++) {
        for (int j = 0; j < NUM_MBATCHS; j++) {
            init_state.mbatch_nxt_stage[i][j] = 0;
        }
    }
    int result = dfs(init_state);

    printf("Minimum time usage: %d\n", result);

    std::vector<PathItem> path;
    restore_memory(init_state, 0, path);

    print_path(result, path);

    return 0;
}