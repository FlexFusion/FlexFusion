#include <array>
#include <bits/stdc++.h>
#include <random>
#include <vector>
using std::vector;


// We define these numbers as a constexpr since we are going to allocate arrays with size N
// You may use metaprogramming if you want to make it more flexible
constexpr int NUM_STAGES = 8; // The number of pipeline stages
constexpr int NUM_NODES = NUM_STAGES;
constexpr int NUM_MODELS = 2; // The number of models
constexpr int NUM_MBATCHS = 8; // The number of micro batches in each model. "mbatch" is micro batch

// The forward and backward time of a micro batch in each model
// We use constexpr to make it more efficient. This can be changed to a normal variable
constexpr int MODEL_FWD_TIMES[NUM_MODELS] = {2, 1};
constexpr int MODEL_BWD_TIMES[NUM_MODELS] = {4, 2};

// A mapping from a stage in a model to the index of a node
int STAGE_TO_NODE[NUM_MODELS][NUM_STAGES*2];
void generate_stage_to_node() {
	assert(NUM_MODELS == 1 || NUM_MODELS == 2);
	for (int j = 0; j < NUM_STAGES*2; ++j) {
		STAGE_TO_NODE[0][j] = (j < NUM_STAGES) ? j : NUM_STAGES*2-j-1;
	}
	if (NUM_MODELS == 2) {
		for (int j = 0; j < NUM_STAGES*2; ++j) {
			STAGE_TO_NODE[1][j] = (j < NUM_STAGES) ? NUM_STAGES-j-1 : j-NUM_STAGES;
		}
	}
}


// constexpr int NUM_STAGES = 4;
// constexpr int NUM_NODES = NUM_STAGES;
// constexpr int NUM_MODELS = 1;
// constexpr int NUM_MBATCHS = 4;

// constexpr int MODEL_FWD_TIMES[NUM_MODELS] = {1};
// constexpr int MODEL_BWD_TIMES[NUM_MODELS] = {2};

// // A mapping from a stage in a model to the index of a node
// constexpr int STAGE_TO_NODE[NUM_MODELS][NUM_STAGES*2] = {
//     {0, 1, 2, 3, 3, 2, 1, 0},
// };

struct PathItem {
    int start_time;
    int duration;
    int selected_model;
    int selected_mbatch;
    int selected_stage;
};

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

struct Task {
	int model_id;
	bool is_fwd;
};

struct TaskSched {
	vector<Task> tasks[NUM_NODES];
};

TaskSched init_tasks() {
	TaskSched res;
	for (int model_id = 0; model_id < NUM_MODELS; ++model_id) {
		for (int stage_id = 0; stage_id < NUM_STAGES; ++stage_id)
			for (int mbatch_id = 0; mbatch_id < NUM_MBATCHS; ++mbatch_id)
				res.tasks[STAGE_TO_NODE[model_id][stage_id]].push_back({model_id, true});
		for (int stage_id = NUM_STAGES; stage_id < 2*NUM_STAGES; ++stage_id)
			for (int mbatch_id = 0; mbatch_id < NUM_MBATCHS; ++mbatch_id)
				res.tasks[STAGE_TO_NODE[model_id][stage_id]].push_back({model_id, false});
	}
	// #define T true
	// #define F false
	// res.tasks[0] = {{0, T}, {0, T}, {0, T}, {0, T}, {1, T}, {1, F}, {1, T}, {1, T}, {1, T}, {1, F}, {1, F}, {0, F}, {1, F}, {0, F}, {0, F}, {0, F}};
	// res.tasks[1] = {{0, T}, {0, T}, {0, T}, {0, T}, {1, T}, {1, T}, {1, T}, {1, T}, {1, F}, {0, F}, {1, F}, {1, F}, {0, F}, {1, F}, {0, F}, {0, F}};
	// res.tasks[2] = {{1, T}, {1, T}, {1, T}, {0, T}, {0, T}, {0, T}, {1, T}, {0, F}, {0, T}, {1, F}, {0, F}, {1, F}, {0, F}, {0, F}, {1, F}, {1, F}};
	// res.tasks[3] = {{1, T}, {1, T}, {1, T}, {1, T}, {0, T}, {0, F}, {0, T}, {0, T}, {0, F}, {0, T}, {0, F}, {0, F}, {1, F}, {1, F}, {1, F}, {1, F}};
	return res;
}

template<bool SAVE_PATH>
int get_time_usage(TaskSched const& tash_sched, vector<PathItem> &path) {
	static int stage_end_time[NUM_MODELS][NUM_MBATCHS][2*NUM_STAGES];
	static int node_idle_time[NUM_NODES];
	static int next_task_index[NUM_NODES];
	static int num_fwded_mbatches[NUM_NODES][NUM_MODELS], num_bwded_mbatches[NUM_NODES][NUM_MODELS];
	static int mbatch_next_stage[NUM_MODELS][NUM_MBATCHS];
	memset(stage_end_time, -1, sizeof(stage_end_time));
	memset(node_idle_time, 0, sizeof(node_idle_time));
	memset(next_task_index, 0, sizeof(next_task_index));
	memset(num_fwded_mbatches, 0, sizeof(num_fwded_mbatches));
	memset(num_bwded_mbatches, 0, sizeof(num_bwded_mbatches));
	memset(mbatch_next_stage, 0, sizeof(mbatch_next_stage));

	for (int i = 0; i < NUM_MODELS*(2*NUM_STAGES)*NUM_MBATCHS; ++i) {
		// Find a task to schedule
		bool task_found = false;
		for (int node_id = 0; node_id < NUM_NODES; ++node_id) {
			if (next_task_index[node_id] == tash_sched.tasks[node_id].size()) {
				continue;
			}
			auto [model_id, is_fwd] = tash_sched.tasks[node_id][next_task_index[node_id]];
			int mbatch_id = is_fwd ? num_fwded_mbatches[node_id][model_id] : num_bwded_mbatches[node_id][model_id];
			int stage_id = mbatch_next_stage[model_id][mbatch_id];
			int task_duration = is_fwd ? MODEL_FWD_TIMES[model_id] : MODEL_BWD_TIMES[model_id];
			if (STAGE_TO_NODE[model_id][stage_id] != node_id) {
				continue;
			}
			if (!is_fwd && num_bwded_mbatches[node_id][model_id] == num_fwded_mbatches[node_id][model_id]) {
				return -1;
			}

			int mbatch_prev_task_end_time = stage_id == 0 ? 0 : stage_end_time[model_id][mbatch_id][stage_id-1];
			if (mbatch_prev_task_end_time == -1) {
				continue;
			}
			int cur_node_next_idle_time = node_idle_time[node_id];
			int cur_task_start_time = std::max(mbatch_prev_task_end_time, cur_node_next_idle_time);
			int cur_task_fin_time = cur_task_start_time + task_duration;

			// Update the state
			stage_end_time[model_id][mbatch_id][stage_id] = cur_task_fin_time;
			node_idle_time[node_id] = cur_task_fin_time;
			next_task_index[node_id]++;
			if (is_fwd) {
				num_fwded_mbatches[node_id][model_id]++;
			} else {
				num_bwded_mbatches[node_id][model_id]++;
			}
			mbatch_next_stage[model_id][mbatch_id]++;

			if constexpr(SAVE_PATH) {
				path.push_back({cur_task_start_time, task_duration, model_id, mbatch_id, stage_id});
			}
			task_found = true;
			break;
		}
		// assert(task_found);
		if (!task_found) {
			return -1;
		}
	}

	int max_time_usage = 0;
	for (int node_id = 0; node_id < NUM_NODES; ++node_id) {
		max_time_usage = std::max(max_time_usage, node_idle_time[node_id]);
	}
	return max_time_usage;
}

int get_time_usage(TaskSched const& task_sched) {
	vector<PathItem> temp_path;
	return get_time_usage<false>(task_sched, temp_path);
}

TaskSched simulated_annealing(int SEED) {
	TaskSched cur_task_sched = init_tasks();
	int best_time_usage = get_time_usage(cur_task_sched);
	int cur_time_usage = best_time_usage;
	
	double temperature = 1e7;
	static constexpr double COOLING_RATE = 0.99999;
	std::mt19937 rng(SEED);
	while (temperature > 1e-9) {
		TaskSched new_task_sched = cur_task_sched;
		int node_id = rng() % NUM_NODES;
		int taskA_id = rng() % new_task_sched.tasks[node_id].size();
		int taskB_id = rng() % new_task_sched.tasks[node_id].size();
		std::swap(new_task_sched.tasks[node_id][taskA_id], new_task_sched.tasks[node_id][taskB_id]);
		int new_time_usage = get_time_usage(new_task_sched);
		bool is_accept = false;
		if (new_time_usage != -1) {
			if (new_time_usage < cur_time_usage) {
				is_accept = true;
			} else {
				double prob = std::exp((cur_time_usage - new_time_usage) / temperature);
				double rnd = std::generate_canonical<double, 10>(rng);
				if (rnd < prob) {
					is_accept = true;
				}
			}
		}
		if (is_accept) {
			cur_task_sched = new_task_sched;
			cur_time_usage = new_time_usage;
			if (cur_time_usage < best_time_usage) {
				best_time_usage = cur_time_usage;
			}
		} else {
			std::swap(new_task_sched.tasks[node_id][taskA_id], new_task_sched.tasks[node_id][taskB_id]);
		}
		temperature *= COOLING_RATE;
	}
	return cur_task_sched;
}

int main() {
	generate_stage_to_node();
	for (int seed : vector<int>{0, 114514, 19981207, 20031208, 20000317}) {
		TaskSched task_sched = simulated_annealing(seed);
		vector<PathItem> path;
		int result = get_time_usage<true>(task_sched, path);
		printf("Minimum time usage: %d\n", result);
		print_path(result, path);
	}
	return 0;
}