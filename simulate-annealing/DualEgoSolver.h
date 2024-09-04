#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
using std::vector, std::string, std::pair;

class DualEgoSolver {
public:
	// The metadata for a model
	struct ModelMeta {
		// The number of microbatches
		int num_mbatches;
		// Forwarding time for one micro batch
		int fwd_time;
		// Backward propagation time for one micro bach
		int bwd_time;
		// The mapping from stages to node_id
		// Should only contain FWD stages
		vector<int> stage2node;
		// Colors used when print the trace
		string fwd_color_code, bwd_color_code;
	};

	// An item in the final trace
	struct TraceItem {
		int start_time;
		int duration;
		int selected_model;
		int selected_mbatch;
		int selected_stage;
	};

	// The trace, represents the final solution found by the solver
	struct Trace {
		int time_usage;
		vector<TraceItem> trace;
	};

	// Define how simulated annealing is initialized
	// TODO May add other initialization methods (like 1F1B) in the future
	enum class sim_anneal_init_t {
		FFFFBBBB
	};

	// Define how simulated annealing is disturbed
	// TODO May add other disturbing methods in the future
	enum class sim_anneal_disturb_t {
		RANDOM_SWAP
	};

	// Configuration for simulated annealing
	struct SimAnnealConfig {
		// Initial temperature
		double init_temp;
		// Cooling rate
		double cooling_rate;
		// Stop temperature
		double stop_temp;
		// The random seed
		int seed;
		// The method for initialization
		sim_anneal_init_t init_method;
		// The method for disturbing
		sim_anneal_disturb_t disturb_method;
	};

private:
	int num_nodes;
	int num_models;
	vector<ModelMeta> model_metas;	// [num_models]
	SimAnnealConfig sim_anneal_config;

	int max_num_mbatches;
	int max_num_stages;
	int tot_num_mbatches_times_stages;

	// The definition of a "task" during simulated annealing
	struct Task {
		int model_id;
		bool is_fwd;
	};

	// The overall schedule (one acceptable but not necessarily optimal solution)
	struct TaskSched {
		vector<vector<Task>> tasks;	// [NUM_NODES, sum(m | all models){m.num_mbatches*2}]
	};

	// Map the stage of one model to a node
	inline int stage_to_node(int model_id, int stage_id) {
		const ModelMeta& meta = model_metas[model_id];
		int num_stages = (int)meta.stage2node.size();
		return stage_id < num_stages ? meta.stage2node[stage_id] : meta.stage2node[2*num_stages-1-stage_id];
	}

	// Return the time usage of one schedule.
	// Return (e2e_time_usage, avg_time_usage*num_tasks)
	// If the schedule is invalid, return -1
	// Save the trace if SAVE_TRACE is true
	template<bool SAVE_TRACE>
	pair<int, int> get_time_usage(TaskSched const& task_sched, vector<TraceItem> &trace) {
		// The ending time for the last stage of a microbatch of a model
		int last_stage_end_time[num_models][max_num_mbatches];
		// The next idle time for a node
		int node_idle_time[num_nodes];
		// The index of the next task within `task_sched[node_id]`
		int next_task_index[num_nodes];
		// The number of forwarded micro batches and backwarded micro batches
		// for every model on every node
		int num_fwded_mbatches[num_nodes][num_models], num_bwded_mbatches[num_nodes][num_models];
		// The index of the next stage for a micro batch of one model
		int mbatch_next_stage[num_models][max_num_mbatches];
		memset(node_idle_time, 0, sizeof(node_idle_time));
		memset(next_task_index, 0, sizeof(next_task_index));
		memset(num_fwded_mbatches, 0, sizeof(num_fwded_mbatches));
		memset(num_bwded_mbatches, 0, sizeof(num_bwded_mbatches));
		memset(mbatch_next_stage, 0, sizeof(mbatch_next_stage));
		int summed_time_usage = 0;	// The sum of time usage of all tasks. = avg_time_usage * num_tasks

		int node_id = num_nodes-1;
		for (int i = 0; i < 2*tot_num_mbatches_times_stages; ++i) {
			// Find a task to schedule
			bool task_found = false;
			// Find a node, and try to schedule its next task
			// Here we start from the node we chosen last time to optimize performance
			for (int _ = 0; _ < num_nodes; ++_) {
				node_id = (node_id+1) % num_nodes;
				if (next_task_index[node_id] == (int)task_sched.tasks[node_id].size()) {
					// No task, go fishing!
					continue;
				}
				auto [model_id, is_fwd] = task_sched.tasks[node_id][next_task_index[node_id]];
				int mbatch_id = is_fwd ? num_fwded_mbatches[node_id][model_id] : num_bwded_mbatches[node_id][model_id];
				int stage_id = mbatch_next_stage[model_id][mbatch_id];
				int target_node_id = stage_to_node(model_id, stage_id);
				if (target_node_id != node_id) {
					continue;
				}
				if (!is_fwd && num_bwded_mbatches[node_id][model_id] == num_fwded_mbatches[node_id][model_id]) {
					return {-1, -1};
				}
				
				const ModelMeta& model_meta = model_metas[model_id];
				int task_duration = is_fwd ? model_meta.fwd_time : model_meta.bwd_time;

				int mbatch_prev_task_end_time = stage_id == 0 ? 0 : last_stage_end_time[model_id][mbatch_id];
				int cur_node_next_idle_time = node_idle_time[node_id];
				int cur_task_start_time = std::max(mbatch_prev_task_end_time, cur_node_next_idle_time);
				int cur_task_fin_time = cur_task_start_time + task_duration;

				// Update the state
				summed_time_usage += cur_task_fin_time;
				last_stage_end_time[model_id][mbatch_id] = cur_task_fin_time;
				node_idle_time[node_id] = cur_task_fin_time;
				next_task_index[node_id]++;
				if (is_fwd) {
					num_fwded_mbatches[node_id][model_id]++;
				} else {
					num_bwded_mbatches[node_id][model_id]++;
				}
				mbatch_next_stage[model_id][mbatch_id]++;

				if constexpr(SAVE_TRACE) {
					trace.push_back({cur_task_start_time, task_duration, model_id, mbatch_id, stage_id});
				}
				task_found = true;
				break;
			}
			// assert(task_found);
			if (!task_found) {
				return {-1, -1};
			}
		}

		int max_time_usage = 0;
		for (int node_id = 0; node_id < num_nodes; ++node_id) {
			max_time_usage = std::max(max_time_usage, node_idle_time[node_id]);
		}
		return {max_time_usage, summed_time_usage};
	}

	// A shorthand for get_time_usage with SAVE_TRACE is false
	pair<int, int> get_time_usage(TaskSched const& task_sched) {
		static vector<TraceItem> temp_trace;
		return get_time_usage<false>(task_sched, temp_trace);
	}

	// Get the initial task sched
	TaskSched get_init_task_sched() {
		TaskSched res;
		res.tasks.resize(num_nodes);
		// for (int node_id = 0; node_id < num_nodes; ++node_id)
		// 	res.tasks[node_id].reserve(2*tot_num_mbatches);
		if (sim_anneal_config.init_method == sim_anneal_init_t::FFFFBBBB) {
			for (int model_id = 0; model_id < num_models; ++model_id) {
				const ModelMeta& meta = model_metas[model_id];
				for (int stage_id = 0; stage_id < (int)meta.stage2node.size(); ++stage_id)
					for (int _ = 0; _ < meta.num_mbatches; ++_)
						res.tasks[stage_to_node(model_id, stage_id)].push_back({model_id, true});
			}
			for (int model_id = 0; model_id < num_models; ++model_id) {
				const ModelMeta& meta = model_metas[model_id];
				int num_stages = (int)meta.stage2node.size();
				for (int stage_id = num_stages; stage_id < 2*num_stages; ++stage_id)
					for (int _ = 0; _ < meta.num_mbatches; ++_)
						res.tasks[stage_to_node(model_id, stage_id)].push_back({model_id, false});
			}
		} else {
			assert(0);
		}
		return res;
	}

	// Disturb
	void disturb(std::mt19937 &rng, TaskSched &new_task_sched) {
		if (sim_anneal_config.disturb_method == sim_anneal_disturb_t::RANDOM_SWAP) {
			int node_id = rng() % num_nodes;
			int taskA_id = rng() % new_task_sched.tasks[node_id].size();
			int taskB_id = rng() % new_task_sched.tasks[node_id].size();
			std::swap(new_task_sched.tasks[node_id][taskA_id], new_task_sched.tasks[node_id][taskB_id]);
		}
	}

	// Run simulated annealing, and return a task_sched
	TaskSched simulated_annealing() {
		TaskSched cur_task_sched = get_init_task_sched();
		TaskSched best_task_sched = cur_task_sched;
		pair<int, int> best_time_usage = get_time_usage(cur_task_sched);
		pair<int, int> cur_time_usage = best_time_usage;
		assert(best_time_usage.first != -1);
		
		double temperature = sim_anneal_config.init_temp;
		double cooling_rate = sim_anneal_config.cooling_rate;
		std::mt19937 rng(sim_anneal_config.seed);
		while (temperature > sim_anneal_config.stop_temp) {
			TaskSched new_task_sched = cur_task_sched;
			disturb(rng, new_task_sched);
			pair<int, int> new_time_usage = get_time_usage(new_task_sched);
			bool is_accept = false;
			if (new_time_usage.first != -1) {
				if (new_time_usage < cur_time_usage) {
					is_accept = true;
				} else {
					double prob = std::exp((cur_time_usage.first - new_time_usage.first) / temperature);
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
					best_task_sched = cur_task_sched;
				}
			}
			temperature *= cooling_rate;
		}
		return best_task_sched;
	}

public:
	DualEgoSolver(int num_stages, vector<ModelMeta> const& model_metas, SimAnnealConfig const& sim_anneal_config):
		num_nodes(num_stages), num_models(model_metas.size()), model_metas(model_metas), sim_anneal_config(sim_anneal_config) {
		max_num_mbatches = 0;
		max_num_stages = 0;
		tot_num_mbatches_times_stages = 0;
		for (const ModelMeta& meta : model_metas) {
			int cur_num_stages = (int)meta.stage2node.size();
			max_num_mbatches = std::max(max_num_mbatches, meta.num_mbatches);
			max_num_stages = std::max(max_num_stages, cur_num_stages);
			tot_num_mbatches_times_stages += meta.num_mbatches * cur_num_stages;
		}
	}

	Trace solve() {
		TaskSched best_sched = simulated_annealing();
		vector<TraceItem> trace;
		int time_usage = get_time_usage<true>(best_sched, trace).first;
		return {
			time_usage,
			trace
		};
	}

	void print_trace(Trace const& trace) {
		using std::string;
		auto colorize = [&](const string &text, int model_id, bool is_bwd) -> string {
			static string clear_color = "\033[0m";
			string color_code = is_bwd ? model_metas[model_id].bwd_color_code : model_metas[model_id].fwd_color_code;
			return color_code + text + clear_color;
		};

		struct Job {
			bool is_occupied = false;
			int model_id;
			int mbatch_id;
			bool is_bwd;
		};
		std::vector<Job> jobs[num_nodes];
		for (auto &jobs_on_node : jobs) {
			jobs_on_node.resize(trace.time_usage);
		}
		for (const TraceItem& trace_item : trace.trace) {
			auto [start_time, duration, model_id, mbatch_id, selected_stage] = trace_item;
			int node_id = stage_to_node(model_id, selected_stage);
			// printf("Start time: %d, Duration: %d, Model: %d, Micro batch: %d, Node: %d\n", start_time, duration, model_id, mbatch_id, node_id);
			bool is_bwd = selected_stage >= (int)model_metas[model_id].stage2node.size();
			for (int t = start_time; t < start_time + duration; ++t) {
				jobs[node_id][t] = {true, model_id, mbatch_id, is_bwd};
			}
		}

		for (int i = 0; i < num_nodes; ++i) {
			for (int t = 0; t < trace.time_usage; ++t) {
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
};
