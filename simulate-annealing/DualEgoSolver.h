#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
using std::vector, std::string, std::pair;

template<typename T, typename F>
static inline std::vector<T> filter(const std::vector<T> &vec, F pred) {
	std::vector<T> res;
	for (const T &item : vec) {
		if (pred(item)) {
			res.push_back(item);
		}
	}
	return res;
}

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
		// The memory pressure of a mbatch that is forwarded but not backwarded on a node
		int mem_pressure;
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
		// The end2end time usage
		int time_usage;
		// The peak memory usage, defined as the maximum of 
		// \sum(model) model.mem_pressure * (num_fwded_mbatch on node i at time t - num_bwded_mbatch on node i at time t)
		// over all i and t
		int peak_memory_usage;
		int sum_peak_memory_usage;
		// The summation of time usage of every stage of every mbatch
		int fin_time_sum;
		// The trace
		vector<TraceItem> trace;
	};

	struct TraceMetric {
		int time_usage;
		int peak_memory_usage;
		float utilization;
		float bubble_rate;	// bubble_rate + utilization = 1.0
	};

	// Define how simulated annealing is initialized
	// TODO May add other initialization methods (like 1F1B) in the future
	enum class sim_anneal_init_t {
		FFFFBBBB_0,
		FFFFBBBB_1,
		FFFFBBBB_OPTIM_3_MODELS,
		ONE_F_ONE_B,
		GREEDY
	};

	// Define how simulated annealing is disturbed
	// TODO May add other disturbing methods in the future
	enum class sim_anneal_disturb_t {
		RANDOM_SWAP,
		RANDOM_ADJACENT_SWAP,
		RANDOM_MOVE,
	};

	// Configuration for simulated annealing (for either e2e time or memory usage)
	struct SimAnnealConfig {
		// Initial temperature
		double init_temp;
		// Cooling rate
		double cooling_rate;
		// Stop temperature
		double stop_temp;
		// The random seed
		int seed;
		// The method for disturbing
		sim_anneal_disturb_t disturb_method;
	};

	static inline bool is_trace_more_optimal(const Trace& trace_a, const Trace& trace_b) {
		return trace_a.time_usage != trace_b.time_usage ?
				trace_a.time_usage < trace_b.time_usage :
				(trace_a.peak_memory_usage != trace_b.peak_memory_usage ?
				trace_a.peak_memory_usage < trace_b.peak_memory_usage :
				(trace_a.sum_peak_memory_usage != trace_b.sum_peak_memory_usage ?
				trace_a.sum_peak_memory_usage < trace_b.sum_peak_memory_usage :
				trace_a.fin_time_sum < trace_b.fin_time_sum));
	}
	
	static inline string fmt_trace(const Trace& trace) {
		return "{" + std::to_string(trace.time_usage) + ", " + std::to_string(trace.peak_memory_usage) + ", " + std::to_string(trace.sum_peak_memory_usage) + ", " + std::to_string(trace.fin_time_sum) + "}";
	}

	static inline string fmt_sim_anneal_config(const SimAnnealConfig &config) {
		static char buf[1024];
		snprintf(buf, 1024, "{{%e, %f, %e, %d, %d}",
			config.init_temp, config.cooling_rate, config.stop_temp, config.seed, (int)config.disturb_method);
		return std::string(buf);
	}

	// The definition of a "task" during simulated annealing
	struct Task {
		int model_id;
		bool is_fwd;
	};

	friend inline bool operator==(const Task &a, const Task &b) {
		return a.model_id == b.model_id && a.is_fwd == b.is_fwd;
	}

	// The overall schedule (one acceptable but not necessarily optimal solution)
	struct TaskSched {
		vector<vector<Task>> tasks;	// [NUM_NODES, sum(m | all models){m.num_mbatches*2}]
	};
	
private:
	int num_nodes;
	int num_models;
	vector<ModelMeta> model_metas;	// [num_models]

	int max_num_mbatches;
	int max_num_stages;
	int tot_num_mbatches_times_stages;

	// Map the stage of one model to a node
	inline int stage_to_node(int model_id, int stage_id) const {
		const ModelMeta& meta = model_metas[model_id];
		int num_stages = (int)meta.stage2node.size();
		return stage_id < num_stages ? meta.stage2node[stage_id] : meta.stage2node[2*num_stages-1-stage_id];
	}

	// Return the time usage of one schedule.
	// Return (e2e_time_usage, avg_time_usage*num_tasks)
	// If the schedule is invalid, return -1
	// Save the trace if SAVE_TRACE is true
	template<bool SAVE_TRACE>
	Trace get_time_usage(TaskSched const& task_sched) const {
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
		// Memory pressure on a node
		int mem_pressure[num_nodes];
		int history_peak_mem_usage[num_nodes];
		memset(node_idle_time, 0, sizeof(node_idle_time));
		memset(next_task_index, 0, sizeof(next_task_index));
		memset(num_fwded_mbatches, 0, sizeof(num_fwded_mbatches));
		memset(num_bwded_mbatches, 0, sizeof(num_bwded_mbatches));
		memset(mbatch_next_stage, 0, sizeof(mbatch_next_stage));
		memset(mem_pressure, 0, sizeof(mem_pressure));
		memset(history_peak_mem_usage, 0, sizeof(history_peak_mem_usage));
		
		Trace result = {0, 0, 0, 0, {}};

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
					return {-1};
				}
				
				const ModelMeta& model_meta = model_metas[model_id];
				int task_duration = is_fwd ? model_meta.fwd_time : model_meta.bwd_time;

				int mbatch_prev_task_end_time = stage_id == 0 ? 0 : last_stage_end_time[model_id][mbatch_id];
				int cur_node_next_idle_time = node_idle_time[node_id];
				int cur_task_start_time = std::max(mbatch_prev_task_end_time, cur_node_next_idle_time);
				int cur_task_fin_time = cur_task_start_time + task_duration;

				// Update the state
				last_stage_end_time[model_id][mbatch_id] = cur_task_fin_time;
				node_idle_time[node_id] = cur_task_fin_time;
				next_task_index[node_id]++;
				if (is_fwd) {
					num_fwded_mbatches[node_id][model_id]++;
					mem_pressure[node_id] += model_meta.mem_pressure;
				} else {
					num_bwded_mbatches[node_id][model_id]++;
					mem_pressure[node_id] -= model_meta.mem_pressure;
				}
				mbatch_next_stage[model_id][mbatch_id]++;

				// Update statistics
				result.fin_time_sum += cur_task_fin_time;
				history_peak_mem_usage[node_id] = std::max(history_peak_mem_usage[node_id], mem_pressure[node_id]);

				if constexpr(SAVE_TRACE) {
					result.trace.push_back({cur_task_start_time, task_duration, model_id, mbatch_id, stage_id});
				}
				task_found = true;
				break;
			}
			// assert(task_found);
			if (!task_found) {
				return {-1};
			}
		}

		for (int node_id = 0; node_id < num_nodes; ++node_id) {
			result.time_usage = std::max(result.time_usage, node_idle_time[node_id]);
			result.peak_memory_usage = std::max(result.peak_memory_usage, history_peak_mem_usage[node_id]);
			result.sum_peak_memory_usage += history_peak_mem_usage[node_id];
		}
		return result;
	}

	// Get the initial task sched
	TaskSched sim_annealing_get_init_task_sched(sim_anneal_init_t init_method) {
		TaskSched res;
		res.tasks.resize(num_nodes);
		// Some helpers
		auto push_fwd_stages = [&](int model_id) {
			const ModelMeta& meta = model_metas[model_id];
			for (int stage_id = 0; stage_id < (int)meta.stage2node.size(); ++stage_id)
				for (int _ = 0; _ < meta.num_mbatches; ++_)
					res.tasks[stage_to_node(model_id, stage_id)].push_back({model_id, true});
		};
		auto push_bwd_stages = [&](int model_id) {
			const ModelMeta& meta = model_metas[model_id];
			int num_stages = (int)meta.stage2node.size();
			for (int stage_id = num_stages; stage_id < 2*num_stages; ++stage_id)
				for (int _ = 0; _ < meta.num_mbatches; ++_)
					res.tasks[stage_to_node(model_id, stage_id)].push_back({model_id, false});
		};
		if (init_method == sim_anneal_init_t::FFFFBBBB_0) {
			for (int model_id = 0; model_id < num_models; ++model_id) {
				push_fwd_stages(model_id);
			}
			for (int model_id = 0; model_id < num_models; ++model_id) {
				push_bwd_stages(model_id);
			}
		} else if (init_method == sim_anneal_init_t::FFFFBBBB_1) {
			for (int model_id = num_models-1; model_id >= 0; --model_id) {
				push_fwd_stages(model_id);
			}
			for (int model_id = 0; model_id < num_models; ++model_id) {
				push_bwd_stages(model_id);
			}
		} else if (init_method == sim_anneal_init_t::FFFFBBBB_OPTIM_3_MODELS) {
			// Manually optimized for scenarios where there are one big model
			// occupying all nodes, with 2 small models, each occupying a half
			// of nodes
			assert(num_models == 3);
			assert((int)model_metas[0].stage2node.size() == num_nodes);
			assert((int)model_metas[1].stage2node.size() == num_nodes/2);
			assert((int)model_metas[2].stage2node.size() == num_nodes/2);
			assert((int)model_metas[1].stage2node[0] == num_nodes/2-1);
			assert((int)model_metas[2].stage2node[0] == num_nodes-1);
			push_fwd_stages(2);
			push_fwd_stages(0);
			push_fwd_stages(1);
			push_bwd_stages(1);
			push_bwd_stages(0);
			push_bwd_stages(2);
		} else if (init_method == sim_anneal_init_t::ONE_F_ONE_B) {
			for (int model_id = 0; model_id < num_models; ++model_id) {
				const ModelMeta& meta = model_metas[model_id];
				int num_stages = (int)meta.stage2node.size();
				int num_mbatch = meta.num_mbatches;
				for (int stage_id = 0; stage_id < num_stages; ++stage_id) {
					int node_id = stage_to_node(model_id, stage_id);
					int num_prefix_fs = num_stages - stage_id - 1;
					for (int i = 0; i < num_prefix_fs; ++i)
						res.tasks[node_id].push_back({model_id, true});
					for (int i = 0; i < num_mbatch - num_prefix_fs; ++i) {
						res.tasks[node_id].push_back({model_id, true});
						res.tasks[node_id].push_back({model_id, false});
					}
					for (int i = 0; i < num_prefix_fs; ++i)
						res.tasks[node_id].push_back({model_id, false});
				}
			}
		} else if (init_method == sim_anneal_init_t::GREEDY) {
			return greedy();
		} else {
			assert(0);
		}
		return res;
	}

	// Disturb
	void disturb(sim_anneal_disturb_t disturb_method, std::mt19937 &rng, TaskSched &new_task_sched) {
		while (true) {
			if (disturb_method == sim_anneal_disturb_t::RANDOM_SWAP) {
				int node_id = rng() % num_nodes;
				int taskA_id = rng() % new_task_sched.tasks[node_id].size();
				int taskB_id = rng() % new_task_sched.tasks[node_id].size();
				if (new_task_sched.tasks[node_id][taskA_id] == new_task_sched.tasks[node_id][taskB_id]) {
					continue;
				} else {
					std::swap(new_task_sched.tasks[node_id][taskA_id], new_task_sched.tasks[node_id][taskB_id]);
					break;
				}
			} else if (disturb_method == sim_anneal_disturb_t::RANDOM_ADJACENT_SWAP) {
				int node_id = rng() % num_nodes;
				int taskA_id = rng() % ((int)new_task_sched.tasks[node_id].size()-1);
				if (new_task_sched.tasks[node_id][taskA_id] == new_task_sched.tasks[node_id][taskA_id+1]) {
					continue;
				} else {
					std::swap(new_task_sched.tasks[node_id][taskA_id], new_task_sched.tasks[node_id][taskA_id+1]);
					break;
				}
			} else if (disturb_method == sim_anneal_disturb_t::RANDOM_MOVE) {
				int node_id = rng() % num_nodes;
				int taskA_id = rng() % new_task_sched.tasks[node_id].size();
				int taskB_id = rng() % new_task_sched.tasks[node_id].size();
				Task task = new_task_sched.tasks[node_id][taskA_id];
				new_task_sched.tasks[node_id].erase(new_task_sched.tasks[node_id].begin() + taskA_id);
				new_task_sched.tasks[node_id].insert(new_task_sched.tasks[node_id].begin() + taskB_id, task);
				break;
			} else {
				assert(0);
			}
		}
	}

	// Run simulated annealing, and return a task_sched
	TaskSched simulated_annealing_e2e_time(const SimAnnealConfig &sim_anneal_config, TaskSched cur_task_sched) {
		TaskSched best_task_sched = cur_task_sched;
		Trace best_trace = get_time_usage<false>(cur_task_sched);
		Trace cur_trace = best_trace;
		assert(best_trace.time_usage != -1);
		
		double temperature = sim_anneal_config.init_temp;
		std::mt19937 rng(sim_anneal_config.seed);
		while (temperature > sim_anneal_config.stop_temp) {
			TaskSched new_task_sched = cur_task_sched;
			disturb(sim_anneal_config.disturb_method, rng, new_task_sched);
			Trace new_trace = get_time_usage<false>(new_task_sched);
			bool is_accept = false;
			if (new_trace.time_usage != -1) {
				if (is_trace_more_optimal(new_trace, cur_trace)) {
					is_accept = true;
				} else {
					int delta = cur_trace.time_usage - new_trace.time_usage;
					double prob = std::exp(delta / temperature);
					double rnd = std::generate_canonical<double, 10>(rng);
					if (rnd < prob) {
						is_accept = true;
					}
				}
			}
			if (is_accept) {
				// printf("%e %s\n", temperature, fmt_trace(cur_trace).c_str());
				cur_task_sched = new_task_sched;
				cur_trace = new_trace;
				if (is_trace_more_optimal(cur_trace, best_trace)) {
					best_task_sched = cur_task_sched;
					best_trace = cur_trace;
				}
			}
			temperature *= sim_anneal_config.cooling_rate;
		}
		return best_task_sched;
	}

	TaskSched simulated_annealing_peak_memory(const SimAnnealConfig &sim_anneal_config, TaskSched cur_task_sched) {
		TaskSched best_task_sched = cur_task_sched;
		Trace best_trace = get_time_usage<false>(cur_task_sched);
		Trace cur_trace = best_trace;
		assert(best_trace.time_usage != -1);

		double temperature = sim_anneal_config.init_temp;
		std::mt19937 rng(sim_anneal_config.seed);
		while (temperature > sim_anneal_config.stop_temp) {
			TaskSched new_task_sched = cur_task_sched;
			disturb(sim_anneal_config.disturb_method, rng, new_task_sched);
			Trace new_trace = get_time_usage<false>(new_task_sched);
			bool is_accept = false;
			if (new_trace.time_usage != -1 && new_trace.time_usage <= best_trace.time_usage) {
				if (is_trace_more_optimal(new_trace, cur_trace)) {
					is_accept = true;
				} else {
					int delta = cur_trace.sum_peak_memory_usage - new_trace.sum_peak_memory_usage;
					double prob = std::exp(delta / temperature);
					double rnd = std::generate_canonical<double, 10>(rng);
					if (rnd < prob) {
						is_accept = true;
					}
				}
			}
			if (is_accept) {
				// printf("%e %s\n", temperature, fmt_trace(cur_trace).c_str());
				cur_task_sched = new_task_sched;
				cur_trace = new_trace;
				if (is_trace_more_optimal(cur_trace, best_trace)) {
					best_task_sched = cur_task_sched;
					best_trace = cur_trace;
				}
			}
			temperature *= sim_anneal_config.cooling_rate;
		}
		return best_task_sched;
	}

	TraceMetric get_trace_metric(const Trace &trace) {
		int effective_work_time = 0;
		for (const ModelMeta &meta : model_metas) {
			effective_work_time += meta.num_mbatches * (int)meta.stage2node.size() * (meta.fwd_time+meta.bwd_time);
		}
		int total_work_time = num_nodes * trace.time_usage;
		float utilization = effective_work_time / (float)total_work_time;
		TraceMetric result {
			trace.time_usage,
			trace.peak_memory_usage,
			utilization,
			1 - utilization
		};
		return result;
	}

	// Get the "theoretically best" metric
	TraceMetric get_theoretical_best_metric() {
		// min_e2e_time is the theoretically minimum e2e time usage. It must be
		// - greater than the theoretically minimum e2e time for every model
		// - greater than ceil(all_work/num_nodes)
		int min_e2e_time = 0;
		for (const ModelMeta &meta : model_metas) {
			int cur_model_min_e2e_time = ((int)meta.stage2node.size() + meta.num_mbatches - 1) * (meta.fwd_time+meta.bwd_time);
			min_e2e_time = std::max(min_e2e_time, cur_model_min_e2e_time);
		}
		int effective_work_time = 0;
		for (const ModelMeta &meta : model_metas) {
			effective_work_time += meta.num_mbatches * (int)meta.stage2node.size() * (meta.fwd_time+meta.bwd_time);
		}
		min_e2e_time = std::max(min_e2e_time, (effective_work_time+num_nodes-1) / num_nodes);

		// min_peak_memory is the theoretically minimum peak memory usage. It must be
		// - greater than the theoretically minimum peak memory usage for every model (assume 1F1B)
		int min_peak_memory = 0;
		for(const ModelMeta &meta : model_metas) {
			min_peak_memory = std::max(min_peak_memory, (int)meta.stage2node.size() * meta.mem_pressure);
		}
			
		int total_work_time = num_nodes * min_e2e_time;
		float utilization = effective_work_time / (float)total_work_time;
		TraceMetric result {
			min_e2e_time,
			min_peak_memory,
			utilization,
			1 - utilization
		};
		return result;
	}

	void print_trace_metric(const TraceMetric &metric) {
		printf("E2E time usage: %d\n", metric.time_usage);
		printf("Peak memory usage: %d\n", metric.peak_memory_usage);
		printf("Utilization: %.2f%%\n", metric.utilization*100);
		printf("Bubble rate: %.2f%%\n", metric.bubble_rate*100);
	}

	TaskSched greedy() {
		TaskSched res;
		res.tasks.resize(num_nodes);

		int last_stage_end_time[num_models][max_num_mbatches];
		int node_idle_time[num_nodes];
		int num_fwded_mbatches[num_nodes][num_models], num_bwded_mbatches[num_nodes][num_models];
		int mbatch_next_stage[num_models][max_num_mbatches];
		memset(last_stage_end_time, 0, sizeof(last_stage_end_time));
		memset(node_idle_time, 0, sizeof(node_idle_time));
		memset(num_fwded_mbatches, 0, sizeof(num_fwded_mbatches));
		memset(num_bwded_mbatches, 0, sizeof(num_bwded_mbatches));
		memset(mbatch_next_stage, 0, sizeof(mbatch_next_stage));

		for (int i = 0; i < 2*tot_num_mbatches_times_stages; ++i) {
			struct Candidate {
				int node_id;
				int model_id;
				int mbatch_id;
				int stage_id;
				bool is_fwd;
				int start_time;
				int fin_time;
			};
			vector<Candidate> candidates;

			// Find all candidates (schedulable jobs)
			for (int node_id = 0; node_id < num_nodes; ++node_id) {
				for (int model_id = 0; model_id < num_models; ++model_id) {
					const ModelMeta &meta = model_metas[model_id];
					for (bool is_fwd : vector<bool>{true, false}) {
						int mbatch_id = is_fwd ? num_fwded_mbatches[node_id][model_id] : num_bwded_mbatches[node_id][model_id];
						if (mbatch_id >= meta.num_mbatches) {
							continue;
						}
						int stage_id = mbatch_next_stage[model_id][mbatch_id];
						if (stage_id == 2*(int)meta.stage2node.size()) {
							continue;
						}
						if (stage_to_node(model_id, stage_id) != node_id) {
							continue;
						}
						if (!is_fwd && num_bwded_mbatches[node_id][model_id] == num_fwded_mbatches[node_id][model_id]) {
							continue;
						}

						const ModelMeta &meta = model_metas[model_id];
						int task_duration = is_fwd ? meta.fwd_time : meta.bwd_time;

						int mbatch_prev_task_end_time = stage_id == 0 ? 0 : last_stage_end_time[model_id][mbatch_id];
						int cur_node_next_idle_time = node_idle_time[node_id];
						int cur_task_start_time = std::max(mbatch_prev_task_end_time, cur_node_next_idle_time);
						int cur_task_fin_time = cur_task_start_time + task_duration;

						candidates.push_back({node_id, model_id, mbatch_id, stage_id, is_fwd, cur_task_start_time, cur_task_fin_time});
					}
				}
			}
			
			// Select the best candidate according to our greedy strategy
			// Filter out candidates that overlaps with large models
			assert(!candidates.empty());
			int large_model_earlist_start_time[num_nodes];
			for (int i = 0; i < num_nodes; ++i) {
				large_model_earlist_start_time[i] = 1e9;
			}
			for (const Candidate &cand : candidates) {
				if (cand.model_id == 0) {
					large_model_earlist_start_time[cand.node_id] = std::min(large_model_earlist_start_time[cand.node_id], cand.start_time);
				}
			}
			candidates = filter(candidates, [&](const Candidate &cand) {
				if (cand.model_id == 0) {
					return true;
				} else {
					return cand.fin_time <= large_model_earlist_start_time[cand.node_id];
				}
			});
			// Select the task with the smallest start time
			assert(!candidates.empty());
			std::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
				return a.start_time != b.start_time ? 
						a.start_time < b.start_time :
						a.is_fwd < b.is_fwd;
			});
			const Candidate &selected_cand = candidates[0];

			// Update statistics
			res.tasks[selected_cand.node_id].push_back({selected_cand.model_id, selected_cand.is_fwd});
			last_stage_end_time[selected_cand.model_id][selected_cand.mbatch_id] = selected_cand.fin_time;
			node_idle_time[selected_cand.node_id] = selected_cand.fin_time;
			if (selected_cand.is_fwd) {
				num_fwded_mbatches[selected_cand.node_id][selected_cand.model_id]++;
			} else {
				num_bwded_mbatches[selected_cand.node_id][selected_cand.model_id]++;
			}
			mbatch_next_stage[selected_cand.model_id][selected_cand.mbatch_id]++;
		}

		return res;
	}

public:
	DualEgoSolver(int num_stages, vector<ModelMeta> const& model_metas):
		num_nodes(num_stages), num_models(model_metas.size()), model_metas(model_metas) {
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

	TaskSched get_init_task_sched(sim_anneal_init_t init_method) {
		return sim_annealing_get_init_task_sched(init_method);
	}

	TaskSched optimize_e2e_time(const SimAnnealConfig &config, TaskSched const& task_sched) {
		return simulated_annealing_e2e_time(config, task_sched);
	}

	TaskSched optimize_peak_memory(const SimAnnealConfig &config, TaskSched const& task_sched) {
		return simulated_annealing_peak_memory(config, task_sched);
	}

	Trace task_sched2trace(TaskSched const& task_sched) const {
		return get_time_usage<true>(task_sched);
	}

	Trace solve_greedy() {
		TaskSched sched = greedy();
		Trace result = get_time_usage<true>(sched);
		return result;
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

		printf("Theoretically best:\n");
		TraceMetric metric_theoretical = get_theoretical_best_metric();
		print_trace_metric(metric_theoretical);
		printf("\n");

		printf("Greedy:\n");
		Trace sched_greedy = solve_greedy();
		TraceMetric metric_greedy = get_trace_metric(sched_greedy);
		print_trace_metric(metric_greedy);
		printf("\n");

		printf("Ours:\n");
		TraceMetric metric_ours = get_trace_metric(trace);
		print_trace_metric(metric_ours);
	}
};
