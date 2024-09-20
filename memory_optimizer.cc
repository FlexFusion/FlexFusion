#include <string>
#include <istream>
#include <fstream>
#include <cassert>
using std::string;

#include "DualEgoSolver.h"
using ds = DualEgoSolver;

#include "omp.h"

vector<ds::ModelMeta> get_model_metas_helper(int pp0, int pp1, int num_mbatch0, int num_mbatch1, int fwd_time0, int fwd_time1, float mem0, float mem1) {
	auto get_range = [](int start, int end, bool is_reverse) {
		vector<int> res;
		for (int i = start; i < end; i++) {
			res.push_back(i);
		}
		if (is_reverse) {
			reverse(res.begin(), res.end());
		}
		return res;
	};
	vector<ds::ModelMeta> res;
	res.push_back({
		num_mbatch0,
		fwd_time0, 2*fwd_time0,
		get_range(0, pp0, false),
		mem0,
		"\033[31m", "\033[32m"
	});
	assert(pp0%pp1 == 0);
	int num_small_models = pp0/pp1;
	for (int i = 0; i < num_small_models; i++) {
		res.push_back({
			num_mbatch1,
			fwd_time1, 2*fwd_time1,
			get_range(i*pp1, (i+1)*pp1, true),
			mem1,
			i == 0 ? "\033[33m" : "\033[35m",
			i == 0 ? "\033[34m" : "\033[36m"
		});
	}
	return res;
}

auto get_sim_anneal_memory_configs_helper(const ds::SimAnnealConfig &base_config, int num_seeds) {
	vector<ds::SimAnnealConfig> res;
	for (int seed = 0; seed < num_seeds; ++seed) {
		ds::SimAnnealConfig cur_config = base_config;
		cur_config.seed = seed;
		res.push_back(cur_config);
	}
	return res;
}

ds::TaskSched read_sched_from_trace(const string& trace_file) {
	string line;
	std::ifstream trace(trace_file);
	assert(trace.is_open());
	ds::TaskSched result;
	while (std::getline(trace, line)) {
		int num_items = std::count(line.begin(), line.end(), '[');
		if (num_items == 0) 
			break;
		
		result.tasks.push_back({});

		static constexpr int MAX_MODELS = 32;
		static constexpr int MAX_MBATCHES = 256;
		bool mbatch_fwded[MAX_MODELS][MAX_MBATCHES];
		memset(mbatch_fwded, 0, sizeof(mbatch_fwded));
		for (int i = 0; i < num_items; ++i) {
			int start = line.find('[');
			int end = line.find(']');
			string item = line.substr(start, end - start + 1);
			line = line.substr(end + 1);
			int model_id, mbatch_id, stage_id, start_time, end_time;
			float mem_usage;
			sscanf(item.c_str(), "[%d,%d,%d,%d,%d,%f]", &model_id, &mbatch_id, &stage_id, &start_time, &end_time, &mem_usage);
			assert(model_id < MAX_MODELS && mbatch_id < MAX_MBATCHES);

			bool is_fwd = !mbatch_fwded[model_id][mbatch_id];
			mbatch_fwded[model_id][mbatch_id] = true;
			result.tasks.back().push_back({model_id, is_fwd});
			// printf("[%d, %d] ", model_id, is_fwd);
		}
		// printf("\n");
	}
	return result;
}

struct ExpConfig {
	vector<ds::ModelMeta> model_metas;
	vector<ds::SimAnnealConfig> sim_anneal_configs;
};

template<typename T>
std::vector<T> operator+(const std::vector<T> &a, const std::vector<T> &b) {
	std::vector result = a;
	result.insert(result.end(), b.begin(), b.end());
	return result;
}

ExpConfig get_config_exp0() {
	return {
		get_model_metas_helper(
			8,
			4,
			32,
			16,
			5,
			4,
			1.95,
			2
		),
		get_sim_anneal_memory_configs_helper(
			{
				200,
				0.999995,
				1e-25,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
			384*4
		)
	};
}

ExpConfig get_config_exp3() {
	return {
		get_model_metas_helper(
			8,
			8,
			32,
			32,
			5,
			2,
			1.95,
			1
		),
		get_sim_anneal_memory_configs_helper(
			{
				80,
				0.999995,
				1e-25,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
			384*4
		) + get_sim_anneal_memory_configs_helper(
			{
				1000,
				0.999995,
				1e-25,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
			384*4
		)
	};
}

ExpConfig get_config_exp6() {
	return {
		get_model_metas_helper(
			16,
			8,
			64,
			32,
			2,
			2,
			1.64,
			2
		),
		get_sim_anneal_memory_configs_helper(
			{
				80,
				0.999996,
				1e-25,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
			384*4
		) + get_sim_anneal_memory_configs_helper(
			{
				1000,
				0.999996,
				1e-25,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
			384*4
		)
	};
}

ExpConfig get_config_exp9() {
	return {
		get_model_metas_helper(
			16,
			16,
			64,
			64,
			2,
			1,
			1.64,
			1
		),
		get_sim_anneal_memory_configs_helper(
			{
				80,
				0.999996,
				1e-25,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
			384*4
		) + get_sim_anneal_memory_configs_helper(
			{
				1000,
				0.999996,
				1e-25,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
			384*4
		)
	};
}

ExpConfig get_config_exp10() {
	return {
		get_model_metas_helper(
			16,
			16,
			32,
			32,
			2,
			1,
			1.64,
			1
		),
		get_sim_anneal_memory_configs_helper(
			{
				80,
				0.999995,
				1e-25,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
			384*4
		) + get_sim_anneal_memory_configs_helper(
			{
				1000,
				0.999995,
				1e-25,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
			384*4
		)
	};
}

int main(int argc, char* argv[]) {
	if (argc != 2) {
		printf("Usage: %s <exp-id>\n", argv[0]);
		return 1;
	}
	string exp_id = argv[1];
	string trace_file = "results/trace_" + exp_id + ".txt";
	ds::TaskSched sched = read_sched_from_trace(trace_file);
	int num_nodes = sched.tasks.size();

	std::unordered_map<string, ExpConfig> exp_configs = {
		{"exp0", get_config_exp0()},
		{"exp3", get_config_exp3()},
		{"exp6", get_config_exp6()},
		{"exp9", get_config_exp9()},
		{"exp10", get_config_exp10()}
	};
	if (exp_configs.find(exp_id) == exp_configs.end()) {
		printf("Invalid exp-id: %s\n", exp_id.c_str());
		return 1;
	}
	ExpConfig config = exp_configs[exp_id];
	vector<ds::ModelMeta> model_metas = config.model_metas;
	vector<ds::SimAnnealConfig> sim_anneal_configs = config.sim_anneal_configs;

	DualEgoSolver plain_solver(num_nodes, model_metas);
	ds::Trace best_trace;
	best_trace = plain_solver.task_sched2trace(sched);

	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < (int)sim_anneal_configs.size(); ++i) {
		int thread_id = omp_get_thread_num();
		printf("Thread %3d: Starting with sim anneal config #%5d: %s\n", thread_id, i, ds::fmt_sim_anneal_config(sim_anneal_configs[i]).c_str());
		DualEgoSolver solver(num_nodes, model_metas);
		ds::TaskSched optimized_sched = solver.optimize_peak_memory(sim_anneal_configs[i], sched);
		ds::Trace optimized_trace = solver.task_sched2trace(optimized_sched);
		#pragma omp critical
		{
			printf("Thread %3d: got peak memory %f (cur best: %f)\n", thread_id, optimized_trace.peak_memory_usage, best_trace.peak_memory_usage);
			if (optimized_trace.peak_memory_usage < best_trace.peak_memory_usage) {
				printf("\033[32mThread %3d: got a better trace with peak memory %f\033[0m\n", thread_id, optimized_trace.peak_memory_usage);
				best_trace = optimized_trace;
			}
		}
	}

	plain_solver.print_trace(best_trace);

	int cur_time = time(0);
	FILE* fp = fopen((trace_file + ".mem-optimized-" + std::to_string(cur_time)).c_str(), "w");
	if (fp == NULL) {
		printf("Failed to open file: %s\n", strerror(errno));
		return 1;
	}
	plain_solver.print_trace(best_trace, fp);

	return 0;
}