#include "ParallelDualEgoSolver.h"

#include <sys/stat.h>

#include "mpi.h"

using ds = DualEgoSolver;

struct ExpConfig {
	int num_nodes;
	vector<ds::ModelMeta> model_metas;
	vector<pair<ds::sim_anneal_init_t, ds::SimAnnealConfig>> sim_anneal_e2e_configs;
	vector<ds::SimAnnealConfig> sim_anneal_memory_configs;
};

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

auto get_sim_anneal_e2e_configs_helper(ds::sim_anneal_init_t init_method, const ds::SimAnnealConfig &base_config, int num_seeds) {
	vector<pair<ds::sim_anneal_init_t, ds::SimAnnealConfig>> res;
	for (int seed = 0; seed < num_seeds; ++seed) {
		ds::SimAnnealConfig cur_config = base_config;
		cur_config.seed = seed;
		res.push_back({init_method, cur_config});
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

ExpConfig get_config_exp0() {
	return {
		8,
		get_model_metas_helper(
			8,
			4,
			32,
			16,
			5,
			4,
			1.95,
			2),
		get_sim_anneal_e2e_configs_helper(
			ds::sim_anneal_init_t::GREEDY,
			 {
				10,
				0.999995,
				1e-12,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_ADJACENT_SWAP
			},
			(768-1)
		),
		get_sim_anneal_memory_configs_helper(
			{
				20,
				0.999995,
				1e-20,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
		4)
	};
}

ExpConfig get_config_exp1() {
	return {
		8,
		get_model_metas_helper(
			8,
			4,
			16,
			8,
			5,
			4,
			1.95,
			2),
		get_sim_anneal_e2e_configs_helper(
			ds::sim_anneal_init_t::GREEDY,
			 {
				10,
				0.999995,
				1e-14,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_ADJACENT_SWAP
			},
			(768-1)
		),
		get_sim_anneal_memory_configs_helper(
			{
				10,
				0.999995,
				1e-20,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
		4)
	};
}

ExpConfig get_config_exp2() {
	return {
		8,
		get_model_metas_helper(
			8,
			4,
			8,
			4,
			5,
			4,
			1.95,
			2),
		get_sim_anneal_e2e_configs_helper(
			ds::sim_anneal_init_t::GREEDY,
			 {
				0,
				0,
				0,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_ADJACENT_SWAP
			},
			768-1
		),
		get_sim_anneal_memory_configs_helper(
			{
				10,
				0.999995,
				1e-16,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
		1)
	};
}

ExpConfig get_config_exp3() {
	return {
		8,
		get_model_metas_helper(
			8,
			8,
			32,
			32,
			5,
			2,
			1.95,
			1),
		get_sim_anneal_e2e_configs_helper(
			ds::sim_anneal_init_t::GREEDY,
			 {
				50,
				0.999995,
				1e-22,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_ADJACENT_SWAP
			},
			(768-1)*2
		),
		get_sim_anneal_memory_configs_helper(
			{
				20,
				0.999995,
				1e-26,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
		8)
	};
}

ExpConfig get_config_exp4() {
	return {
		8,
		get_model_metas_helper(
			8,
			8,
			16,
			16,
			5,
			2,
			1.95,
			1),
		get_sim_anneal_e2e_configs_helper(
			ds::sim_anneal_init_t::GREEDY,
			 {
				10,
				0.999995,
				1e-22,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_ADJACENT_SWAP
			},
			(768-1)
		),
		get_sim_anneal_memory_configs_helper(
			{
				10,
				0.999995,
				1e-26,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
		8)
	};
}

ExpConfig get_config_exp5() {
	return {
		8,
		get_model_metas_helper(
			8,
			8,
			8,
			8,
			5,
			2,
			1.95,
			1),
		get_sim_anneal_e2e_configs_helper(
			ds::sim_anneal_init_t::GREEDY,
			 {
				0,
				0,
				0,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_ADJACENT_SWAP
			},
			768-1
		),
		get_sim_anneal_memory_configs_helper(
			{
				10,
				0.999995,
				1e-16,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
		2)
	};
}

ExpConfig get_config_exp6() {
	return {
		16,
		get_model_metas_helper(
			16,
			8,
			64,
			32,
			2,
			2,
			1.64,
			2),
		get_sim_anneal_e2e_configs_helper(
			ds::sim_anneal_init_t::GREEDY,
			 {
				200,
				0.999997,
				1e-16,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_ADJACENT_SWAP
			},
			(768-1)*8
		),
		get_sim_anneal_memory_configs_helper(
			{
				200,
				0.999997,
				1e-20,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
		16)
	};
}

ExpConfig get_config_exp7() {
	return {
		16,
		get_model_metas_helper(
			16,
			8,
			32,
			16,
			2,
			2,
			1.64,
			2),
		get_sim_anneal_e2e_configs_helper(
			ds::sim_anneal_init_t::GREEDY,
			 {
				50,
				0.999995,
				1e-16,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_ADJACENT_SWAP
			},
			(768-1)*4
		),
		get_sim_anneal_memory_configs_helper(
			{
				50,
				0.999995,
				1e-20,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
		4)
	};
}

ExpConfig get_config_exp8() {
	return {
		16,
		get_model_metas_helper(
			16,
			8,
			16,
			8,
			2,
			2,
			1.64,
			2),
		get_sim_anneal_e2e_configs_helper(
			ds::sim_anneal_init_t::GREEDY,
			 {
				50,
				0.999995,
				1e-16,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_ADJACENT_SWAP
			},
			(768-1)*4
		),
		get_sim_anneal_memory_configs_helper(
			{
				50,
				0.999995,
				1e-20,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
		4)
	};
}

ExpConfig get_config_exp9() {
	return {
		16,
		get_model_metas_helper(
			16,
			16,
			64,
			64,
			2,
			1,
			1.64,
			1),
		get_sim_anneal_e2e_configs_helper(
			ds::sim_anneal_init_t::GREEDY,
			 {
				40,
				0.999995,
				1e-22,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_ADJACENT_SWAP
			},
			(768-1)*8
		),
		get_sim_anneal_memory_configs_helper(
			{
				200,
				0.999995,
				1e-20,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
		8)
	};
}

ExpConfig get_config_exp10() {
	return {
		16,
		get_model_metas_helper(
			16,
			16,
			32,
	32,
			2,
			1,
			1.64,
			1),
		get_sim_anneal_e2e_configs_helper(
			ds::sim_anneal_init_t::GREEDY,
			 {
				20,
				0.999995,
				1e-20,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_ADJACENT_SWAP
			},
			(768-1)*4
		),
		get_sim_anneal_memory_configs_helper(
			{
				20,
				0.999995,
				1e-20,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
		4)
	};
}

ExpConfig get_config_exp11() {
	return {
		16,
		get_model_metas_helper(
			16,
			16,
			16,
			16,
			2,
			1,
			1.64,
			1),
		get_sim_anneal_e2e_configs_helper(
			ds::sim_anneal_init_t::GREEDY,
			 {
				10,
				0.999995,
				1e-14,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_ADJACENT_SWAP
			},
			768-1
		),
		get_sim_anneal_memory_configs_helper(
			{
				10,
				0.999995,
				1e-16,
				0,
				DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
			},
		2)
	};
}



void run_exp(const string& exp_name, const ExpConfig &config) {
	auto start_time = std::chrono::steady_clock::now();
	int rank;
	MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	if (rank == 0) {
		// Print the result of greedy
		DualEgoSolver solver(config.num_nodes, config.model_metas);
		ds::Trace trace = solver.solve_greedy();
		solver.print_trace(trace);
	}
	ParallelDualEgoSolver parallel_solver(config.num_nodes, config.model_metas, config.sim_anneal_e2e_configs, config.sim_anneal_memory_configs);
	ds::Trace best_trace = parallel_solver.solve();
	if (rank == 0) {
		ds trace_printer(config.num_nodes, config.model_metas);
		trace_printer.print_trace(best_trace);

		if (mkdir("results", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
			if (errno != EEXIST) {
				printf("Failed to create directory: %s\n", strerror(errno));
				return;
			}
		}
		FILE* fp = fopen(("results/trace_" + exp_name + ".txt").c_str(), "w");
		if (fp == NULL) {
			printf("Failed to open file: %s\n", strerror(errno));
			return;
		}
		trace_printer.print_trace(best_trace, fp);
		auto end_time = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
		fprintf(fp, "Time usage: %ld ms\n", duration.count());
		fclose(fp);
	}
}

int main(int argc, char* argv[]) {
	MPI_CHECK(MPI_Init(&argc, &argv));

	std::unordered_map<std::string, ExpConfig> exp_configs_map = {
		{"exp0", get_config_exp0()},
		{"exp1", get_config_exp1()},
		{"exp2", get_config_exp2()},
		{"exp3", get_config_exp3()},
		{"exp4", get_config_exp4()},
		{"exp5", get_config_exp5()},
		{"exp6", get_config_exp6()},
		{"exp7", get_config_exp7()},
		{"exp8", get_config_exp8()},
		{"exp9", get_config_exp9()},
		{"exp10", get_config_exp10()},
		{"exp11", get_config_exp11()}
	};

	if (argc != 2) {
		printf("Usage: %s <exp_name>\n", argv[0]);
		return 1;
	}

	std::string exp_name = argv[1];
	if (exp_configs_map.find(exp_name) == exp_configs_map.end()) {
		printf("Invalid exp_name: %s\n", exp_name.c_str());
		return 1;
	}

	run_exp(exp_name, exp_configs_map[exp_name]);

	MPI_CHECK(MPI_Finalize());

	return 0;
}
