#include "DualEgoSolver.h"

int main() {
	int num_stages = 4;
	int num_small_models = 4;
	int large_model_fwd_time = 2;
	int large_model_bwd_time = 4;
	int small_model_fwd_time = 1;
	int small_model_bwd_time = 2;
	int num_large_model_mbatchs = 4;
	int num_small_model_mbatchs = 1;

	// int num_stages = 8;
	// int num_small_models = 1;
	// int large_model_fwd_time = 2;
	// int large_model_bwd_time = 4;
	// int small_model_fwd_time = 1;
	// int small_model_bwd_time = 2;
	// int num_large_model_mbatchs = 8;
	// int num_small_model_mbatchs = 8;

	vector<DualEgoSolver::ModelMeta> model_metas;
	model_metas.push_back({
		num_large_model_mbatchs,
		large_model_fwd_time,
		large_model_bwd_time,
		true,
		"\033[31m", "\033[32m"
	});
	for (int i = 0; i < num_small_models; ++i) {
		model_metas.push_back({
			num_small_model_mbatchs,
			small_model_fwd_time,
			small_model_bwd_time,
			false,
			"\033[33m", "\033[34m"
		});
	}

	DualEgoSolver::SimAnnealConfig sim_anneal_config = {
		1e7,
		0.99999,
		1e-9,
		0,
		DualEgoSolver::sim_anneal_init_t::FFFFBBBB,
		DualEgoSolver::sim_anneal_disturb_t::RANDOM_SWAP
	};

	DualEgoSolver solver(num_stages, model_metas, sim_anneal_config);
	DualEgoSolver::Trace best_trace = solver.solve();

	printf("Best time usage: %d\n", best_trace.time_usage);
	solver.print_trace(best_trace);

	return 0;
}
