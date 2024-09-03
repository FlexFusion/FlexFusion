#include "DualEgoSolver.h"

int main() {
	int num_nodes = 4;

	// The large model
	vector<DualEgoSolver::ModelMeta> model_metas;
	model_metas.push_back({
		4,
		2,
		4,
		{0, 1, 2, 3},
		"\033[31m", "\033[32m"
	});
	model_metas.push_back({
		2,
		1,
		2,
		{1, 0},
		"\033[33m", "\033[34m"
	});
	model_metas.push_back({
		2,
		1,
		2,
		{3, 2},
		"\033[35m", "\033[36m"
	});

	DualEgoSolver::SimAnnealConfig sim_anneal_config = {
		1e7,
		0.99999,
		1e-9,
		0,
		DualEgoSolver::sim_anneal_init_t::FFFFBBBB,
		DualEgoSolver::sim_anneal_disturb_t::RANDOM_SWAP
	};

	DualEgoSolver solver(num_nodes, model_metas, sim_anneal_config);
	DualEgoSolver::Trace best_trace = solver.solve();

	printf("Best time usage: %d\n", best_trace.time_usage);
	solver.print_trace(best_trace);

	return 0;
}
