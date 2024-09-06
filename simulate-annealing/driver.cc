#include "DualEgoSolver.h"

int main() {
	// The large model
	vector<DualEgoSolver::ModelMeta> model_metas;
	int num_nodes = 16;
	model_metas.push_back({
		32,							// #micro-batches
		2,							// fwd-time
		4,							// bwd-time
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},				// stage2node mapping
		2,
		"\033[31m", "\033[32m"		// fwd-color, bwd-color
	});
	model_metas.push_back({
		16,
		1,
		2,
		{15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0},
		1,
		"\033[35m", "\033[36m"
	});

	// int num_nodes = 4;
	// model_metas.push_back({
	// 	4,							// #micro-batches
	// 	2,							// fwd-time
	// 	4,							// bwd-time
	// 	{0, 1, 2, 3},				// stage2node mapping
	// 	2,
	// 	"\033[31m", "\033[32m"		// fwd-color, bwd-color
	// });
	// model_metas.push_back({
	// 	2,
	// 	1,
	// 	2,
	// 	{1, 0},
	// 	1,
	// 	"\033[33m", "\033[34m"
	// });
	// model_metas.push_back({
	// 	2,
	// 	1,
	// 	2,
	// 	{3, 2},
	// 	1,
	// 	"\033[35m", "\033[36m"
	// });

	// DualEgoSolver::SimAnnealConfig sim_anneal_config = {
	// 	4,
	// 	0.999995,
	// 	1e-14,
	// 	DualEgoSolver::sim_anneal_disturb_t::RANDOM_ADJACENT_SWAP,
	// 	4,
	// 	0.999995,
	// 	1e-14,
	// 	DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE,
	// 	0,
	// 	DualEgoSolver::sim_anneal_init_t::ONE_F_ONE_B,
	// };

	DualEgoSolver::SimAnnealConfig sim_anneal_config = {
		4,
		0.999995,
		1e-14,
		DualEgoSolver::sim_anneal_disturb_t::RANDOM_ADJACENT_SWAP,
		4,
		0.999995,
		1e-14,
		DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE,
		0,
		DualEgoSolver::sim_anneal_init_t::GREEDY,
	};

	DualEgoSolver solver(num_nodes, model_metas, sim_anneal_config);
	DualEgoSolver::Trace best_trace = solver.solve();
	// DualEgoSolver::Trace best_trace = solver.solve_greedy();

	solver.print_trace(best_trace);

	return 0;
}
