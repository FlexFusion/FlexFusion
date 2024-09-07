#include "DualEgoSolver.h"
using ds = DualEgoSolver;

int main() {
	// The large model
	vector<ds::ModelMeta> model_metas;
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
	// 	{3, 2, 1, 0},
	// 	1,
	// 	"\033[33m", "\033[34m"
	// });

	ds::sim_anneal_init_t init_method = ds::sim_anneal_init_t::GREEDY;
	ds::SimAnnealConfig sim_anneal_config_e2e = {
		4,
		0.999995,
		1e-14,
		0,
		ds::sim_anneal_disturb_t::RANDOM_ADJACENT_SWAP,
	};
	ds::SimAnnealConfig sim_anneal_config_memory = {
		4,
		0.999995,
		1e-14,
		0,
		ds::sim_anneal_disturb_t::RANDOM_MOVE,
	};

	ds solver(num_nodes, model_metas);
	ds::TaskSched init_sched = solver.get_init_task_sched(init_method);
	printf("Init trace:\n");
	solver.print_trace(solver.task_sched2trace(init_sched));
	printf("\n");

	ds::TaskSched e2e_optimized_sched = solver.optimize_e2e_time(sim_anneal_config_e2e, init_sched);
	printf("E2E optimized trace:\n");
	solver.print_trace(solver.task_sched2trace(e2e_optimized_sched));
	printf("\n");

	ds::TaskSched memory_optimized_sched = solver.optimize_peak_memory(sim_anneal_config_memory, e2e_optimized_sched);
	printf("Memory optimized trace:\n");
	solver.print_trace(solver.task_sched2trace(memory_optimized_sched));
	printf("\n");

	return 0;
}
