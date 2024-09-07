#include "ParallelDualEgoSolver.h"
#include "mpi.h"

int main(int argc, char* argv[]) {
	using ds = DualEgoSolver;

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
		{7, 6, 5, 4, 3, 2, 1, 0},
		1,
		"\033[35m", "\033[36m"
	});
	model_metas.push_back({
		16,
		1,
		2,
		{15, 14, 13, 12, 11, 10, 9, 8},
		1,
		"\033[33m", "\033[34m"
	});

	vector<pair<ds::sim_anneal_init_t, ds::SimAnnealConfig>> sim_anneal_e2e_configs;
	for (int seed = 0; seed < (192-1); ++seed) {
		DualEgoSolver::SimAnnealConfig cur_config = {
			0,
			0.999995,
			0,
			seed,
			DualEgoSolver::sim_anneal_disturb_t::RANDOM_ADJACENT_SWAP
		};
		sim_anneal_e2e_configs.push_back({
			DualEgoSolver::sim_anneal_init_t::GREEDY,
			cur_config
		});
	}

	vector<ds::SimAnnealConfig> sim_anneal_memory_configs;
	for (int seed = 0; seed < 4; ++seed) {
		DualEgoSolver::SimAnnealConfig cur_config = {
			10,
			0.999995,
			1e-16,
			seed,
			DualEgoSolver::sim_anneal_disturb_t::RANDOM_MOVE
		};
		sim_anneal_memory_configs.push_back(cur_config);
	}

	MPI_CHECK(MPI_Init(&argc, &argv));
	ParallelDualEgoSolver parallel_solver(num_nodes, model_metas, sim_anneal_e2e_configs, sim_anneal_memory_configs);
	ds::Trace best_trace = parallel_solver.solve();

	int rank;
	MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	if (rank == 0) {
		ds trace_printer(num_nodes, model_metas);
		trace_printer.print_trace(best_trace);
	}

	MPI_CHECK(MPI_Finalize());

	return 0;
}
