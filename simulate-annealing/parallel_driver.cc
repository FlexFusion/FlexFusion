#include "ParallelDualEgoSolver.h"
#include "mpi.h"

int main(int argc, char* argv[]) {

	// The large model
	vector<DualEgoSolver::ModelMeta> model_metas;
	int num_nodes = 16;
	model_metas.push_back({
		32,							// #micro-batches
		2,							// fwd-time
		4,							// bwd-time
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},				// stage2node mapping
		"\033[31m", "\033[32m"		// fwd-color, bwd-color
	});
	model_metas.push_back({
		16,
		1,
		2,
		{15, 14, 13, 12, 11, 10, 9, 8},
		"\033[33m", "\033[34m"
	});
	model_metas.push_back({
		16,
		1,
		2,
		{7, 6, 5, 4, 3, 2, 1, 0},
		"\033[35m", "\033[36m"
	});

	vector<DualEgoSolver::SimAnnealConfig> sim_anneal_configs;
	for (int seed = 0; seed < 768; ++seed) {
		DualEgoSolver::SimAnnealConfig sim_anneal_config = {
			1e6,
			0.99999,
			1e-8,
			seed,
			DualEgoSolver::sim_anneal_init_t::FFFFBBBB,
			DualEgoSolver::sim_anneal_disturb_t::RANDOM_SWAP
		};
		sim_anneal_configs.push_back(sim_anneal_config);
	}

	MPI_CHECK(MPI_Init(&argc, &argv));
	ParallelDualEgoSolver parallel_solver(num_nodes, model_metas, sim_anneal_configs);
	DualEgoSolver::Trace best_trace = parallel_solver.solve();

	int rank;
	MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	if (rank == 0) {
		printf("Best time usage: %d\n", best_trace.time_usage);
		// Trace printing do not care about the sim_anneal_config, so just feed with a random one
		DualEgoSolver trace_printer(num_nodes, model_metas, sim_anneal_configs[0]);
		trace_printer.print_trace(best_trace);
	}

	MPI_CHECK(MPI_Finalize());

	return 0;
}
