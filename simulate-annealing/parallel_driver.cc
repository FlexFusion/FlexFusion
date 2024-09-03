#include "ParallelDualEgoSolver.h"
#include "mpi.h"
#include "mpi_proto.h"

int main(int argc, char* argv[]) {
	int num_stages = 16;
	int num_small_models = 1;
	int large_model_fwd_time = 2;
	int large_model_bwd_time = 4;
	int small_model_fwd_time = 1;
	int small_model_bwd_time = 2;
	int num_large_model_mbatchs = 16;
	int num_small_model_mbatchs = 16;

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

	vector<DualEgoSolver::SimAnnealConfig> sim_anneal_configs;
	for (int seed = 0; seed < 30; ++seed) {
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
	ParallelDualEgoSolver parallel_solver(num_stages, model_metas, sim_anneal_configs);
	DualEgoSolver::Trace best_trace = parallel_solver.solve();

	int rank;
	MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	if (rank == 0) {
		printf("Best time usage: %d\n", best_trace.time_usage);
		// Trace printing do not care about the sim_anneal_config, so just feed with a random one
		DualEgoSolver trace_printer(num_stages, model_metas, sim_anneal_configs[0]);
		trace_printer.print_trace(best_trace);
	}

	MPI_CHECK(MPI_Finalize());

	return 0;
}
