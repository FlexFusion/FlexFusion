#include "DualEgoSolver.h"

#include <cstring>
#include <limits>

#include "mpi.h"
#include "mpi_proto.h"
#include "unistd.h"

#define MPI_CHECK(x) \
	do { \
		int __return_code_ = x;	\
		if (__return_code_ != MPI_SUCCESS) { \
			printf("MPI Error on %s:%d: %d\n", __FILE__, __LINE__, __return_code_); \
		} \
	} while(0);

class ParallelDualEgoSolver {
private:
	using ds = DualEgoSolver;
	static constexpr int TAG_TRACE = 1;
	static constexpr int TAG_WHETHER_HAS_JOB = 2;
	static constexpr int TAG_JOB_CONFIG = 3;

	int rank, world_size;
	int num_stages;
	vector<ds::ModelMeta> model_metas;
	vector<ds::SimAnnealConfig> sim_anneal_configs;

	ds::Trace run_master_routine() {
		int sum_num_mbatches = 0;
		for (ds::ModelMeta const& meta : model_metas)
			sum_num_mbatches += meta.num_mbatches;
		int num_traceitems = sum_num_mbatches * (2*num_stages);
		size_t trace_size = sizeof(ds::Trace::time_usage) + sizeof(ds::TraceItem)*num_traceitems;
		char* recv_buf = new char[trace_size];

		ds::Trace best_trace;
		best_trace.time_usage = std::numeric_limits<decltype(ds::Trace::time_usage)>::max();
		int next_config_index = 0;
		int num_terminated_workers = 0;

		auto send_a_job_to_worker = [&](int worker_rank) {
			char has_job = (next_config_index != (int)sim_anneal_configs.size());
			MPI_CHECK(MPI_Send(&has_job, 1, MPI_CHAR, worker_rank, TAG_WHETHER_HAS_JOB, MPI_COMM_WORLD));
			if (has_job) {
				ds::SimAnnealConfig cur_config = sim_anneal_configs[next_config_index];
				next_config_index += 1;
				MPI_CHECK(MPI_Send(&cur_config, sizeof(ds::SimAnnealConfig), MPI_CHAR, worker_rank, TAG_JOB_CONFIG, MPI_COMM_WORLD));
			} else {
				num_terminated_workers += 1;
			}
		};
		
		// Give every worker a job
		for (int worker_rank = 1; worker_rank < world_size; ++worker_rank)
			send_a_job_to_worker(worker_rank);

		while (num_terminated_workers < world_size-1) {
			// Receive a result from any worker
			MPI_Status sender_status;
			MPI_CHECK(MPI_Recv(recv_buf, trace_size, MPI_CHAR, MPI_ANY_SOURCE, TAG_TRACE, MPI_COMM_WORLD, &sender_status));

			// Send the worker a new job, or inform that there is no remaining jobs
			send_a_job_to_worker(sender_status.MPI_SOURCE);

			// Save the trace if necessary
			ds::Trace cur_trace;
			cur_trace.time_usage = *((int*)recv_buf);
			cur_trace.trace.resize(num_traceitems);
			memcpy(cur_trace.trace.data(), recv_buf+sizeof(ds::Trace::time_usage), sizeof(ds::TraceItem)*num_traceitems);

			if (cur_trace.time_usage < best_trace.time_usage) {
				printf("\033[32mWorker %d made a breakthrough! The best answer is updated to %d!\033[0m\n", sender_status.MPI_SOURCE-1, cur_trace.time_usage);
				best_trace = cur_trace;
			}
		}

		delete[] recv_buf;
		MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
		return best_trace;
	}

	void run_worker_routine() {
		int sum_num_mbatches = 0;
		for (ds::ModelMeta const& meta : model_metas)
			sum_num_mbatches += meta.num_mbatches;
		int num_traceitems = sum_num_mbatches * (2*num_stages);
		size_t trace_size = sizeof(ds::Trace::time_usage) + sizeof(ds::TraceItem)*num_traceitems;
		char* send_buf = new char[trace_size];
		MPI_Status status;

		while (true) {
			// See whether there are any remaining jobs
			char has_job;
			MPI_CHECK(MPI_Recv(&has_job, 1, MPI_CHAR, 0, TAG_WHETHER_HAS_JOB, MPI_COMM_WORLD, &status));
			if (!has_job) {
				break;
			}
			// Get the job detail
			ds::SimAnnealConfig cur_config;
			MPI_CHECK(MPI_Recv(&cur_config, sizeof(ds::SimAnnealConfig), MPI_CHAR, 0, TAG_JOB_CONFIG, MPI_COMM_WORLD, &status));
			printf("Worker %d received job with seed %d\n", rank-1, cur_config.seed);
			// Run it
			DualEgoSolver solver(num_stages, model_metas, cur_config);
			ds::Trace cur_trace = solver.solve();
			printf("Worker %d gets an answer of %d\n", rank-1, cur_trace.time_usage);
			assert((int)cur_trace.trace.size() == num_traceitems);
			// Handle the result back to the master
			*((int*)send_buf) = cur_trace.time_usage;
			memcpy(send_buf+sizeof(ds::Trace::time_usage), cur_trace.trace.data(), sizeof(ds::TraceItem)*num_traceitems);
			MPI_CHECK(MPI_Send(send_buf, trace_size, MPI_CHAR, 0, TAG_TRACE, MPI_COMM_WORLD));
		}

		printf("Worker %d finds no job, exiting...\n", rank-1);
		MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
		delete[] send_buf;
	}

public:
	ParallelDualEgoSolver(
		int num_stages,
		vector<ds::ModelMeta> const& model_metas,
		vector<ds::SimAnnealConfig> const& sim_anneal_config):
		num_stages(num_stages),
		model_metas(model_metas),
		sim_anneal_configs(sim_anneal_config) {
		int mpi_inited;
		MPI_CHECK(MPI_Initialized(&mpi_inited));
		assert(mpi_inited && "Please call MPI_Init before initializing the parallel dual ego solver!");

		MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
		MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
		assert(world_size > 1 && "The world size must be greater than one");

		MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
		pid_t pid = getpid();
		if (rank == 0) {
			printf("Master started with pid = %d\n", pid);
		} else {
			printf("Worker %d started with pid = %d\n", rank-1, pid);
		}
	}

	ds::Trace solve() {
		if (rank == 0) {
			// I am the master
			return run_master_routine();
		} else {
			// I am the dog of Astar
			run_worker_routine();
			return {};
		}
	}
};	