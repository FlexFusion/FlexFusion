#include "DualEgoSolver.h"

#include <chrono>
#include <cstring>
#include <limits>

#include "mpi.h"
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
	
	int num_nodes;
	vector<ds::ModelMeta> model_metas;
	vector<ds::SimAnnealConfig> sim_anneal_configs;

	int num_traceitems;

	ds::Trace run_master_routine() {
		size_t trace_size = offsetof(ds::Trace, trace) + sizeof(ds::TraceItem)*num_traceitems;
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

		int finished_jobs = 0;
		int tot_num_jobs = sim_anneal_configs.size();
		std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

		while (num_terminated_workers < world_size-1) {
			// Receive a result from any worker
			MPI_Status sender_status;
			MPI_CHECK(MPI_Recv(recv_buf, trace_size, MPI_CHAR, MPI_ANY_SOURCE, TAG_TRACE, MPI_COMM_WORLD, &sender_status));

			// Send the worker a new job, or inform that there is no remaining jobs
			send_a_job_to_worker(sender_status.MPI_SOURCE);

			// Forge the "cur_trace" which contains every field other than "trace"
			ds::Trace *cur_trace_ptr = (ds::Trace*)recv_buf;
			ds::Trace cur_trace = {
				cur_trace_ptr->time_usage,
				cur_trace_ptr->peak_memory_usage,
				cur_trace_ptr->sum_peak_memory_usage,
				cur_trace_ptr->fin_time_sum,
				{}
			};
			if (ds::is_trace_more_optimal(cur_trace, best_trace)) {
				// Save the trace
				printf("\033[32mWorker %d made a breakthrough! The best answer is updated to %s!\033[0m\n", sender_status.MPI_SOURCE-1, ds::fmt_trace(cur_trace).c_str());
				best_trace = cur_trace;
				best_trace.trace.resize(num_traceitems);
				memcpy(best_trace.trace.data(), recv_buf+offsetof(ds::Trace, trace), sizeof(ds::TraceItem)*num_traceitems);
			}

			// Print the progress
			finished_jobs += 1;
			if (finished_jobs%20 == 0) {
				std::chrono::steady_clock::time_point cur_time = std::chrono::steady_clock::now();
				double elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(cur_time-start_time).count();
				printf("\033[33mProgress: %d/%d (%.2f%%), elapsed time: %.2fs, estimated time left: %.2fs. Current best: %s\033[0m\n",
					finished_jobs, tot_num_jobs, 100.0*finished_jobs/tot_num_jobs,
					elapsed_time,
					elapsed_time/finished_jobs*(tot_num_jobs-finished_jobs),
					ds::fmt_trace(best_trace).c_str());
			}
		}

		delete[] recv_buf;
		usleep(100000);	// Sleep for 0.1s to let workers end first
		return best_trace;
	}

	void run_worker_routine() {
		size_t trace_size = offsetof(ds::Trace, trace) + sizeof(ds::TraceItem)*num_traceitems;
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
			printf("Worker %d received job with config %s\n", rank-1, ds::fmt_sim_anneal_config(cur_config).c_str());
			// Run it
			DualEgoSolver solver(num_nodes, model_metas, cur_config);
			ds::Trace cur_trace = solver.solve();
			printf("Worker %d gets an answer of {%d, %d, %d, %d}\n", rank-1, cur_trace.time_usage, cur_trace.peak_memory_usage, cur_trace.sum_peak_memory_usage, cur_trace.fin_time_sum);
			assert((int)cur_trace.trace.size() == num_traceitems);
			// Handle the result back to the master
			ds::Trace *cur_trace_ptr = (ds::Trace*)send_buf;
			cur_trace_ptr->time_usage = cur_trace.time_usage;
			cur_trace_ptr->peak_memory_usage = cur_trace.peak_memory_usage;
			cur_trace_ptr->sum_peak_memory_usage = cur_trace.sum_peak_memory_usage;
			cur_trace_ptr->fin_time_sum = cur_trace.fin_time_sum;
			memcpy(send_buf+offsetof(ds::Trace, trace), cur_trace.trace.data(), sizeof(ds::TraceItem)*num_traceitems);
			MPI_CHECK(MPI_Send(send_buf, trace_size, MPI_CHAR, 0, TAG_TRACE, MPI_COMM_WORLD));
		}

		printf("Worker %d finds no job, exiting...\n", rank-1);
		delete[] send_buf;
	}

public:
	ParallelDualEgoSolver(
		int num_nodes,
		vector<ds::ModelMeta> const& model_metas,
		vector<ds::SimAnnealConfig> const& sim_anneal_config):
		num_nodes(num_nodes),
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

		num_traceitems = 0;
		for (const ds::ModelMeta &meta : model_metas) {
			int num_stages = meta.stage2node.size();
			num_traceitems += 2 * meta.num_mbatches * num_stages;
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