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
	using ds = DualEgoSolver;

private:
	static constexpr int TAG_TRACE = 1;
	static constexpr int TAG_WHETHER_HAS_JOB = 2;
	static constexpr int TAG_JOB_CONFIG = 3;
	static constexpr int TAG_REQUEST_BEST_TIME_USAGE = 4;
	static constexpr int TAG_RESPONSE_BEST_TIME_USAGE = 5;
	static constexpr int TAG_REQUEST_JOB = 6;

	int rank, world_size;
	
	int num_nodes;
	vector<ds::ModelMeta> model_metas;
	vector<pair<ds::sim_anneal_init_t, ds::SimAnnealConfig>> sim_anneal_e2e_configs;
	vector<ds::SimAnnealConfig> sim_anneal_memory_configs;

	int num_traceitems;

	ds::Trace run_master_routine() {
		size_t trace_size = offsetof(ds::Trace, trace) + sizeof(ds::TraceItem)*num_traceitems;
		char* recv_buf = new char[trace_size];

		ds::Trace best_trace;
		best_trace.time_usage = std::numeric_limits<decltype(ds::Trace::time_usage)>::max();
		int next_config_index = 0;
		int num_terminated_workers = 0;

		bool requested_jobs_before[world_size];
		memset(requested_jobs_before, 0, sizeof(requested_jobs_before));
		int finished_jobs = 0;
		int tot_num_jobs = sim_anneal_e2e_configs.size();
		std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

		while (num_terminated_workers < world_size-1) {
			// Receive a message from any worker
			int message_id;
			MPI_Status sender_status;
			MPI_CHECK(MPI_Recv(&message_id, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &sender_status));
			int worker_rank = sender_status.MPI_SOURCE;
			if (message_id == TAG_TRACE) {
				// A worker has finished a job and is going to send me a trace
				MPI_CHECK(MPI_Recv(recv_buf, trace_size, MPI_CHAR, worker_rank, TAG_TRACE, MPI_COMM_WORLD, &sender_status));

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
					printf("\033[32mWorker %d made a breakthrough! The best answer is updated to %s!\033[0m\n", worker_rank-1, ds::fmt_trace(cur_trace).c_str());
					best_trace = cur_trace;
					best_trace.trace.resize(num_traceitems);
					memcpy(best_trace.trace.data(), recv_buf+offsetof(ds::Trace, trace), sizeof(ds::TraceItem)*num_traceitems);
				}
			} else if (message_id == TAG_REQUEST_BEST_TIME_USAGE) {
				// A worker wants to know the best time usage
				int cur_best_time_usage = best_trace.time_usage;
				MPI_CHECK(MPI_Send(&cur_best_time_usage, 1, MPI_INT, worker_rank, TAG_RESPONSE_BEST_TIME_USAGE, MPI_COMM_WORLD));
			} else if (message_id == TAG_REQUEST_JOB) {
				// A worker wants to request a job
				// Send the worker a new job, or inform that there is no remaining jobs
				char has_job = (next_config_index != (int)sim_anneal_e2e_configs.size());
				MPI_CHECK(MPI_Send(&has_job, 1, MPI_CHAR, worker_rank, TAG_WHETHER_HAS_JOB, MPI_COMM_WORLD));
				if (has_job) {
					int cur_e2e_config_index = next_config_index;
					next_config_index += 1;
					MPI_CHECK(MPI_Send(&cur_e2e_config_index, 1, MPI_INT, worker_rank, TAG_JOB_CONFIG, MPI_COMM_WORLD));
				} else {
					num_terminated_workers += 1;
				}
				if (requested_jobs_before[worker_rank]) {
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
				requested_jobs_before[worker_rank] = true;
			} else {
				// Unknown message
				printf("Unknown message with tag %d\n", sender_status.MPI_TAG);
				assert(0);
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
			auto submit_task_sched = [&](ds const& solver, ds::TaskSched const& task_sched) {
				ds::Trace cur_trace = solver.task_sched2trace(task_sched);
				assert((int)cur_trace.trace.size() == num_traceitems);
				ds::Trace *cur_trace_ptr = (ds::Trace*)send_buf;
				cur_trace_ptr->time_usage = cur_trace.time_usage;
				cur_trace_ptr->peak_memory_usage = cur_trace.peak_memory_usage;
				cur_trace_ptr->sum_peak_memory_usage = cur_trace.sum_peak_memory_usage;
				cur_trace_ptr->fin_time_sum = cur_trace.fin_time_sum;
				memcpy(send_buf+offsetof(ds::Trace, trace), cur_trace.trace.data(), sizeof(ds::TraceItem)*num_traceitems);

				int message_id = TAG_TRACE;
				MPI_CHECK(MPI_Send(&message_id, 1, MPI_INT, 0, TAG_TRACE, MPI_COMM_WORLD));
				MPI_CHECK(MPI_Send(send_buf, trace_size, MPI_CHAR, 0, TAG_TRACE, MPI_COMM_WORLD));
			};

			auto get_cur_best_time_usage = [&]() {
				int message_id = TAG_REQUEST_BEST_TIME_USAGE;
				MPI_CHECK(MPI_Send(&message_id, 1, MPI_INT, 0, TAG_REQUEST_BEST_TIME_USAGE, MPI_COMM_WORLD));
				int best_time_usage;
				MPI_CHECK(MPI_Recv(&best_time_usage, 1, MPI_INT, 0, TAG_RESPONSE_BEST_TIME_USAGE, MPI_COMM_WORLD, &status));
				return best_time_usage;
			};

			// Request a job
			{
				int message_id = TAG_REQUEST_JOB;
				MPI_CHECK(MPI_Send(&message_id, 1, MPI_INT, 0, TAG_REQUEST_JOB, MPI_COMM_WORLD));
			}

			// See whether there are any remaining jobs
			char has_job;
			MPI_CHECK(MPI_Recv(&has_job, 1, MPI_CHAR, 0, TAG_WHETHER_HAS_JOB, MPI_COMM_WORLD, &status));
			if (!has_job) {
				printf("Worker %d finds no job, exiting...\n", rank-1);
				break;
			}

			// Get the job detail
			int cur_e2e_config_index;
			MPI_CHECK(MPI_Recv(&cur_e2e_config_index, 1, MPI_INT, 0, TAG_JOB_CONFIG, MPI_COMM_WORLD, &status));
			auto [cur_init_method, cur_e2e_config] = sim_anneal_e2e_configs[cur_e2e_config_index];
			printf("Worker %d received job with config (%d, %s). Optimizing e2e time...\n", rank-1, (int)cur_init_method, ds::fmt_sim_anneal_config(cur_e2e_config).c_str());

			// Optimize e2e time
			DualEgoSolver solver(num_nodes, model_metas);
			ds::TaskSched cur_tasksched = solver.get_init_task_sched(cur_init_method);
			cur_tasksched = solver.optimize_e2e_time(cur_e2e_config, cur_tasksched);
			int cur_time_usage = solver.task_sched2trace(cur_tasksched).time_usage;
			if (cur_time_usage > get_cur_best_time_usage()) {
				// No advantage on time usage, give up
				printf("Worker %d gets a time usage of %d, which is greater than the best one, giving up...\n", rank-1, cur_time_usage);
				continue;
			}
			submit_task_sched(solver, cur_tasksched);

			// Optimize memory usage
			for (ds::SimAnnealConfig cur_optim_config : sim_anneal_memory_configs) {
				// Check whether we have the best time usage
				int best_time_usage = get_cur_best_time_usage();
				if (cur_time_usage > best_time_usage) {
					// No advantage on time usage, give up
					printf("Worker %d gets a time usage of %d, which is smaller than the best one (%d), giving up...\n", rank-1, cur_time_usage, best_time_usage);
					break;
				}
				// Modify the random seed, in order to let all e2e configs have different random seeds
				cur_optim_config.seed += cur_e2e_config_index * 10000000;
				// Try to optimize peak memory
				ds::TaskSched memory_optimized_tasksched = solver.optimize_peak_memory(cur_optim_config, cur_tasksched);
				submit_task_sched(solver, memory_optimized_tasksched);
				ds::Trace cur_trace = solver.task_sched2trace(memory_optimized_tasksched);
				printf("Worker %d gets an answer of {%d, %d, %d, %d}\n", rank-1, cur_trace.time_usage, cur_trace.peak_memory_usage, cur_trace.sum_peak_memory_usage, cur_trace.fin_time_sum);
			}
			
			
		}

		delete[] send_buf;
	}

public:
	ParallelDualEgoSolver(
		int num_nodes,
		vector<ds::ModelMeta> const& model_metas,
		vector<pair<ds::sim_anneal_init_t, ds::SimAnnealConfig>> const& sim_anneal_e2e_configs,
		vector<ds::SimAnnealConfig> const& sim_anneal_memory_configs
		):
		num_nodes(num_nodes),
		model_metas(model_metas),
		sim_anneal_e2e_configs(sim_anneal_e2e_configs),
		sim_anneal_memory_configs(sim_anneal_memory_configs) {
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