# Compilers
CXX ?= g++-11
CXXFLAGS = -Wall -Ofast -march=native

MPICXX ?= mpicxx
MPICXXFLAGS ?= -Wall -Ofast -march=native

# Targets
TARGETS = driver parallel_driver memory_optimizer

# Dependencies
DRIVER_OBJS = driver.o
PARALLEL_DRIVER_OBJS = parallel_driver.o

# Default target
all: $(TARGETS)

# Compile driver
driver: $(DRIVER_OBJS)
	$(CXX) $(CXXFLAGS) -o driver $(DRIVER_OBJS)

driver.o: driver.cc DualEgoSolver.h
	$(CXX) $(CXXFLAGS) -c driver.cc

# Compile parallel_driver
parallel_driver: $(PARALLEL_DRIVER_OBJS)
	$(MPICXX) $(MPICXXFLAGS) -o parallel_driver $(PARALLEL_DRIVER_OBJS)

parallel_driver.o: parallel_driver.cc ParallelDualEgoSolver.h
	$(MPICXX) $(MPICXXFLAGS) -c parallel_driver.cc

memory_optimizer: memory_optimizer.cc
	$(CXX) $(CXXFLAGS) -o memory_optimizer memory_optimizer.cc -fopenmp
	
# Clean target to remove generated files
clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
