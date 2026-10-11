# Builds and runs the CPU-simulator harness for kernel/matrix_least_squares.cpp.
#   make -C test          build test/sim_harness
#   make -C test run      run the built-in suite
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -ffp-contract=off -Wall -Wextra

sim_harness: sim_harness.cpp ../kernel/matrix_least_squares.cpp ../kernel/matrix_least_squares_layout.h ../include/simd_machine.hpp
	$(CXX) $(CXXFLAGS) -I../include -I../kernel -Istubs sim_harness.cpp -o $@

run: sim_harness
	./sim_harness

clean:
	rm -f sim_harness

.PHONY: run clean
