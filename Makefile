CC = g++
CFLAGS = -std=c++17 -O2
BINARY = noisy_average
SRC = source/noisy_average.cpp

.PHONY: all build run clean pipeline run-eps0.5 run-eps1.0 validate plots

all: build

build:
	$(CC) $(CFLAGS) -o $(BINARY) $(SRC)

run: run-eps0.5

# Run noisy generation for eps=0.5
run-eps0.5: build
	./$(BINARY) --data-dir data --epsilon 0.5 --trials 1000 --global-range 100

# Run noisy generation for eps=1.0
run-eps1.0: build
	./$(BINARY) --data-dir data --epsilon 1.0 --trials 1000 --global-range 100

# Run the validation script
validate:
	python3 scripts/validate_privacy.py

# Produce plots (requires venv python or system matplotlib)
plots:
	/mnt/c/Users/seamu/Desktop/differential_privacy/.venv/bin/python scripts/plot_errors_and_mae.py

# Full pipeline: build, generate both eps, validate, plots
pipeline: build run-eps0.5 run-eps1.0 validate plots

clean:
	rm -f $(BINARY)
	rm -f data/noisy_results_eps*.txt
	rm -f data/adult_minus_*.data
	rm -rf outputs
