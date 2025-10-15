CC = g++
CFLAGS = -std=c++17 -O2
BINARY = noisy_average
SRC = source/noisy_average.cpp

.PHONY: all build run clean

all: build

build:
	$(CC) $(CFLAGS) -o $(BINARY) $(SRC)

run: build
	./$(BINARY) --data-dir data --epsilon 0.5 --trials 1000

clean:
	rm -f $(BINARY)
	rm -f data/noisy_results_eps*.txt
	rm -f data/adult_minus_*.data
