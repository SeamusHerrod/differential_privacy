Project: Differential Privacy - HW2

This small C++ program computes the average age of records with age > 25
from the Adult dataset (`data/adult.data`) and applies the Laplace mechanism
to produce differentially private noisy averages.

Files
- `source/noisy_average.cpp`: C++ program. Reads `data/adult.data` by default.

Usage
1. Build (from repository root):

   g++ -std=c++17 -O2 -o noisy_average source/noisy_average.cpp

2. Run (defaults produce 1000 noisy outputs for epsilon=0.5):

   ./noisy_average

Options:
- `--input` / `-i` : input file path (default `data/adult.data`)
- `--output` / `-o`: output file (default `data/noisy_results_eps0.5.txt`)
- `--epsilon` / `-e`: privacy parameter epsilon (default `0.5`)
- `--trials` / `-t`: number of noisy trials to generate (default `1000`)

Sensitivity assumption
----------------------
The code uses the Laplace mechanism for the query "average age of records with age > 25".
To compute sensitivity, the implementation uses an observed bound: sensitivity = (max_age - min_age)/m,
where `m` is the number of records in the subset (age>25), and `min_age`/`max_age` are the min/max ages
observed in that subset. This is a practical (data-driven) sensitivity bound; if a strict global bound is
required, replace `max_age - min_age` with the known global range (e.g., 100 - 0 = 100 for ages).

Notes
- The implementation assumes the first column in `adult.data` is the age field (an integer),
  consistent with typical formatting of the UCI Adult dataset.
- The program skips malformed lines.
