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


Implemented algorithm (step-by-step)

1) Data parsing and filtering
   - The program reads the input file line-by-line. Each non-empty line is expected to be a comma-separated record where the first field is the integer `age` (this matches the UCI Adult dataset layout).
   - For each line the code trims whitespace, parses the first comma-separated token as an integer age, and ignores malformed lines.
   - The program collects all ages and a subset containing only records with `age > 25`.

2) Deterministic query computation
   - Let m be the number of records with age > 25.
   - Compute the true (non-private) average age on that subset:
     avg = (1/m) * sum_{i=1..m} age_i
   - Also compute the observed minimum and maximum age in the subset (min_age and max_age). These values are used to bound the possible contribution of any single record.

3) Sensitivity choice
   - For the average query over a fixed-size subset of m records, a natural bound on how much the average can change when one individual's value changes is (range)/m, where `range` is an upper bound on possible age values in the subset.
   - The program uses `range = max_age - min_age` observed in the subset, therefore sensitivity = (max_age - min_age) / m.
   - This is a data-driven (empirical) sensitivity. If you need a data-independent (global) sensitivity, replace `max_age - min_age` with a global bound such as 100 (or another agreed range) to obtain sensitivity = global_range / m.

4) Laplace mechanism (adding noise)
   - To achieve epsilon-differential privacy for the numeric average, the Laplace mechanism adds noise drawn from Laplace(0, b), where the scale parameter b = sensitivity / epsilon.
   - The Laplace probability density function with scale b is p(x) = (1/(2b)) exp(-|x|/b).
   - The implementation samples Laplace noise via inverse CDF using a uniform random draw u ~ Uniform(0,1) and the transformation:
       noise = b * log(2u)          if u < 0.5
       noise = -b * log(2(1-u))     otherwise
   - The noisy output is noisy_avg = avg + noise.

5) Repeated trials
   - The program supports generating `trials` independent noisy releases (default 1000) by re-sampling Laplace noise each time and writing each noisy average to the output file on its own line.

Design notes and edge cases
   - Subset size m = 0: the program detects and exits with an error if there are no records with age > 25.
   - Malformed lines are skipped; if the first token cannot be parsed as an integer, the line is ignored.
   - Randomness: the program uses `std::random_device` to seed a 64-bit Mersenne Twister (`std::mt19937_64`). This yields independent Laplace draws per trial.
   - Floating point precision: outputs are printed with 10 digits of precision.

Privacy remark
   - The program reports noisy averages computed using the Laplace mechanism with the chosen sensitivity and epsilon. The current sensitivity is empirical (depends on dataset). For strict differential privacy guarantees that hold independently of the dataset, use a known global bound on ages when computing sensitivity.

Mathematical contract (inputs/outputs)
   - Input: CSV-like dataset where the first column is integer age; parameters epsilon > 0 and trials (positive integer).
   - Output: `trials` lines containing independent noisy estimates of the true average age over records with age > 25.
   - Error modes: non-existent input file, empty subset (m=0), or inability to open the output file.


============================================================
Binning into 5 bins over range [42.7300, 42.8400]

D: N=1000 min=42.73 max=42.82
D1: N=1000 min=42.73 max=42.83
D2: N=1000 min=42.74 max=42.84
D3: N=1000 min=42.74 max=42.84
Dp_combined: N=3000

D vs D1: max ratio = 1.1196  -> satisfies eps=0.5? True
D vs D2: max ratio = 1.1548  -> satisfies eps=0.5? True
D vs D3: max ratio = 1.1333  -> satisfies eps=0.5? True
D vs Dp_combined: max ratio = 1.0295 -> satisfies eps=0.5? True

Top bins for D vs D1 (ratio, bin_idx, [lo,hi], p_D, p_D1):
  1.1195652173913042    2  [42.7740,42.7960]  pD=0.7210 p_D1=0.6440
  1.0         4  [42.8180,42.8400]  pD=0.0030 p_D1=0.0030
  0.9444444444444445    0  [42.7300,42.7520]  pD=0.0170 p_D1=0.0180
  0.9090909090909092    3  [42.7960,42.8180]  pD=0.0800 p_D1=0.0880
  0.7246963562753036    1  [42.7520,42.7740]  pD=0.1790 p_D1=0.2470

Top bins for D vs D2 (ratio, bin_idx, [lo,hi], p_D, p_D2):
  1.1548387096774193    1  [42.7520,42.7740]  pD=0.1790 p_D2=0.1550
  1.0126582278481013    3  [42.7960,42.8180]  pD=0.0800 p_D2=0.0790
  0.9743243243243243    2  [42.7740,42.7960]  pD=0.7210 p_D2=0.7400
  0.8500000000000001    0  [42.7300,42.7520]  pD=0.0170 p_D2=0.0200
  0.5         4  [42.8180,42.8400]  pD=0.0030 p_D2=0.0060

Top bins for D vs D3 (ratio, bin_idx, [lo,hi], p_D, p_D3):
  1.1333333333333335    0  [42.7300,42.7520]  pD=0.0170 p_D3=0.0150
  1.0666666666666667    3  [42.7960,42.8180]  pD=0.0800 p_D3=0.0750
  1.00557880055788    2  [42.7740,42.7960]  pD=0.7210 p_D3=0.7170
  0.988950276243094    1  [42.7520,42.7740]  pD=0.1790 p_D3=0.1810
  0.25        4  [42.8180,42.8400]  pD=0.0030 p_D3=0.0120

Top bins for D vs Dp_combined (ratio, bin_idx, [lo,hi], p_D, p_Dp_combined):
  1.0295097572584482    2  [42.7740,42.7960]  pD=0.7210 p_Dp_combined=0.7003
  0.9917355371900827    3  [42.7960,42.8180]  pD=0.0800 p_Dp_combined=0.0807
  0.9622641509433962    0  [42.7300,42.7520]  pD=0.0170 p_Dp_combined=0.0177
  0.9210977701543739    1  [42.7520,42.7740]  pD=0.1790 p_Dp_combined=0.1943
  0.42857142857142855    4  [42.8180,42.8400]  pD=0.0030 p_Dp_combined=0.0070

