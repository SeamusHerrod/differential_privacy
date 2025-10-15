#include <bits/stdc++.h>
using namespace std;

// Trim helpers
static inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    return s;
}
static inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
    return s;
}
static inline std::string &trim(std::string &s) { return ltrim(rtrim(s)); }

// Laplace noise generator using inverse CDF
double sample_laplace(double scale, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    double u = unif(rng);
    if (u == 0.5) return 0.0;
    if (u < 0.5) {
        return scale * std::log(2.0 * u);
    } else {
        return -scale * std::log(2.0 * (1.0 - u));
    }
}

int main(int argc, char **argv) {
    // defaults
    string input_path = "data/adult.data";
    string output_path = "data/noisy_results_eps0.5.txt";
    double epsilon = 0.5;
    int trials = 1000;

    // simple CLI parsing
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if ((arg == "-i" || arg == "--input") && i + 1 < argc) { input_path = argv[++i]; }
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc) { output_path = argv[++i]; }
        else if ((arg == "-e" || arg == "--epsilon") && i + 1 < argc) { epsilon = stod(argv[++i]); }
        else if ((arg == "-t" || arg == "--trials") && i + 1 < argc) { trials = stoi(argv[++i]); }
        else if (arg == "-h" || arg == "--help") {
            cout << "Usage: " << argv[0] << " [--input PATH] [--output PATH] [--epsilon E] [--trials N]\n";
            return 0;
        }
    }

    // Read dataset
    ifstream infile(input_path);
    if (!infile.is_open()) {
        cerr << "Failed to open input file: " << input_path << "\n";
        return 2;
    }

    vector<int> ages_all;
    vector<int> ages_subset; // age > 25
    string line;
    while (std::getline(infile, line)) {
        trim(line);
        if (line.empty()) continue;
        // split line by comma
        stringstream ss(line);
        string token;
        if (!std::getline(ss, token, ',')) continue;
        trim(token);
        try {
            int age = stoi(token);
            ages_all.push_back(age);
            if (age > 25) ages_subset.push_back(age);
        } catch (...) {
            // skip malformed
            continue;
        }
    }
    infile.close();

    if (ages_subset.empty()) {
        cerr << "No records with age > 25 found in input.\n";
        return 3;
    }

    // compute average on subset
    double sum = 0.0;
    int m = (int)ages_subset.size();
    int min_age = INT_MAX, max_age = INT_MIN;
    for (int a : ages_subset) {
        sum += a;
        min_age = min(min_age, a);
        max_age = max(max_age, a);
    }
    double avg = sum / m;

    // We assume ages are bounded in [min_age, max_age] observed in the subset.
    // Using sensitivity for the average over this fixed subset: (max_age - min_age)/m
    double sensitivity = double(max_age - min_age) / double(max(1, m));
    double scale = sensitivity / epsilon; // Laplace scale parameter b

    // random engine
    std::random_device rd;
    std::mt19937_64 rng(rd());

    ofstream outfile(output_path);
    if (!outfile.is_open()) {
        cerr << "Failed to open output file: " << output_path << "\n";
        return 4;
    }

    // generate trials noisy outputs
    for (int i = 0; i < trials; ++i) {
        double noise = sample_laplace(scale, rng);
        double noisy = avg + noise;
        outfile << std::setprecision(10) << noisy << "\n";
    }

    outfile.close();

    cout << "Wrote " << trials << " noisy results to " << output_path << "\n";
    cout << "Original average (age>25) = " << avg << " over m=" << m << " records.\n";
    cout << "Used sensitivity=(max-min)/m = " << sensitivity << ", scale=b=" << scale << " (epsilon=" << epsilon << ")\n";

    return 0;
}
