#include <bits/stdc++.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
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

struct Record { int idx; int age; string line; };

// Read all lines from a file (preserve original line text)
vector<string> read_all_lines(const string &path) {
    vector<string> lines;
    ifstream in(path);
    if (!in.is_open()) return lines;
    string line;
    while (std::getline(in, line)) lines.push_back(line);
    return lines;
}

// Parse records (index, age, original line) skipping malformed lines
vector<Record> parse_records(const vector<string> &lines) {
    vector<Record> records;
    for (size_t i = 0; i < lines.size(); ++i) {
        string s = lines[i];
        string t = s;
        trim(t);
        if (t.empty()) continue;
        stringstream ss(t);
        string token;
        if (!getline(ss, token, ',')) continue;
        trim(token);
        try {
            int age = stoi(token);
            records.push_back({(int)i, age, lines[i]});
        } catch (...) {
            continue;
        }
    }
    return records;
}

// Write three filtered dataset files based on the original records
// - remove the single record with the oldest age (first occurrence)
// - remove all records with age == 26
// - remove the single record with the youngest age (first occurrence)
void write_filtered_files(const string &orig_path, const string &out_dir) {
    vector<string> lines = read_all_lines(orig_path);
    if (lines.empty()) {
        cerr << "Warning: original file empty or not found: " << orig_path << "\n";
        return;
    }
    vector<Record> records = parse_records(lines);
    if (records.empty()) {
        cerr << "Warning: no parseable records in " << orig_path << "\n";
        return;
    }

    int max_age = INT_MIN, min_age = INT_MAX;
    for (auto &r : records) {
        max_age = max(max_age, r.age);
        min_age = min(min_age, r.age);
    }

    int oldest_idx = -1, youngest_idx = -1;
    for (auto &r : records) {
        if (r.age == max_age && oldest_idx == -1) oldest_idx = r.idx;
        if (r.age == min_age && youngest_idx == -1) youngest_idx = r.idx;
    }

    string out1 = out_dir + "/adult_minus_oldest.data";
    string out2 = out_dir + "/adult_minus_age26.data";
    string out3 = out_dir + "/adult_minus_youngest.data";

    // remove oldest (single index)
    ofstream f1(out1);
    for (size_t i = 0; i < lines.size(); ++i) {
        if ((int)i == oldest_idx) continue;
        f1 << lines[i] << '\n';
    }
    f1.close();

    // remove all age 26
    ofstream f2(out2);
    for (auto &r : records) {
        if (r.age == 26) continue;
        f2 << r.line << '\n';
    }
    f2.close();

    // remove youngest (single index)
    ofstream f3(out3);
    for (size_t i = 0; i < lines.size(); ++i) {
        if ((int)i == youngest_idx) continue;
        f3 << lines[i] << '\n';
    }
    f3.close();

    cout << "Wrote filtered files: " << out1 << ", " << out2 << ", " << out3 << "\n";
}

// Run analysis (compute average over age>25, add Laplace noise, write trials)
bool run_analysis_on_file(const string &input_path, const string &output_path, double epsilon, int trials, std::mt19937_64 &rng) {
    ifstream infile(input_path);
    if (!infile.is_open()) {
        cerr << "Failed to open input file: " << input_path << "\n";
        return false;
    }

    vector<int> ages_subset; // age > 25
    string line;
    while (std::getline(infile, line)) {
        string t = line;
        trim(t);
        if (t.empty()) continue;
        stringstream ss(t);
        string token;
        if (!getline(ss, token, ',')) continue;
        trim(token);
        try {
            int age = stoi(token);
            if (age > 25) ages_subset.push_back(age);
        } catch (...) { continue; }
    }
    infile.close();

    if (ages_subset.empty()) {
        cerr << "No records with age > 25 found in input: " << input_path << "\n";
        return false;
    }

    double sum = 0.0;
    int m = (int)ages_subset.size();
    int min_age = INT_MAX, max_age = INT_MIN;
    for (int a : ages_subset) {
        sum += a;
        min_age = min(min_age, a);
        max_age = max(max_age, a);
    }
    double avg = sum / m;

    double sensitivity = double(max_age - min_age) / double(max(1, m));
    double scale = sensitivity / epsilon;

    ofstream outfile(output_path);
    if (!outfile.is_open()) {
        cerr << "Failed to open output file: " << output_path << "\n";
        return false;
    }

    for (int i = 0; i < trials; ++i) {
        double noise = sample_laplace(scale, rng);
        double noisy = avg + noise;
        outfile << std::setprecision(10) << noisy << "\n";
    }
    outfile.close();

    cout << "[ANALYSIS] input=" << input_path << " m=" << m << " avg=" << avg << " sens=" << sensitivity << " b=" << scale << " -> wrote " << trials << " to " << output_path << "\n";
    return true;
}

int main(int argc, char **argv) {
    // defaults
    string input_path = "data/adult.data";
    double epsilon = 0.5;
    int trials = 1000;
    string data_dir = "data";

    // CLI parsing (supports --epsilon, --trials, --input, --data-dir)
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if ((arg == "-i" || arg == "--input") && i + 1 < argc) { input_path = argv[++i]; }
        else if ((arg == "-e" || arg == "--epsilon") && i + 1 < argc) { epsilon = stod(argv[++i]); }
        else if ((arg == "-t" || arg == "--trials") && i + 1 < argc) { trials = stoi(argv[++i]); }
        else if ((arg == "--data-dir") && i + 1 < argc) { data_dir = argv[++i]; }
        else if (arg == "-h" || arg == "--help") {
            cout << "Usage: " << argv[0] << " [--input PATH] [--data-dir DIR] [--epsilon E] [--trials N]\n";
            return 0;
        }
    }

    // Ensure output directory exists
    { struct stat st; if (stat(data_dir.c_str(), &st) != 0) {
        if (mkdir(data_dir.c_str(), 0755) != 0) {
            cerr << "Failed to create data dir: " << data_dir << "\n"; return 2;
        }
    }}

    // Generate filtered datasets (written to data_dir)
    write_filtered_files(input_path, data_dir);

    // Random engine seeded once
    std::random_device rd;
    std::mt19937_64 rng(rd());

    // Run analyses for original and the three filtered files
    string out_orig = data_dir + "/noisy_results_eps" + to_string(epsilon) + "_original.txt";
    string out_oldest = data_dir + "/noisy_results_eps" + to_string(epsilon) + "_minus_oldest.txt";
    string out_age26 = data_dir + "/noisy_results_eps" + to_string(epsilon) + "_minus_age26.txt";
    string out_youngest = data_dir + "/noisy_results_eps" + to_string(epsilon) + "_minus_youngest.txt";

    run_analysis_on_file(input_path, out_orig, epsilon, trials, rng);
    run_analysis_on_file(data_dir + "/adult_minus_oldest.data", out_oldest, epsilon, trials, rng);
    run_analysis_on_file(data_dir + "/adult_minus_age26.data", out_age26, epsilon, trials, rng);
    run_analysis_on_file(data_dir + "/adult_minus_youngest.data", out_youngest, epsilon, trials, rng);

    return 0;
}
