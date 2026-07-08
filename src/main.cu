#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;

struct Options {
  std::string input_dir = "data/input";
  std::string output_dir = "output/filtered";
  int limit = 100;
  int generate = 100;
  int length = 2048;
  int cutoff = 96;
  float attenuation = 0.05f;
};

struct SignalFile {
  std::string path;
  std::vector<float> samples;
};

struct Result {
  std::string input_file;
  std::string output_file;
  int samples = 0;
  int cutoff = 0;
  float attenuation = 0.0f;
  float input_roughness = 0.0f;
  float output_roughness = 0.0f;
  float gpu_batch_ms = 0.0f;
  float gpu_ms_per_file = 0.0f;
};

__global__ void ApplyLowPassFilter(cufftComplex *values, int samples_per_signal,
                                   int total_values, int cutoff,
                                   float attenuation) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total_values) {
    return;
  }

  int bin = index % samples_per_signal;
  int mirrored_bin = samples_per_signal - bin;
  int frequency_bin = bin < mirrored_bin ? bin : mirrored_bin;
  if (frequency_bin > cutoff) {
    values[index].x *= attenuation;
    values[index].y *= attenuation;
  }
}

__global__ void NormalizeInverseFft(cufftComplex *values, int total_values,
                                    float scale) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total_values) {
    return;
  }

  values[index].x *= scale;
  values[index].y *= scale;
}

void CheckCuda(cudaError_t status, const char *message) {
  if (status != cudaSuccess) {
    std::cerr << message << ": " << cudaGetErrorString(status) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

void CheckCufft(cufftResult status, const char *message) {
  if (status != CUFFT_SUCCESS) {
    std::cerr << message << ": cuFFT status " << static_cast<int>(status)
              << "\n";
    std::exit(EXIT_FAILURE);
  }
}

void EnsureDirectory(const std::string &path) {
  if (path.empty()) {
    return;
  }

  for (size_t i = 1; i <= path.size(); ++i) {
    if (i != path.size() && path[i] != '/') {
      continue;
    }

    std::string part = path.substr(0, i);
    if (part.empty()) {
      continue;
    }

    if (mkdir(part.c_str(), 0755) != 0 && errno != EEXIST) {
      std::cerr << "Failed to create directory " << part << ": "
                << std::strerror(errno) << "\n";
      std::exit(EXIT_FAILURE);
    }
  }
}

std::string JoinPath(const std::string &left, const std::string &right) {
  if (left.empty()) {
    return right;
  }
  return left.back() == '/' ? left + right : left + "/" + right;
}

std::string BaseName(const std::string &path) {
  size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

bool EndsWithCsv(const std::string &name) {
  return name.size() >= 4 && name.substr(name.size() - 4) == ".csv";
}

std::vector<std::string> ListCsvFiles(const std::string &dir_path) {
  std::vector<std::string> files;
  DIR *dir = opendir(dir_path.c_str());
  if (dir == nullptr) {
    return files;
  }

  while (dirent *entry = readdir(dir)) {
    std::string name = entry->d_name;
    if (EndsWithCsv(name)) {
      files.push_back(JoinPath(dir_path, name));
    }
  }
  closedir(dir);
  std::sort(files.begin(), files.end());
  return files;
}

std::vector<float> ReadCsv(const std::string &path) {
  std::ifstream file(path);
  if (!file) {
    std::cerr << "Failed to read " << path << "\n";
    std::exit(EXIT_FAILURE);
  }

  std::vector<float> values;
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream row(line);
    std::string cell;
    while (std::getline(row, cell, ',')) {
      if (!cell.empty()) {
        values.push_back(std::stof(cell));
      }
    }
  }
  return values;
}

void WriteCsv(const std::string &path, const std::vector<float> &values) {
  std::ofstream file(path);
  if (!file) {
    std::cerr << "Failed to write " << path << "\n";
    std::exit(EXIT_FAILURE);
  }

  file << std::fixed << std::setprecision(6);
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      file << ",";
    }
    file << values[i];
  }
  file << "\n";
}

float Roughness(const std::vector<float> &values) {
  if (values.size() < 2) {
    return 0.0f;
  }

  double total = 0.0;
  for (size_t i = 1; i < values.size(); ++i) {
    total += std::fabs(values[i] - values[i - 1]);
  }
  return static_cast<float>(total / static_cast<double>(values.size() - 1));
}

void GenerateSignals(const std::string &input_dir, int count, int length) {
  EnsureDirectory(input_dir);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> noise(-0.18f, 0.18f);

  for (int file_index = 0; file_index < count; ++file_index) {
    std::ostringstream name;
    name << "signal_" << std::setw(4) << std::setfill('0') << file_index
         << ".csv";

    std::vector<float> values(length);
    float phase = static_cast<float>(file_index % 17) * 0.11f;
    for (int i = 0; i < length; ++i) {
      float t = static_cast<float>(i) / static_cast<float>(length);
      float slow = std::sin(2.0f * kPi * (5.0f * t + phase));
      float mid = 0.35f * std::sin(2.0f * kPi * (19.0f * t + phase));
      float high = 0.22f * std::sin(2.0f * kPi * (240.0f * t + phase));
      values[i] = slow + mid + high + noise(rng);
    }
    WriteCsv(JoinPath(input_dir, name.str()), values);
  }
}

std::vector<SignalFile> LoadSignals(const std::vector<std::string> &files,
                                    int limit) {
  std::vector<SignalFile> signals;
  int expected_size = -1;

  for (const std::string &file : files) {
    if (static_cast<int>(signals.size()) >= limit) {
      break;
    }

    std::vector<float> samples = ReadCsv(file);
    if (samples.empty()) {
      std::cerr << "Input file is empty: " << file << "\n";
      std::exit(EXIT_FAILURE);
    }
    if (expected_size < 0) {
      expected_size = static_cast<int>(samples.size());
    }
    if (static_cast<int>(samples.size()) != expected_size) {
      std::cerr << "All batched input files must have the same sample count. "
                << file << " has " << samples.size() << " samples, expected "
                << expected_size << "\n";
      std::exit(EXIT_FAILURE);
    }

    signals.push_back({file, std::move(samples)});
  }
  return signals;
}

Options ParseArgs(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };

    if (arg == "--input") {
      options.input_dir = require_value("--input");
    } else if (arg == "--output") {
      options.output_dir = require_value("--output");
    } else if (arg == "--limit") {
      options.limit = std::stoi(require_value("--limit"));
    } else if (arg == "--generate") {
      options.generate = std::stoi(require_value("--generate"));
    } else if (arg == "--length") {
      options.length = std::stoi(require_value("--length"));
    } else if (arg == "--cutoff") {
      options.cutoff = std::stoi(require_value("--cutoff"));
    } else if (arg == "--attenuation") {
      options.attenuation = std::stof(require_value("--attenuation"));
    } else if (arg == "--help") {
      std::cout << "Usage: ./batch_fft_filter --input data/input "
                   "--output output/filtered --limit 100 --generate 100 "
                   "--length 2048 --cutoff 96 --attenuation 0.05\n";
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      std::exit(EXIT_FAILURE);
    }
  }

  if (options.limit < 1 || options.length < 2 || options.generate < 0 ||
      options.cutoff < 0 || options.attenuation < 0.0f ||
      options.attenuation > 1.0f) {
    std::cerr << "Invalid arguments: limit and length must be positive, "
                 "generate and cutoff must be non-negative, and attenuation "
                 "must be between 0 and 1.\n";
    std::exit(EXIT_FAILURE);
  }
  return options;
}

std::vector<Result> ProcessBatch(const std::vector<SignalFile> &signals,
                                 const Options &options) {
  if (signals.empty()) {
    return {};
  }

  int batch = static_cast<int>(signals.size());
  int samples_per_signal = static_cast<int>(signals.front().samples.size());
  long long total_values_long =
      static_cast<long long>(batch) * static_cast<long long>(samples_per_signal);
  if (total_values_long > INT_MAX) {
    std::cerr << "Batch is too large for this demo program: " << batch
              << " signals * " << samples_per_signal << " samples\n";
    std::exit(EXIT_FAILURE);
  }
  int total_values = static_cast<int>(total_values_long);
  int cutoff = std::min(options.cutoff, samples_per_signal / 2);

  std::vector<cufftComplex> host_values(total_values);
  for (int signal_index = 0; signal_index < batch; ++signal_index) {
    for (int i = 0; i < samples_per_signal; ++i) {
      int index = signal_index * samples_per_signal + i;
      host_values[index].x = signals[signal_index].samples[i];
      host_values[index].y = 0.0f;
    }
  }

  cufftComplex *device_values = nullptr;
  size_t bytes = sizeof(cufftComplex) * host_values.size();
  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&device_values), bytes),
            "cudaMalloc device_values failed");
  CheckCuda(cudaMemcpy(device_values, host_values.data(), bytes,
                       cudaMemcpyHostToDevice),
            "cudaMemcpy host to device failed");

  cufftHandle plan;
  CheckCufft(cufftPlan1d(&plan, samples_per_signal, CUFFT_C2C, batch),
             "cufftPlan1d failed");

  cudaEvent_t start;
  cudaEvent_t stop;
  CheckCuda(cudaEventCreate(&start), "cudaEventCreate start failed");
  CheckCuda(cudaEventCreate(&stop), "cudaEventCreate stop failed");

  int threads = 256;
  int blocks = (total_values + threads - 1) / threads;
  CheckCuda(cudaEventRecord(start), "cudaEventRecord start failed");
  CheckCufft(cufftExecC2C(plan, device_values, device_values, CUFFT_FORWARD),
             "forward cufftExecC2C failed");
  ApplyLowPassFilter<<<blocks, threads>>>(device_values, samples_per_signal,
                                          total_values, cutoff,
                                          options.attenuation);
  CheckCuda(cudaGetLastError(), "ApplyLowPassFilter launch failed");
  CheckCufft(cufftExecC2C(plan, device_values, device_values, CUFFT_INVERSE),
             "inverse cufftExecC2C failed");
  NormalizeInverseFft<<<blocks, threads>>>(
      device_values, total_values, 1.0f / static_cast<float>(samples_per_signal));
  CheckCuda(cudaGetLastError(), "NormalizeInverseFft launch failed");
  CheckCuda(cudaEventRecord(stop), "cudaEventRecord stop failed");
  CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");

  float gpu_batch_ms = 0.0f;
  CheckCuda(cudaEventElapsedTime(&gpu_batch_ms, start, stop),
            "cudaEventElapsedTime failed");
  CheckCuda(cudaMemcpy(host_values.data(), device_values, bytes,
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy device to host failed");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cufftDestroy(plan);
  cudaFree(device_values);

  std::vector<Result> results;
  results.reserve(signals.size());
  for (int signal_index = 0; signal_index < batch; ++signal_index) {
    std::vector<float> output(samples_per_signal);
    for (int i = 0; i < samples_per_signal; ++i) {
      int index = signal_index * samples_per_signal + i;
      output[i] = host_values[index].x;
    }

    std::string output_path =
        JoinPath(options.output_dir, BaseName(signals[signal_index].path));
    WriteCsv(output_path, output);

    results.push_back({signals[signal_index].path,
                       output_path,
                       samples_per_signal,
                       cutoff,
                       options.attenuation,
                       Roughness(signals[signal_index].samples),
                       Roughness(output),
                       gpu_batch_ms,
                       gpu_batch_ms / static_cast<float>(batch)});
  }
  return results;
}

void WriteSummary(const std::vector<Result> &results) {
  EnsureDirectory("output");
  std::ofstream summary("output/summary.csv");
  if (!summary) {
    std::cerr << "Failed to write output/summary.csv\n";
    std::exit(EXIT_FAILURE);
  }

  summary << "input_file,output_file,samples,cutoff,attenuation,"
             "input_roughness,output_roughness,gpu_batch_ms,gpu_ms_per_file\n";
  summary << std::fixed << std::setprecision(6);
  for (const Result &result : results) {
    summary << result.input_file << "," << result.output_file << ","
            << result.samples << "," << result.cutoff << ","
            << result.attenuation << "," << result.input_roughness << ","
            << result.output_roughness << "," << result.gpu_batch_ms << ","
            << result.gpu_ms_per_file << "\n";
  }
}

}  // namespace

int main(int argc, char **argv) {
  Options options = ParseArgs(argc, argv);
  EnsureDirectory(options.input_dir);
  EnsureDirectory(options.output_dir);

  std::vector<std::string> files = ListCsvFiles(options.input_dir);
  if (files.empty() && options.generate > 0) {
    std::cout << "Generating " << options.generate << " input signals with "
              << options.length << " samples each.\n";
    GenerateSignals(options.input_dir, options.generate, options.length);
    files = ListCsvFiles(options.input_dir);
  }

  if (files.empty()) {
    std::cerr << "No input CSV files found in " << options.input_dir << "\n";
    return EXIT_FAILURE;
  }

  std::vector<SignalFile> signals = LoadSignals(files, options.limit);
  std::cout << "Loaded " << signals.size() << " signals with "
            << signals.front().samples.size() << " samples each.\n";
  std::cout << "Running batched cuFFT low-pass filter with cutoff "
            << options.cutoff << " and attenuation " << options.attenuation
            << ".\n";

  std::vector<Result> results = ProcessBatch(signals, options);
  for (const Result &result : results) {
    std::cout << "processed=" << BaseName(result.input_file)
              << " samples=" << result.samples
              << " roughness_before=" << std::fixed << std::setprecision(6)
              << result.input_roughness
              << " roughness_after=" << result.output_roughness
              << " gpu_ms_per_file=" << result.gpu_ms_per_file << "\n";
  }

  WriteSummary(results);
  std::cout << "Processed " << results.size()
            << " signal files using batched cuFFT frequency filtering.\n";
  std::cout << "Summary: output/summary.csv\n";
  std::cout << "Filtered signals: " << options.output_dir << "\n";
  return EXIT_SUCCESS;
}
