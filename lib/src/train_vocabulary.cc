#include <cmath>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <algorithm>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include "opencv2/core/core.hpp"
#include "opencv2/flann/flann.hpp"



DEFINE_string(input, "", "the training features file.");
DEFINE_string(output, "", "the computed vocabulary file");
DEFINE_int32(size, 2, "vocabulary size");


void RandomSample(int k, int n, std::vector<int> *samples) {
  std::vector<bool> mask(n, false);
  while (samples->size() < k) {
    int i = rand() % mask.size();
    if (!mask[i]) {
      mask[i] = true;
      samples->push_back(i);
    }
  }
}

void ApproximateKMeans(const cv::Mat &points,
                       int k,
                       cv::Mat *centers,
                       std::vector<int> *labels) {
  const int n = points.rows;
  *centers = cv::Mat(k, points.cols, CV_32F);
  cv::Mat old_centers = cv::Mat(k, points.cols, CV_32F);
  labels->resize(n);

  std::vector<int> samples;
  RandomSample(k, n, &samples);
  for (int i = 0; i < k; ++i) {
    centers->row(i) = points.row(samples[i]);
  }

  const float tol = 1e-3;
  float delta = tol;
  while (delta >= tol) {
    // Classify features
    cvflann::KDTreeIndexParams indexParams(4);
    cvflann::SearchParams searchParams(32);

    cv::flann::GenericIndex< cvflann::L2<float> > index(*centers, indexParams);
    cv::Mat nn(n, 1, CV_32S), dist(n, 1, CV_32F);
    index.knnSearch(points, nn, dist, 1, searchParams);
    for (int i = 0; i < n; ++i) {
      (*labels)[i] = nn.at<int>(i, 0);
    }

    // Recompute centers
    std::swap(*centers, old_centers);
    (*centers) = cv::Scalar(0.0f);
    std::vector<float> weights(k, 0.0f);
    for (int i = 0; i < n; ++i) {
      centers->row((*labels)[i]) += points.row(i);
      weights[(*labels)[i]] += 1;
    }
    for (int i = 0; i < k; ++i) {
      centers->row(i) /= weights[i];
    }
    delta = cv::norm(*centers, old_centers, cv::NORM_INF);
    std::cerr << "delta " << delta << std::endl;
  }
}

void ReadFeatures(const char *file, cv::Mat *features) {
  std::ifstream fin(file);
  std::vector<float> data;
  std::string line;
  int num_features = 0;
  while(std::getline(fin, line)) {
    std::istringstream s(line);
    std::string word;
    while (getline(s, word, ' ')) {
      if (word.size()) {
        data.push_back(std::stof(word));
      }
    }
    num_features++;
  }
  int num_dims = data.size() / num_features;
  
  *features = cv::Mat(num_features, num_dims, cv::DataType<float>::type);
  for (int i = 0; i < num_features; ++i) {
    for (int j = 0; j < num_dims; ++j) {
      (*features).at<float>(i,j) = data[i * num_dims + j];
    }
  }
}

void WriteFeatures(cv::Mat features, const char *file) {
  std::ofstream fout(file);
  for (int i = 0; i < features.rows; ++i) {
    for (int j = 0; j < features.cols; ++j) {
      fout << features.at<float>(i,j) << ((j == features.cols - 1) ? "\n" : " ");
    }
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (!FLAGS_input.size() || !FLAGS_output.size()) {
    std::cerr << "usage: simple_bundle_adjuster --input <features file> --output <vocabulary file>\n";
    return 1;
  }

  std::cerr << "Reading features\n";
  cv::Mat features, centers;
  std::vector<int> labels;
  ReadFeatures(FLAGS_input.c_str(), &features);
  std::cerr << features.rows << " " << features.cols << " features read\n";

  int k = FLAGS_size;
  std::cerr << "Compute Approximate K-Means K = " << k << "\n";
  ApproximateKMeans(features, k, &centers, &labels);

  std::cerr << "Writing centers\n";
  WriteFeatures(features, FLAGS_output.c_str());
  std::cerr << "Done\n";

  return 0;
}

