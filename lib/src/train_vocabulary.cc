#include <iostream>
#include <algorithm>

#include "opencv2/core/core.hpp"
#include "opencv2/flann/flann.hpp"


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
struct ApproximateKMeansParams {
  int max_iterations;
  float tolerance;
  int num_kdtrees;
  int kdtree_checks;
  ApproximateKMeansParams(int max_iterations,
                          float tolerance,
                          int num_kdtrees,
                          int kdtree_checks
  ) : max_iterations(max_iterations),
      tolerance(tolerance),
      num_kdtrees(num_kdtrees),
      kdtree_checks(kdtree_checks)
  {}
};

void ApproximateKMeans(const cv::Mat &points,
                       const cv::Mat &initial_centers,
                       int k,
                       const ApproximateKMeansParams &params,
                       cv::Mat *centers,
                       std::vector<int> *labels) {
  const int n = points.rows;
  const int d = points.cols;
  *centers = cv::Mat(k, d, CV_32F);
  cv::Mat old_centers = cv::Mat(k, d, CV_32F);
  labels->resize(n);

  if (initial_centers.rows == k && initial_centers.cols == d) {
    initial_centers.copyTo(*centers);
  } else {
    std::vector<int> samples;
    RandomSample(k, n, &samples);
    for (int i = 0; i < k; ++i) {
      points.row(samples[i]).copyTo(centers->row(i));
    }
  }

  float delta = params.tolerance;
  for (int iteration = 0;
       delta >= params.tolerance && iteration < params.max_iterations;
       ++iteration) {
    // Classify points
    cvflann::KDTreeIndexParams indexParams(params.num_kdtrees);
    cvflann::SearchParams searchParams(params.kdtree_checks);

    std::cerr << "Building centers' index\n";
    cv::flann::GenericIndex< cvflann::L2<float> > index(*centers, indexParams);
    cv::Mat nn(n, 1, CV_32S), dist(n, 1, CV_32F);
    int label_changes = 0;
    std::cerr << "Labeling\n";
    index.knnSearch(points, nn, dist, 1, searchParams);
    for (int i = 0; i < n; ++i) {
      int new_label = nn.at<int>(i, 0);
      if ((*labels)[i] != new_label) {
        (*labels)[i] = new_label;
        label_changes++;
      }
    }

    std::cerr << "Recomputing centers\n";

    // Recompute centers
    std::swap(*centers, old_centers);
    (*centers) = cv::Scalar(0.0f);
    std::vector<float> weights(k, 0.0f);
    for (int i = 0; i < n; ++i) {
      centers->row((*labels)[i]) += points.row(i);
      weights[(*labels)[i]] += 1;
    }
    for (int i = 0; i < k; ++i) {
      if (weights[i] != 0) {
        centers->row(i) /= weights[i];
      } // TODO(pau): deal with empty clusters.
    }
    delta = cv::norm(*centers, old_centers, cv::NORM_L2);
    std::cerr << "iteration " << iteration << std::endl;
    std::cerr << "label_changes " << label_changes << std::endl;
    std::cerr << "delta " << delta << std::endl;
  }
}
