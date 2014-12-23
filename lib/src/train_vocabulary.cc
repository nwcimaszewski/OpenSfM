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
