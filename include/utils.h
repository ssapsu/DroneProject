#pragma once

#include <torch/torch.h>

#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>

// generate_scale 함수 선언
float generate_scale(const cv::Mat& image, const std::vector<int>& target_size);

// letterbox 함수 선언
float letterbox(const cv::Mat &input_image, cv::Mat &output_image, const std::vector<int> &target_size);

// non_max_suppression 함수 선언
torch::Tensor non_max_suppression(torch::Tensor& prediction, float conf_thres = 0.25, float iou_thres = 0.45, int max_det = 300);

// xywh2xyxy 변환 함수 선언
torch::Tensor xywh2xyxy(const torch::Tensor& x);

// nms 함수 선언
torch::Tensor nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold);

#endif // UTILS_H
