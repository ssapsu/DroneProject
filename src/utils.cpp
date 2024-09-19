#include "utils.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <cmath>

using torch::indexing::Slice;
using torch::indexing::None;

float generate_scale(const cv::Mat& image, const std::vector<int>& target_size) {
    int origin_w = image.cols;
    int origin_h = image.rows;

    int target_h = target_size[0];
    int target_w = target_size[1];

    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    float resize_scale = std::min(ratio_h, ratio_w);
    return resize_scale;
}

float letterbox(const cv::Mat &input_image, cv::Mat &output_image, const std::vector<int> &target_size) {
    if (input_image.cols == target_size[1] && input_image.rows == target_size[0]) {
        if (input_image.data == output_image.data) {
            return 1.;
        } else {
            output_image = input_image.clone();
            return 1.;
        }
    }

    float resize_scale = generate_scale(input_image, target_size);
    int new_shape_w = std::round(input_image.cols * resize_scale);
    int new_shape_h = std::round(input_image.rows * resize_scale);

    // 패딩을 계산할 때 음수값이 나오지 않도록 보정
    float padw = (target_size[1] - new_shape_w) / 2.0f;
    float padh = (target_size[0] - new_shape_h) / 2.0f;

    // 패딩 값이 음수로 나오지 않도록 max 함수를 사용해 0 이상으로 보정
    int top = std::max(0, static_cast<int>(std::floor(padh)));
    int bottom = std::max(0, static_cast<int>(std::ceil(padh)));
    int left = std::max(0, static_cast<int>(std::floor(padw)));
    int right = std::max(0, static_cast<int>(std::ceil(padw)));

    // 이미지 리사이즈
    cv::resize(input_image, output_image, cv::Size(new_shape_w, new_shape_h), 0, 0, cv::INTER_AREA);

    // 패딩을 추가 (만약 패딩이 전부 0일 경우, copyMakeBorder는 그대로 원본을 반환)
    cv::copyMakeBorder(output_image, output_image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114.));

    return resize_scale;
}

// non_max_suppression 함수 정의
torch::Tensor non_max_suppression(torch::Tensor& prediction, float conf_thres, float iou_thres, int max_det) {
    auto bs = prediction.size(0);
    auto nc = prediction.size(1) - 4;
    auto nm = prediction.size(1) - nc - 4;
    auto mi = 4 + nc;
    auto xc = prediction.index({torch::indexing::Slice(), torch::indexing::Slice(4, mi)}).amax(1) > conf_thres;

    prediction = prediction.transpose(-1, -2);
    prediction.index_put_({"...", torch::indexing::Slice(None, 4)}, xywh2xyxy(prediction.index({"...", torch::indexing::Slice(None, 4)})));

    std::vector<torch::Tensor> output;
    for (int i = 0; i < bs; i++) {
        output.push_back(torch::zeros({0, 6 + nm}, prediction.device()));
    }

    for (int xi = 0; xi < prediction.size(0); xi++) {
        auto x = prediction[xi];
        x = x.index({xc[xi]});
        auto x_split = x.split({4, nc, nm}, 1);
        auto box = x_split[0], cls = x_split[1], mask = x_split[2];
        auto [conf, j] = cls.max(1, true);
        x = torch::cat({box, conf, j.toType(torch::kFloat), mask}, 1);
        x = x.index({conf.view(-1) > conf_thres});
        int n = x.size(0);
        if (!n) { continue; }

        // NMS
        auto c = x.index({torch::indexing::Slice(), torch::indexing::Slice(5, 6)}) * 7680;
        auto boxes = x.index({torch::indexing::Slice(), torch::indexing::Slice(None, 4)}) + c;
        auto scores = x.index({torch::indexing::Slice(), 4});
        auto i = nms(boxes, scores, iou_thres);
        i = i.index({torch::indexing::Slice(None, max_det)});
        output[xi] = x.index({i});
    }

    return torch::stack(output);
}

// xywh2xyxy 변환 함수 정의
torch::Tensor xywh2xyxy(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    auto dw = x.index({"...", 2}).div(2);
    auto dh = x.index({"...", 3}).div(2);
    y.index_put_({"...", 0}, x.index({"...", 0}) - dw);
    y.index_put_({"...", 1}, x.index({"...", 1}) - dh);
    y.index_put_({"...", 2}, x.index({"...", 0}) + dw);
    y.index_put_({"...", 3}, x.index({"...", 1}) + dh);
    return y;
}

// nms 함수 정의
torch::Tensor nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold) {
    if (bboxes.numel() == 0)
        return torch::empty({0}, bboxes.options().dtype(torch::kLong));

    auto x1_t = bboxes.select(1, 0).contiguous();
    auto y1_t = bboxes.select(1, 1).contiguous();
    auto x2_t = bboxes.select(1, 2).contiguous();
    auto y2_t = bboxes.select(1, 3).contiguous();

    torch::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

    auto order_t = std::get<1>(scores.sort(0, true));

    torch::Tensor keep = torch::empty({bboxes.size(0)}, bboxes.options().dtype(torch::kLong));
    int64_t num_to_keep = 0;
    torch::Tensor suppressed = torch::zeros({bboxes.size(0)}, torch::kByte);

    for (int64_t i = 0; i < bboxes.size(0); ++i) {
        auto idx = order_t[i].item<int64_t>();
        if (suppressed[idx].item<uint8_t>() == 1) {
            continue;
        }
        keep[num_to_keep++] = idx;

        for (int64_t j = i + 1; j < bboxes.size(0); ++j) {
            auto idx2 = order_t[j].item<int64_t>();
            if (suppressed[idx2].item<uint8_t>() == 1) {
                continue;
            }

            auto xx1 = std::max(x1_t[idx].item<float>(), x1_t[idx2].item<float>());
            auto yy1 = std::max(y1_t[idx].item<float>(), y1_t[idx2].item<float>());
            auto xx2 = std::min(x2_t[idx].item<float>(), x2_t[idx2].item<float>());
            auto yy2 = std::min(y2_t[idx].item<float>(), y2_t[idx2].item<float>());

            auto w = std::max(0.0f, xx2 - xx1);
            auto h = std::max(0.0f, yy2 - yy1);

            auto inter = w * h;
            auto ovr = inter / (areas_t[idx].item<float>() + areas_t[idx2].item<float>() - inter);

            if (ovr > iou_threshold) {
                suppressed[idx2] = 1;
            }
        }
    }

    return keep.narrow(0, 0, num_to_keep);
}

torch::Tensor scale_boxes(const std::vector<int>& img1_shape, torch::Tensor& boxes, const std::vector<int>& img0_shape) {
    auto gain = (std::min)((float)img1_shape[0] / img0_shape[0], (float)img1_shape[1] / img0_shape[1]);
    auto pad0 = std::round((float)(img1_shape[1] - img0_shape[1] * gain) / 2. - 0.1);
    auto pad1 = std::round((float)(img1_shape[0] - img0_shape[0] * gain) / 2. - 0.1);

    boxes.index_put_({"...", 0}, boxes.index({"...", 0}) - pad0);
    boxes.index_put_({"...", 2}, boxes.index({"...", 2}) - pad0);
    boxes.index_put_({"...", 1}, boxes.index({"...", 1}) - pad1);
    boxes.index_put_({"...", 3}, boxes.index({"...", 3}) - pad1);
    boxes.index_put_({"...", Slice(None, 4)}, boxes.index({"...", Slice(None, 4)}).div(gain));
    return boxes;
}
