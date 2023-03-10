#ifndef KIWI_APP_YOLOV5_HPP
#define KIWI_APP_YOLOV5_HPP

#include "kiwi-common-box.hpp"
#include <opencv2/opencv.hpp>
#include <future>
#include <vector>
#include <memory>
#include <string>

namespace yolov5{

    class Infer{
    public:
        virtual std::shared_future<kiwi::BoxArray> commit(const cv::Mat& image) = 0;
    };

    std::shared_ptr<Infer> create_infer(
        const std::string& engine_file,
        float confidence_threshold=0.25f, float nms_threshold=0.5f
    );
}; // namespace kiwi

#endif // KIWI_APP_YOLOV5_HPP