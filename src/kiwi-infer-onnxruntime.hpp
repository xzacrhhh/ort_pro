#ifndef KIWI_INFER_ONNXRUNTIME_HPP
#define KIWI_INFER_ONNXRUNTIME_HPP

#include "kiwi-infer.hpp"

namespace onnxruntime{

    std::shared_ptr<kiwi::Infer> load_infer(const std::string& file);

}; // namespace onnxruntime

#endif // KIWI_INFER_ONNXRUNTIME_HPP