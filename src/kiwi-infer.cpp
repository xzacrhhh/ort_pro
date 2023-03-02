#include "kiwi-infer.hpp"
#include "kiwi-infer-onnxruntime.hpp"
#include "kiwi-logger.hpp"

namespace kiwi{

    static Backend backend_ = Backend::None;
    Backend set_backend(Backend backend){
        auto old = backend_;
        backend_ = backend;
        return old;
    }

    Backend get_backend(){
        return backend_;
    }

    std::shared_ptr<kiwi::Infer> load_infer_with_backend(Backend backend, const std::string& file){
        
        // if(backend == Backend::RKNN){
        //     return rknn::load_infer(file);
        // }else if(backend==Backend::OnnxRuntime){
        //     return onnxruntime::load_infer(file);
        // }
        // else if(backend == Backend::OpenVINO){
        //     return openvino::load_infer(file);
        // }else{
        //     INFOE("Unknow backend: %d", backend);
        // }

        return onnxruntime::load_infer(file);
    }

    std::shared_ptr<kiwi::Infer> load_infer(const std::string& file){
        return load_infer_with_backend(backend_, file);
    }
};