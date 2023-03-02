#include <onnxruntime_cxx_api.h>
#include "kiwi-infer-onnxruntime.hpp"
#include "kiwi-logger.hpp"
#include <algorithm>
#include <fstream>
#include<iostream>


namespace onnxruntime{
    using namespace std;
    using namespace kiwi;
    using namespace Ort;

    class OrtModel{
        public:


            Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "ONNX");
            Ort::Session *ort_session = nullptr;
            Ort::SessionOptions sessionOptions = Ort::SessionOptions();
            vector<char*> input_names_;
            vector<char*> output_names_;
            vector<vector<int64_t>> input_node_dims_; // >=1 outputs
            vector<vector<int64_t>> output_node_dims_; // >=1 outputs
            vector<Ort::Value> ort_inputs_;
            vector<Ort::Value> ort_outputs_;
            size_t input_nums_;
            size_t output_nums_;
            size_t numInputNodes_;
            size_t numOutputNodes_;
            ONNXTensorElementDataType input_type_;
            ONNXTensorElementDataType output_type_;
            bool load_model(const string& file){
                try{
                    sessionOptions.SetInterOpNumThreads(1);
                    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                    printf("%s",file.c_str());
                    ort_session = new Session(env, file.c_str(), sessionOptions);
                    numInputNodes_ = ort_session->GetInputCount();
                    numOutputNodes_ = ort_session->GetOutputCount();
                    AllocatorWithDefaultOptions allocator;
                    for (int i = 0; i < numInputNodes_; i++)
                    {
                        input_names_.push_back(ort_session->GetInputNameAllocated(i, allocator).release());
                        Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
                        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
                        auto input_dims = input_tensor_info.GetShape();
                        input_type_ = input_tensor_info.GetElementType();
                        input_node_dims_.push_back(input_dims);
                    }
                    for (int i = 0; i < numOutputNodes_; i++)
                    {
                        output_names_.push_back(ort_session->GetOutputNameAllocated(i, allocator).release());
                        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
                        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
                        auto output_dims = output_tensor_info.GetShape();
                        output_type_= output_tensor_info.GetElementType();
                        output_node_dims_.push_back(output_dims);
                    }

                    return true;
                }catch(Ort::Exception ex){
                    INFOE("Load onnx model %s failed: %s", file.c_str(), ex.what());
                }
                return false;
            }



        };
        class InferImpl : public Infer {

        public:
            virtual ~InferImpl();
            virtual bool load(const std::string& file);
            virtual void destroy();
            virtual bool forward() override;
            virtual std::shared_ptr<MixMemory> get_workspace() override;
            virtual std::shared_ptr<Tensor> input(int index = 0) override;
            virtual std::string get_input_name(int index = 0) override;
            virtual std::shared_ptr<Tensor> output(int index = 0) override;
            virtual std::string get_output_name(int index = 0) override;
            virtual std::shared_ptr<Tensor> tensor(const std::string& name) override;
            virtual bool is_output_name(const std::string& name) override;
            virtual bool is_input_name(const std::string& name) override;
            virtual void set_input (int index, std::shared_ptr<Tensor> tensor) override;
            virtual void set_output(int index, std::shared_ptr<Tensor> tensor) override;

            virtual void print() override;

            virtual int num_output() override;
            virtual int num_input() override;

        private:
            bool build_engine_input_and_outputs_mapper();

        private:
            std::vector<std::shared_ptr<Tensor>> inputs_;
            std::vector<std::shared_ptr<Tensor>> outputs_;
            std::vector<int> inputs_map_to_ordered_index_;
            std::vector<int> outputs_map_to_ordered_index_;
            std::vector<std::string> inputs_name_;
            std::vector<std::string> outputs_name_;
            std::vector<std::shared_ptr<Tensor>> orderdBlobs_;
            std::map<std::string, int> blobsNameMapper_;
            std::shared_ptr<MixMemory> workspace_;
            std::shared_ptr<OrtModel> ortmodel_;
        };




        InferImpl::~InferImpl(){
            destroy();
        }

    void InferImpl::destroy() {

        this->blobsNameMapper_.clear();
        this->outputs_.clear();
        this->inputs_.clear();
        this->inputs_name_.clear();
        this->outputs_name_.clear();
        this->ortmodel_.reset();
    }


    void InferImpl::print(){
        INFO("Infer %p detail", this);
        INFO("\tInputs: %d", inputs_.size());
        for(int i = 0; i < inputs_.size(); ++i){
            auto& tensor = inputs_[i];
            auto& name = inputs_name_[i];
            INFO("\t\t%d.%s : shape {%s}, %s", i, name.c_str(), tensor->shape_string(), data_type_string(tensor->type()));
        }

        INFO("\tOutputs: %d", outputs_.size());
        for(int i = 0; i < outputs_.size(); ++i){
            auto& tensor = outputs_[i];
            auto& name = outputs_name_[i];
            INFO("\t\t%d.%s : shape {%s}, %s", i, name.c_str(), tensor->shape_string(), data_type_string(tensor->type()));
        }
    }




    bool InferImpl::load(const std::string& file) {

        this->ortmodel_.reset(new OrtModel());
        if(!this->ortmodel_->load_model(file))
            return false;

        workspace_.reset(new MixMemory());
        return build_engine_input_and_outputs_mapper();
    }


    template<typename _T>
    vector<int> convert_shape(const vector<_T>& shape){
        vector<int> new_shape;
        for(auto i : shape)
            new_shape.push_back(i);
        return new_shape;
    }
    //onnxruntime Float16 is not supported
    static kiwi::DataType convert_trt_datatype(ONNXTensorElementDataType dt){
        switch(dt){               
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return kiwi::DataType::Float;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return kiwi::DataType::Int32;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return kiwi::DataType::UInt8;
            default:
                INFOE("Unsupport data type %d", dt);
                return kiwi::DataType::Float;
        }
    }


    bool InferImpl::build_engine_input_and_outputs_mapper() {

        inputs_.clear();
        inputs_name_.clear();
        outputs_.clear();
        outputs_name_.clear();
        orderdBlobs_.clear();
        blobsNameMapper_.clear();
        auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        for(int i = 0; i < ortmodel_->numInputNodes_; ++i){


            auto bindingName = ortmodel_->input_names_[i];
            auto shape = convert_shape(ortmodel_->input_node_dims_[i]);
            auto newTensor = make_shared<Tensor>(shape, convert_trt_datatype(ortmodel_->input_type_));
            newTensor->set_workspace(this->workspace_);
            inputs_.push_back(newTensor);
            inputs_name_.push_back(bindingName);
            inputs_map_to_ordered_index_.push_back(orderdBlobs_.size());
            blobsNameMapper_[bindingName] = i;
            orderdBlobs_.push_back(newTensor);

            auto input_tensor= Value::CreateTensor(allocator_info,newTensor->cpu(),newTensor->bytes(),ortmodel_->input_node_dims_[i].data(), ortmodel_->input_node_dims_[i].size(),ortmodel_->input_type_);
            ortmodel_->ort_inputs_.push_back(std::move(input_tensor));

        }

        for(int i = 0; i < ortmodel_->numOutputNodes_; ++i){

            auto bindingName = ortmodel_->output_names_[i];
            auto shape = convert_shape(ortmodel_->output_node_dims_[i]);
            auto newTensor = make_shared<Tensor>(shape, convert_trt_datatype(ortmodel_->output_type_));
            newTensor->set_workspace(this->workspace_);
            outputs_.push_back(newTensor);
            outputs_name_.push_back(bindingName);
            outputs_map_to_ordered_index_.push_back(orderdBlobs_.size());
            blobsNameMapper_[bindingName] = i;
            orderdBlobs_.push_back(newTensor);
 
            auto output_tensor= Value::CreateTensor(allocator_info,newTensor->cpu(),newTensor->bytes(),ortmodel_->output_node_dims_[i].data(), ortmodel_->output_node_dims_[i].size(),ortmodel_->output_type_);
            ortmodel_->ort_outputs_.push_back(std::move(output_tensor));
        }
        return true;
    }



    bool InferImpl::is_output_name(const std::string& name){
        return std::find(outputs_name_.begin(), outputs_name_.end(), name) != outputs_name_.end();
    }

    bool InferImpl::is_input_name(const std::string& name){
        return std::find(inputs_name_.begin(), inputs_name_.end(), name) != inputs_name_.end();
    }




    bool InferImpl::forward() {
       ortmodel_-> ort_session->Run(RunOptions{ nullptr },ortmodel_->input_names_.data(),ortmodel_->ort_inputs_.data(),ortmodel_->input_names_.size(),ortmodel_->output_names_.data(),ortmodel_->ort_outputs_.data(),ortmodel_->output_names_.size());
        return true;
    }

    std::shared_ptr<MixMemory> InferImpl::get_workspace() {
        return workspace_;
    }

    int InferImpl::num_input() {
        return static_cast<int>(this->inputs_.size());
    }

    int InferImpl::num_output() {
        return static_cast<int>(this->outputs_.size());
    }

    void InferImpl::set_input (int index, std::shared_ptr<Tensor> tensor){

        if(index < 0 || index >= inputs_.size()){
            INFOF("Input index[%d] out of range [size=%d]", index, inputs_.size());
        }

        this->inputs_[index] = tensor;
        int order_index = inputs_map_to_ordered_index_[index];
        this->orderdBlobs_[order_index] = tensor;
    }

    void InferImpl::set_output(int index, std::shared_ptr<Tensor> tensor){

        if(index < 0 || index >= outputs_.size()){
            INFOF("Output index[%d] out of range [size=%d]", index, outputs_.size());
        }

        this->outputs_[index] = tensor;
        int order_index = outputs_map_to_ordered_index_[index];
        this->orderdBlobs_[order_index] = tensor;
    }


    std::shared_ptr<Tensor> InferImpl::input(int index) {
        if(index < 0 || index >= inputs_.size()){
            INFOF("Input index[%d] out of range [size=%d]", index, inputs_.size());
        }
        return this->inputs_[index];
    }

    std::string InferImpl::get_input_name(int index){
        if(index < 0 || index >= inputs_name_.size()){
            INFOF("Input index[%d] out of range [size=%d]", index, inputs_name_.size());
        }
        return inputs_name_[index];
    }

    std::shared_ptr<Tensor> InferImpl::output(int index) {
        if(index < 0 || index >= outputs_.size()){
            INFOF("Output index[%d] out of range [size=%d]", index, outputs_.size());
        }
        return outputs_[index];
    }

    std::string InferImpl::get_output_name(int index){
        if(index < 0 || index >= outputs_name_.size()){
            INFOF("Output index[%d] out of range [size=%d]", index, outputs_name_.size());
        }
        return outputs_name_[index];
    }

    std::shared_ptr<Tensor> InferImpl::tensor(const std::string& name) {

        auto node = this->blobsNameMapper_.find(name);
        if(node == this->blobsNameMapper_.end()){
            INFOF("Could not found the input/output node '%s', please makesure your model", name.c_str());
        }
        return orderdBlobs_[node->second];
    }


    std::shared_ptr<Infer> load_infer(const string& file) {

        std::shared_ptr<InferImpl> infer(new InferImpl());
        if (!infer->load(file))
            infer.reset();
        return infer;
    }

};//onnxruntime