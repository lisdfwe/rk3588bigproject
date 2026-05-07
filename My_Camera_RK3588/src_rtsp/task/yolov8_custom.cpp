#include "yolov8_custom.h"
#include <random>
#include "utils/logging.h"
#include "process/preprocess.h"
#include "process/postprocess.h"

// COCO 80类标签
static std::vector<std::string> g_classes = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

Yolov8Custom::Yolov8Custom()
{
    engine_ = CreateRKNNEngine();
    input_tensor_.data = nullptr;
    want_float_ = false; // 是否使用浮点数版本的后处理
    ready_ = false;
}

Yolov8Custom::~Yolov8Custom()
{
    // release input tensor and output tensor
    NN_LOG_DEBUG("release input tensor");
    if (input_tensor_.data != nullptr)
    {
        free(input_tensor_.data);
        input_tensor_.data = nullptr;
    }
    NN_LOG_DEBUG("release output tensor");
    for (auto &tensor : output_tensors_)
    {
        if (tensor.data != nullptr)
        {
            free(tensor.data);
            tensor.data = nullptr;
        }
    }
}

nn_error_e Yolov8Custom::LoadModel(const char *model_path)
{
    auto ret = engine_->LoadModelFile(model_path);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("yolov8 load model file failed");
        return ret;
    }
    // get input tensor
    auto input_shapes = engine_->GetInputShapes();

    // check number of input and n_dims
    if (input_shapes.size() != 1)
    {
        NN_LOG_ERROR("yolov8 input tensor number is not 1, but %ld", input_shapes.size());
        return NN_RKNN_INPUT_ATTR_ERROR;
    }
    nn_tensor_attr_to_cvimg_input_data(input_shapes[0], input_tensor_);
    input_tensor_.data = malloc(input_tensor_.attr.size);

    auto output_shapes = engine_->GetOutputShapes();
    
    // 支持6输出（旧格式）和9输出（新格式，包含class_sum）
    if (output_shapes.size() != 6 && output_shapes.size() != 9)
    {
        NN_LOG_ERROR("yolov8 output tensor number is not 6 or 9, but %ld", output_shapes.size());
        return NN_RKNN_OUTPUT_ATTR_ERROR;
    }
    
    // 记录输出数量，用于后处理
    output_num_ = output_shapes.size();
    NN_LOG_INFO("yolov8 output tensor number: %d", output_num_);
    
    if (output_shapes[0].type == NN_TENSOR_FLOAT16)
    {
        want_float_ = true;
        NN_LOG_WARNING("yolov8 output tensor type is float16, want type set to float32");
    }
    for (int i = 0; i < output_shapes.size(); i++)
    {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims = output_shapes[i].n_dims;
        for (int j = 0; j < output_shapes[i].n_dims; j++)
        {
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        }
        // output tensor needs to be float32
        tensor.attr.type = want_float_ ? NN_TENSOR_FLOAT : output_shapes[i].type;
        tensor.attr.index = 0;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        output_tensors_.push_back(tensor);
        out_zps_.push_back(output_shapes[i].zp);
        out_scales_.push_back(output_shapes[i].scale);
    }

    ready_ = true;
    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox)
{

    // 预处理包含：letterbox、归一化、BGR2RGB、NCWH
    // 其中RKNN会做：归一化、NCWH转换（详见课程文档），所以这里只需要做letterbox、BGR2RGB

    // 比例
    float wh_ratio = (float)input_tensor_.attr.dims[2] / (float)input_tensor_.attr.dims[1];

    // lettorbox

    if (process_type == "opencv")
    {
        // BGR2RGB，resize，再放入input_tensor_中
        letterbox_info_ = letterbox(img, image_letterbox, wh_ratio);
        cvimg2tensor(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
    }
    else if (process_type == "rga")
    {
        // rga resize
        letterbox_info_ = letterbox_rga(img, image_letterbox, wh_ratio);
        // save img
        // cv::imwrite("rga.jpg", image_letterbox);
        cvimg2tensor_rga(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
    }

    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::Inference()
{
    std::vector<tensor_data_s> inputs;
    inputs.push_back(input_tensor_);
    return engine_->Run(inputs, output_tensors_, want_float_);
}

nn_error_e Yolov8Custom::Postprocess(const cv::Mat &img, std::vector<Detection> &objects)
{
    std::vector<float> DetectiontRects;

    if (output_num_ == 6)
    {
        void *output_data[6];
        for (int i = 0; i < 6; i++)
            output_data[i] = (void *)output_tensors_[i].data;

        if (want_float_)
            yolo::GetConvDetectionResult((float **)output_data, DetectiontRects);
        else
            yolo::GetConvDetectionResultInt8((int8_t **)output_data, out_zps_, out_scales_, DetectiontRects);
    }
    else if (output_num_ == 9)
    {
        // 跳过 class_sum（索引2,5,8），映射: 0,1, 3,4, 6,7
        void *output_data[6];
        std::vector<int>   out_zps_6;
        std::vector<float> out_scales_6;

        int indices[] = {0, 1, 3, 4, 6, 7};
        for (int i = 0; i < 6; i++)
        {
            output_data[i] = (void *)output_tensors_[indices[i]].data;
            out_zps_6.push_back(out_zps_[indices[i]]);
            out_scales_6.push_back(out_scales_[indices[i]]);
        }

        if (want_float_)
            yolo::GetConvDetectionResult((float **)output_data, DetectiontRects);
        else
            yolo::GetConvDetectionResultInt8_9Out((int8_t **)output_data, out_zps_6, out_scales_6, DetectiontRects);
    }

    int img_width  = img.cols;
    int img_height = img.rows;
    const float confidence_threshold = 0.25f;

    for (int i = 0; i < (int)DetectiontRects.size(); i += 6)
    {
        int   classId = (int)DetectiontRects[i + 0];
        float conf    = DetectiontRects[i + 1];

        if (conf < confidence_threshold) continue;
        if (classId < 0 || classId >= (int)g_classes.size()) continue;

        int xmin = int(DetectiontRects[i + 2] * float(img_width)  + 0.5f);
        int ymin = int(DetectiontRects[i + 3] * float(img_height) + 0.5f);
        int xmax = int(DetectiontRects[i + 4] * float(img_width)  + 0.5f);
        int ymax = int(DetectiontRects[i + 5] * float(img_height) + 0.5f);

        xmin = std::max(0, xmin);
        ymin = std::max(0, ymin);
        xmax = std::min(img_width  - 1, xmax);
        ymax = std::min(img_height - 1, ymax);
        if (xmax <= xmin || ymax <= ymin) continue;

        Detection result;
        result.class_id   = classId;
        result.confidence = conf;
        result.className  = g_classes[classId];
        result.box        = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);

        std::mt19937 gen(classId * 1234);
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen), dis(gen), dis(gen));

        objects.push_back(result);
    }

    return NN_SUCCESS;
}

void letterbox_decode(std::vector<Detection> &objects, bool hor, int pad)
{
    for (auto &obj : objects)
    {
        if (hor)
        {
            obj.box.x -= pad;
        }
        else
        {
            obj.box.y -= pad;
        }
    }
}

nn_error_e Yolov8Custom::Run(const cv::Mat &img, std::vector<Detection> &objects)
{

    // letterbox后的图像
    cv::Mat image_letterbox;
    // 预处理，支持opencv或rga
    Preprocess(img, "opencv", image_letterbox);
    // Preprocess(img, "rga", image_letterbox);
    // 推理
    Inference();
    // 后处理
    Postprocess(image_letterbox, objects);

    letterbox_decode(objects, letterbox_info_.hor, letterbox_info_.pad);

    return NN_SUCCESS;
}