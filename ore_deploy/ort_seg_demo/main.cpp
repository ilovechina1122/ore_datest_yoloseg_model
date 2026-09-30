#include <iostream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>

namespace fs = std::filesystem;

struct BoxInfo
{
    float x1, y1, x2, y2;
    float score;
    int cls_id;
    std::vector<float> mask_coeff;
};

cv::Mat letterbox(const cv::Mat& src, int target_size,
                  float& scale, int& pad_h, int& pad_w)
{
    int h = src.rows;
    int w = src.cols;
    scale = std::min(float(target_size)/h, float(target_size)/w);

    int new_h = int(h * scale);
    int new_w = int(w * scale);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    pad_h = (target_size - new_h) / 2;
    pad_w = (target_size - new_w) / 2;

    cv::Mat dst = cv::Mat::zeros(target_size, target_size, src.type());
    dst.setTo(cv::Scalar(114,114,114));
    resized.copyTo(dst(cv::Rect(pad_w, pad_h, new_w, new_h)));
    return dst;
}

float compute_iou(const BoxInfo& a, const BoxInfo& b)
{
    float ax1 = a.x1, ay1 = a.y1, ax2 = a.x2, ay2 = a.y2;
    float bx1 = b.x1, by1 = b.y1, bx2 = b.x2, by2 = b.y2;

    float inter_x1 = std::max(ax1, bx1);
    float inter_y1 = std::max(ay1, by1);
    float inter_x2 = std::min(ax2, bx2);
    float inter_y2 = std::min(ay2, by2);
    if(inter_x1 >= inter_x2 || inter_y1 >= inter_y2) return 0.f;

    float inter_area = (inter_x2-inter_x1)*(inter_y2-inter_y1);
    float area_a = (ax2-ax1)*(ay2-ay1);
    float area_b = (bx2-bx1)*(by2-by1);
    return inter_area / (area_a + area_b - inter_area);
}

std::vector<BoxInfo> nms_per_class(std::vector<BoxInfo> boxes, float iou_thr)
{
    std::vector<BoxInfo> res;
    if(boxes.empty()) return res;

    std::sort(boxes.begin(), boxes.end(), [](const BoxInfo& a, const BoxInfo& b){
        return a.score > b.score;
    });

    std::vector<bool> suppress(boxes.size(), false);
    for(size_t i = 0; i < boxes.size(); ++i)
    {
        if(suppress[i]) continue;
        res.push_back(boxes[i]);
        for(size_t j = i+1; j < boxes.size(); ++j)
        {
            if(suppress[j]) continue;
            if(boxes[i].cls_id == boxes[j].cls_id)
            {
                float iou = compute_iou(boxes[i], boxes[j]);
                if(iou > iou_thr)
                    suppress[j] = true;
            }
        }
    }
    return res;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

int main()
{
    const float CONF_THRES = 0.25f;
    const float IOU_THRES  = 0.45f;
    const int MODEL_INPUT_SIZE = 640;
    const int NUM_CLASSES = 5;
    const int MASK_COEFF_DIM = 32;
    const int MASK_PROTO_H = 160;
    const int MASK_PROTO_W = 160;

    std::string model_path = "/home/sunweining/ore_deploy/best.onnx";
    std::string img_dir = "/home/sunweining/ore_deploy/test_img";
    std::string out_dir = "/home/sunweining/ore_deploy/ort_out";

    // 创建输出文件夹
    if(!fs::exists(out_dir)){
        fs::create_directories(out_dir);
    }

    // 全局初始化session，模型只加载一次，不要循环内重复加载！
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "seg_demo");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1,3,MODEL_INPUT_SIZE,MODEL_INPUT_SIZE};

    const char* input_names[]  = {"images"};
    const char* output_names[] = {"output0", "output1"};

    // 统计总耗时
    double total_read_ms = 0.0;
    double total_prep_ms = 0.0;
    double total_infer_ms = 0.0;
    double total_post_ms = 0.0;
    int img_cnt = 0;

    // 遍历图片目录
    for(const auto& entry : fs::directory_iterator(img_dir))
    {
        if(!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(),ext.end(),ext.begin(),::tolower);
        if(ext != ".jpg" && ext != ".jpeg" && ext != ".png"){
            continue;
        }

        std::string img_path = entry.path().string();
        std::string stem = entry.path().stem().string();
        std::string out_path = out_dir + "/" + stem + "_out.jpg";
        img_cnt ++;

        // ========== 1.读图计时 ==========
        auto t0 = std::chrono::high_resolution_clock::now();
        cv::Mat img_origin = cv::imread(img_path);
        auto t1 = std::chrono::high_resolution_clock::now();
        double read_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        total_read_ms += read_ms;

        if(img_origin.empty()){
            std::cout << "["<<img_cnt<<"] skip empty file:" << img_path << std::endl;
            continue;
        }
        int orig_h = img_origin.rows;
        int orig_w = img_origin.cols;

        // ==========2.预处理计时 letterbox + HWC->CHW ==========
        auto t2 = std::chrono::high_resolution_clock::now();
        float scale;
        int pad_h, pad_w;
        cv::Mat lb_img = letterbox(img_origin, MODEL_INPUT_SIZE, scale, pad_h, pad_w);

        cv::Mat rgb;
        cv::cvtColor(lb_img, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);

        std::vector<float> input_tensor(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
        for(int c = 0; c < 3; c++){
            for(int y = 0; y < MODEL_INPUT_SIZE; y++){
                for(int x = 0; x < MODEL_INPUT_SIZE; x++){
                    input_tensor[c*MODEL_INPUT_SIZE*MODEL_INPUT_SIZE + y*MODEL_INPUT_SIZE + x]
                        = rgb.at<cv::Vec3f>(y,x)[c];
                }
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double prep_ms = std::chrono::duration<double,std::milli>(t3-t2).count();
        total_prep_ms += prep_ms;

        // ==========3.推理Run计时 ==========
        auto t4 = std::chrono::high_resolution_clock::now();
        Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
            mem_info,
            input_tensor.data(),
            input_tensor.size(),
            input_shape.data(),
            input_shape.size()
        );
        std::vector<Ort::Value> output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor_ort,
            1,
            output_names,
            2
        );
        auto t5 = std::chrono::high_resolution_clock::now();
        double infer_ms = std::chrono::duration<double,std::milli>(t5-t4).count();
        total_infer_ms += infer_ms;

        float* out0_ptr = output_tensors[0].GetTensorMutableData<float>();
        float* out1_ptr = output_tensors[1].GetTensorMutableData<float>();

        // ==========4.后处理计时：转置、置信过滤、NMS、mask解码绘图 ==========
        auto t6 = std::chrono::high_resolution_clock::now();
        const int DIM0 = 41;
        const int NUM_ANCHOR = 8400;

        std::vector<std::vector<float>> det(NUM_ANCHOR, std::vector<float>(DIM0));
        for(int a = 0; a < NUM_ANCHOR; a++)
        {
            for(int d = 0; d < DIM0; d++)
            {
                det[a][d] = out0_ptr[d * NUM_ANCHOR + a];
            }
        }

        std::vector<BoxInfo> pre_boxes;
        for(int a = 0; a < NUM_ANCHOR; a++)
        {
            float cx = det[a][0];
            float cy = det[a][1];
            float w  = det[a][2];
            float h  = det[a][3];

            float max_score = 0.f;
            int max_cls = 0;
            for(int c = 0; c < NUM_CLASSES; c++)
            {
                float s = det[a][4 + c];
                if(s > max_score)
                {
                    max_score = s;
                    max_cls = c;
                }
            }
            if(max_score < CONF_THRES) continue;

            BoxInfo bi;
            bi.score = max_score;
            bi.cls_id = max_cls;
            bi.x1 = cx - w/2.f;
            bi.y1 = cy - h/2.f;
            bi.x2 = cx + w/2.f;
            bi.y2 = cy + h/2.f;

            bi.mask_coeff.resize(MASK_COEFF_DIM);
            for(int k=0;k<MASK_COEFF_DIM;k++){
                bi.mask_coeff[k] = det[a][4 + NUM_CLASSES + k];
            }
            pre_boxes.push_back(bi);
        }

        std::vector<BoxInfo> final_boxes = nms_per_class(pre_boxes, IOU_THRES);

        std::vector<float> proto_data(MASK_COEFF_DIM * MASK_PROTO_H * MASK_PROTO_W);
        for(int c=0;c<MASK_COEFF_DIM;c++){
            for(int y=0;y<MASK_PROTO_H;y++){
                for(int x=0;x<MASK_PROTO_W;x++){
                    size_t src_idx = c * MASK_PROTO_H * MASK_PROTO_W + y * MASK_PROTO_W + x;
                    proto_data[src_idx] = out1_ptr[src_idx];
                }
            }
        }

        cv::Mat draw_out = img_origin.clone();
        for(auto& b : final_boxes)
        {
            float x1_640 = b.x1;
            float y1_640 = b.y1;
            float x2_640 = b.x2;
            float y2_640 = b.y2;

            float x1 = (x1_640 - pad_w) / scale;
            float y1 = (y1_640 - pad_h) / scale;
            float x2 = (x2_640 - pad_w) / scale;
            float y2 = (y2_640 - pad_h) / scale;

            x1 = std::max(0.f, std::min(x1, float(orig_w-1)));
            y1 = std::max(0.f, std::min(y1, float(orig_h-1)));
            x2 = std::max(0.f, std::min(x2, float(orig_w-1)));
            y2 = std::max(0.f, std::min(y2, float(orig_h-1)));

            cv::rectangle(draw_out, cv::Point((int)x1,(int)y1), cv::Point((int)x2,(int)y2),
                          cv::Scalar(0,255,0),2);

            cv::Mat mask_160(MASK_PROTO_H, MASK_PROTO_W, CV_32FC1);
            for(int y=0;y<MASK_PROTO_H;y++)
            {
                for(int x=0;x<MASK_PROTO_W;x++)
                {
                    float sum = 0.0f;
                    for(int k=0;k<MASK_COEFF_DIM;k++)
                    {
                        size_t p_idx = k * MASK_PROTO_H * MASK_PROTO_W + y * MASK_PROTO_W + x;
                        sum += b.mask_coeff[k] * proto_data[p_idx];
                    }
                    float val = sigmoid(sum);
                    mask_160.at<float>(y,x) = val;
                }
            }

            cv::Mat mask_640;
            cv::resize(mask_160, mask_640, cv::Size(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE));

            cv::Rect box_640((int)x1_640,(int)y1_640,(int)(x2_640-x1_640),(int)(y2_640-y1_640));
            cv::Mat mask_crop = cv::Mat::zeros(mask_640.size(), CV_32FC1);
            box_640 &= cv::Rect(0,0,MODEL_INPUT_SIZE,MODEL_INPUT_SIZE);
            mask_640(box_640).copyTo(mask_crop(box_640));

            cv::Mat mask_bin;
            cv::threshold(mask_crop, mask_bin, 0.5f, 255, cv::THRESH_BINARY);
            mask_bin.convertTo(mask_bin, CV_8UC1);

            cv::Mat mask_nopad = mask_bin(cv::Rect(pad_w, pad_h, MODEL_INPUT_SIZE-2*pad_w, MODEL_INPUT_SIZE-2*pad_h));
            cv::Mat mask_origin;
            cv::resize(mask_nopad, mask_origin, cv::Size(orig_w, orig_h));

            cv::Mat color_mask = cv::Mat::zeros(draw_out.size(), CV_8UC3);
            cv::Scalar color;
            switch (b.cls_id % 5)
            {
                case 0: color = cv::Scalar(255,50,50);break;
                case 1: color = cv::Scalar(50,255,50);break;
                case 2: color = cv::Scalar(50,50,255);break;
                case 3: color = cv::Scalar(255,255,50);break;
                case 4: color = cv::Scalar(255,50,255);break;
                default:color = cv::Scalar(128,128,128);
            }
            color_mask.setTo(color, mask_origin);
            cv::addWeighted(draw_out,1.0,color_mask,0.4,0,draw_out);
        }
        auto t7 = std::chrono::high_resolution_clock::now();
        double post_ms = std::chrono::duration<double,std::milli>(t7-t6).count();
        total_post_ms += post_ms;

        cv::imwrite(out_path, draw_out);
        std::cout << "["<<img_cnt<<"] "<<entry.path().filename()
                  <<" | read:"<<read_ms<<" ms | prep:"<<prep_ms<<" ms | infer:"<<infer_ms<<" ms | post:"<<post_ms<<" ms | obj:"<<final_boxes.size()<<"\n";
    }

    // 打印整套平均耗时，用于和Python脚本做论文性能对比
    if(img_cnt > 0)
    {
        std::cout << "\n====== 数据集平均耗时（C++ onnxruntime）======\n";
        std::cout << "图片总数量:" << img_cnt << "\n";
        std::cout << "平均读图:    " << total_read_ms / img_cnt << " ms\n";
        std::cout << "平均预处理:  " << total_prep_ms / img_cnt << " ms\n";
        std::cout << "平均推理Run: " << total_infer_ms / img_cnt << " ms\n";
        std::cout << "平均后处理:  " << total_post_ms / img_cnt << " ms\n";
        std::cout << "平均全套总:  " << (total_read_ms+total_prep_ms+total_infer_ms+total_post_ms)/img_cnt << " ms\n";
    }

    return 0;
}

