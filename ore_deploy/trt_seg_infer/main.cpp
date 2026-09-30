// YOLOv11-seg TensorRT C++ 批量推理  (TensorRT 8.6 现代 API)
// 输入: ore_seg_best.engine (1x3x640x640)
// 输出: output0 1x41x8400 (检测头), output1 1x32x160x160 (分割 mask)
// 说明: YOLOv11 anchor-free, 41 = 4(bbox) + 5(类别) + 32(mask系数)
// 用法: ./trt_seg_infer <engine.engine> <图片文件夹> [输出文件夹]
//       输出文件夹默认 = 输入文件夹/results
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <chrono>

#include <opencv2/opencv.hpp>

using namespace nvinfer1;
namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

// ---------------- 日志 ----------------
class Logger : public ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cout << "[TRT] " << msg << "\n";
    }
};

// ---------------- 配置 ----------------
struct Config {
    int    inputH = 640, inputW = 640;
    int    numClasses = 5;
    int    numMask    = 32;
    int    numAnchors = 8400;
    float  confThr = 0.6f;
    float  iouThr  = 0.45f;
};

// ---------------- 分阶段计时 ----------------
struct Timing {
    double read  = 0;   // 读图 ms
    double pre   = 0;   // 预处理 ms
    double infer = 0;   // 推理 Run ms
    double post  = 0;   // 后处理 ms
    double total = 0;   // 全套总 ms
};

// ---------------- 工具 ----------------
static std::vector<char> loadFile(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "Cannot open file: " << p << "\n"; exit(1); }
    std::streamsize n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> b(n);
    f.read(b.data(), n);
    return b;
}

static bool isImageFile(const fs::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
           ext == ".bmp"  || ext == ".tif"  || ext == ".tiff";
}

// ---------------- 检测结果 ----------------
struct Detection {
    float cx, cy, w, h, score, classId;
    std::vector<float> mask;
};

static float iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.cx - a.w / 2, b.cx - b.w / 2);
    float y1 = std::max(a.cy - a.h / 2, b.cy - b.h / 2);
    float x2 = std::min(a.cx + a.w / 2, b.cx + b.w / 2);
    float y2 = std::min(a.cy + a.h / 2, b.cy + b.h / 2);
    float iw = std::max(0.f, x2 - x1), ih = std::max(0.f, y2 - y1);
    float inter = iw * ih;
    float ua = a.w * a.h + b.w * b.h - inter;
    return ua > 0 ? inter / ua : 0.f;
}

static void nms(std::vector<Detection>& dets, float iouThr) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b){ return a.score > b.score; });
    std::vector<Detection> keep;
    std::vector<bool> removed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        keep.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!removed[j] && dets[i].classId == dets[j].classId && iou(dets[i], dets[j]) > iouThr)
                removed[j] = true;
        }
    }
    dets = keep;
}

// ---------------- 前处理 letterbox ----------------
static void preprocess(cv::Mat& img, float* hostInput, int targetW, int targetH,
                       float& scale, int& padX, int& padY) {
    int iw = img.cols, ih = img.rows;
    scale = std::min((float)targetW / iw, (float)targetH / ih);
    int nw = (int)std::round(iw * scale), nh = (int)std::round(ih * scale);
    padX = (targetW - nw) / 2;
    padY = (targetH - nh) / 2;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(nw, nh));
    cv::Mat canvas = cv::Mat::zeros(targetH, targetW, CV_8UC3);
    resized.copyTo(canvas(cv::Rect(padX, padY, nw, nh)));
    std::vector<cv::Mat> chs(3);
    cv::split(canvas, chs);
    for (int c = 0; c < 3; ++c) {
        cv::Mat m;
        chs[c].convertTo(m, CV_32F, 1.0 / 255.0);
        std::memcpy(hostInput + c * targetH * targetW, m.data,
                    targetH * targetW * sizeof(float));
    }
}

// ---------------- 单张图推理 + 画框/mask ----------------
static cv::Mat inferAndDraw(
    IExecutionContext* ctx, const Config& cfg,
    const char* inputName, const char* outDetName, const char* outMaskName,
    void* dInput, void* dDet, void* dMask,
    std::vector<float>& hInput, std::vector<float>& hDet, std::vector<float>& hMask,
    cudaStream_t stream,
    cv::Mat& img, Timing& tm)
{
    const size_t inBytes  = 1 * 3 * cfg.inputH * cfg.inputW * sizeof(float);
    const size_t detBytes = 1 * (4 + cfg.numClasses + cfg.numMask) * cfg.numAnchors * sizeof(float);
    const size_t maskBytes= 1 * cfg.numMask * 160 * 160 * sizeof(float);

    // ---- 预处理 ----
    auto t0 = Clock::now();
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    float scale; int padX, padY;
    preprocess(rgb, hInput.data(), cfg.inputW, cfg.inputH, scale, padX, padY);
    tm.pre = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    // ---- 推理 Run ----
    t0 = Clock::now();
    Dims4 inDims{1, 3, cfg.inputH, cfg.inputW};
    ctx->setInputShape(inputName, inDims);
    ctx->setTensorAddress(inputName, dInput);
    ctx->setTensorAddress(outDetName, dDet);
    ctx->setTensorAddress(outMaskName, dMask);
    cudaMemcpyAsync(dInput, hInput.data(), inBytes, cudaMemcpyHostToDevice, stream);
    ctx->enqueueV3(stream);
    cudaMemcpyAsync(hDet.data(), dDet, detBytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(hMask.data(), dMask, maskBytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    tm.infer = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    // ---- 后处理 (解析+NMS+画框+mask) ----
    t0 = Clock::now();

    // 解析检测头: ch[0-3]=bbox, ch[4-8]=类别, ch[9-40]=mask系数
    const int nc = cfg.numClasses, nm = cfg.numMask, na = cfg.numAnchors;
    std::vector<Detection> dets;
    for (int a = 0; a < na; ++a) {
        float cx = hDet[0 * na + a];
        float cy = hDet[1 * na + a];
        float w  = hDet[2 * na + a];
        float h  = hDet[3 * na + a];
        int bestCls = -1; float bestScore = -1;
        for (int c = 0; c < nc; ++c) {
            float s = 1.0f / (1.0f + std::exp(-hDet[(4 + c) * na + a]));
            if (s > bestScore) { bestScore = s; bestCls = c; }
        }
        if (bestScore < cfg.confThr) continue;
        Detection d;
        d.cx = cx; d.cy = cy; d.w = w; d.h = h;
        d.score = bestScore; d.classId = bestCls;
        d.mask.resize(nm);
        for (int i = 0; i < nm; ++i) d.mask[i] = hDet[(4 + nc + i) * na + a];
        dets.push_back(d);
    }
    nms(dets, cfg.iouThr);

    // 画框
    cv::Mat out = img.clone();
    for (auto& d : dets) {
        float x1 = (d.cx - d.w/2 - padX) / scale;
        float y1 = (d.cy - d.h/2 - padY) / scale;
        float x2 = (d.cx + d.w/2 - padX) / scale;
        float y2 = (d.cy + d.h/2 - padY) / scale;
        if (x2 - x1 < 1 || y2 - y1 < 1) continue;
        cv::rectangle(out, cv::Rect((int)x1, (int)y1, (int)(x2-x1), (int)(y2-y1)),
                      cv::Scalar(0, 255, 0), 2);
        std::stringstream ss; ss << "cls" << (int)d.classId << " " << d.score;
        cv::putText(out, ss.str(), cv::Point((int)x1, (int)(y1-5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
    }

    // mask 叠加
    if (!dets.empty()) {
        int mw = 160, mh = 160;
        for (auto& d : dets) {
            cv::Mat acc = cv::Mat::zeros(mh, mw, CV_32F);
            for (int i = 0; i < cfg.numMask; ++i) {
                const float* proto = &hMask[i * mh * mw];
                for (int y = 0; y < mh; ++y) {
                    float* row = acc.ptr<float>(y);
                    const float* pr = proto + y * mw;
                    for (int x = 0; x < mw; ++x) row[x] += d.mask[i] * pr[x];
                }
            }
            cv::Mat mask160;
            cv::exp(-acc, acc);
            acc = 1.0f / (1.0f + acc);
            cv::threshold(acc, mask160, 0.5, 1.0, cv::THRESH_BINARY);
            cv::Mat mask640;
            cv::resize(mask160, mask640, cv::Size(cfg.inputW, cfg.inputH));

            float bx1 = d.cx - d.w/2, by1 = d.cy - d.h/2;
            float bx2 = d.cx + d.w/2, by2 = d.cy + d.h/2;
            for (int y640 = (int)by1; y640 < (int)by2; ++y640) {
                for (int x640 = (int)bx1; x640 < (int)bx2; ++x640) {
                    if (y640 < 0 || y640 >= cfg.inputH || x640 < 0 || x640 >= cfg.inputW) continue;
                    if (mask640.at<float>(y640, x640) <= 0.5f) continue;
                    int xo = (int)((x640 - padX) / scale);
                    int yo = (int)((y640 - padY) / scale);
                    if (yo < 0 || yo >= out.rows || xo < 0 || xo >= out.cols) continue;
                    cv::Vec3b& px = out.at<cv::Vec3b>(yo, xo);
                    px = px * 0.5f + cv::Vec3b(0, 120, 255) * 0.5f;
                }
            }
        }
    }
    tm.post = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    tm.total = tm.pre + tm.infer + tm.post;
    return out;
}

// ---------------- 主函数 ----------------
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <engine.engine> <图片文件夹> [输出文件夹]\n";
        return -1;
    }
    Config cfg;

    // 1. 加载 engine (只加载一次)
    Logger logger;
    auto rt = std::unique_ptr<IRuntime>(createInferRuntime(logger));
    auto engData = loadFile(argv[1]);
    auto engine = std::unique_ptr<ICudaEngine>(
        rt->deserializeCudaEngine(engData.data(), engData.size()));
    if (!engine) { std::cerr << "deserialize engine failed\n"; return -1; }
    auto ctx = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
    std::cout << "Engine loaded. bindings: " << engine->getNbIOTensors() << "\n";

    // 2. 找输入/输出 binding (按维度判断, 不按顺序)
    const char* inputName = nullptr;
    const char* outDetName = nullptr;
    const char* outMaskName = nullptr;
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* n = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(n) == TensorIOMode::kINPUT) inputName = n;
        else {
            Dims d = engine->getTensorShape(n);
            if (d.nbDims == 3) outDetName = n;
            else outMaskName = n;
        }
    }

    // 3. 分配显存 (只分配一次)
    const size_t inBytes  = 1 * 3 * cfg.inputH * cfg.inputW * sizeof(float);
    const size_t detBytes = 1 * (4 + cfg.numClasses + cfg.numMask) * cfg.numAnchors * sizeof(float);
    const size_t maskBytes= 1 * cfg.numMask * 160 * 160 * sizeof(float);
    void *dInput=nullptr, *dDet=nullptr, *dMask=nullptr;
    cudaMalloc(&dInput, inBytes);
    cudaMalloc(&dDet, detBytes);
    cudaMalloc(&dMask, maskBytes);
    std::vector<float> hInput(inBytes/4), hDet(detBytes/4), hMask(maskBytes/4);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 4. 收集输入文件夹里的图片
    fs::path inDir = argv[2];
    if (!fs::is_directory(inDir)) {
        std::cerr << "Not a directory: " << inDir << "\n";
        return -1;
    }
    std::vector<fs::path> imgFiles;
    for (auto& e : fs::directory_iterator(inDir)) {
        if (e.is_regular_file() && isImageFile(e.path()))
            imgFiles.push_back(e.path());
    }
    std::sort(imgFiles.begin(), imgFiles.end());
    if (imgFiles.empty()) {
        std::cerr << "No images found in " << inDir << "\n";
        return -1;
    }
    std::cout << "Found " << imgFiles.size() << " images.\n";

    // 5. 输出目录
    fs::path outDir = (argc >= 4) ? fs::path(argv[3]) : inDir / "results";
    fs::create_directories(outDir);

    // 6. 逐张推理 + 计时
    int ok = 0, fail = 0;
    Timing sum;   // 累计
    for (size_t i = 0; i < imgFiles.size(); ++i) {
        const auto& p = imgFiles[i];
        std::cout << "[" << (i+1) << "/" << imgFiles.size() << "] " << p.filename().string() << " ... ";

        Timing tm;
        auto tRead = Clock::now();
        cv::Mat img = cv::imread(p.string());
        tm.read = std::chrono::duration<double, std::milli>(Clock::now() - tRead).count();
        if (img.empty()) { std::cout << "SKIP (read fail)\n"; ++fail; continue; }

        cv::Mat result = inferAndDraw(ctx.get(), cfg, inputName, outDetName, outMaskName,
                                      dInput, dDet, dMask, hInput, hDet, hMask, stream, img, tm);
        fs::path outPath = outDir / p.filename();
        cv::imwrite(outPath.string(), result);
        std::cout << "done  read=" << tm.read << "ms pre=" << tm.pre
                  << "ms infer=" << tm.infer << "ms post=" << tm.post
                  << "ms total=" << tm.total << "ms\n";

        sum.read  += tm.read;
        sum.pre   += tm.pre;
        sum.infer += tm.infer;
        sum.post  += tm.post;
        sum.total += tm.read + tm.total;
        ++ok;
    }

    // 7. 打印平均耗时
    std::cout << "\n====== 数据集平均耗时（C++ TensorRT）======\n";
    std::cout << "图片总数量：" << ok << "\n";
    std::cout << std::fixed;
    std::cout.precision(4);
    std::cout << "平均读图:    " << sum.read  / ok << " ms\n";
    std::cout << "平均预处理:  " << sum.pre   / ok << " ms\n";
    std::cout << "平均推理 Run: " << sum.infer / ok << " ms\n";
    std::cout << "平均后处理:  " << sum.post  / ok << " ms\n";
    std::cout << "平均全套总:  " << sum.total / ok << " ms\n";
    std::cout << "Results saved to: " << outDir << "\n";

    cudaStreamDestroy(stream);
    cudaFree(dInput); cudaFree(dDet); cudaFree(dMask);
    return 0;
}

