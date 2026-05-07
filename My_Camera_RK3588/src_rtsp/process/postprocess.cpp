#include "postprocess.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <numeric>
#include <vector>

#include "utils/logging.h"

int get_top(float *pfProb, float *pfMaxProb, uint32_t *pMaxClass, uint32_t outputCount, uint32_t topNum)
{
    uint32_t i, j;
#define MAX_TOP_NUM 20
    if (topNum > MAX_TOP_NUM) return 0;
    memset(pfMaxProb, 0, sizeof(float) * topNum);
    memset(pMaxClass, 0xff, sizeof(float) * topNum);
    for (j = 0; j < topNum; j++)
        for (i = 0; i < outputCount; i++)
        {
            if ((i==*(pMaxClass+0))||(i==*(pMaxClass+1))||(i==*(pMaxClass+2))||
                (i==*(pMaxClass+3))||(i==*(pMaxClass+4))) continue;
            if (pfProb[i] > *(pfMaxProb+j)) { *(pfMaxProb+j)=pfProb[i]; *(pMaxClass+j)=i; }
        }
    return 1;
}

namespace yolo
{
    typedef struct {
        float xmin, ymin, xmax, ymax;
        float score;
        int   classId;
    } DetectRect;

    static int   input_w         = 640;
    static int   input_h         = 640;
    static float objectThreshold = 0.25f;
    static float nmsThreshold    = 0.45f;
    static int   headNum         = 3;
    static int   class_num       = 80;
    static int   strides[3]      = {8, 16, 32};
    static int   mapSize[3][2]   = {{80,80},{40,40},{20,20}};
    static int   REG_CHANNEL     = 16; // DFL bins per direction

#define ZQ_MAX(a,b) ((a)>(b)?(a):(b))
#define ZQ_MIN(a,b) ((a)<(b)?(a):(b))

    static inline float fast_exp(float x)
    {
        union { uint32_t i; float f; } v;
        v.i = (uint32_t)(12102203.1616540672f * x + 1064807160.56887296f);
        return v.f;
    }
    static inline float sigmoid(float x) { return 1.0f/(1.0f+fast_exp(-x)); }
    static inline float DeQnt2F32(int8_t q, int zp, float sc) { return ((float)q-(float)zp)*sc; }

    static inline float IOU(float x1min,float y1min,float x1max,float y1max,
                            float x2min,float y2min,float x2max,float y2max)
    {
        float iw = ZQ_MAX(0,ZQ_MIN(x1max,x2max)-ZQ_MAX(x1min,x2min));
        float ih = ZQ_MAX(0,ZQ_MIN(y1max,y2max)-ZQ_MAX(y1min,y2min));
        float inter = iw*ih;
        float total = (x1max-x1min)*(y1max-y1min)+(x2max-x2min)*(y2max-y2min)-inter;
        return (total>0)?inter/total:0;
    }

    // DFL softmax加权求和
    static float dfl_decode(float *bins, int n)
    {
        float mx = bins[0];
        for (int i=1;i<n;i++) if(bins[i]>mx) mx=bins[i];
        float s=0, r=0, ev[16];
        for (int i=0;i<n;i++) { ev[i]=fast_exp(bins[i]-mx); s+=ev[i]; }
        for (int i=0;i<n;i++) r+=(float)i*(ev[i]/s);
        return r;
    }

    // Meshgrid（只生成一次）
    static bool meshgrid_generated = false;
    static std::vector<float> meshgrid;
    std::vector<float>& GenerateMeshgrid()
    {
        if (meshgrid_generated) return meshgrid;
        meshgrid.clear();
        for (int idx=0;idx<headNum;idx++)
            for (int i=0;i<mapSize[idx][0];i++)
                for (int j=0;j<mapSize[idx][1];j++)
                { meshgrid.push_back(j+0.5f); meshgrid.push_back(i+0.5f); }
        meshgrid_generated=true;
        printf("=== yolov8 Meshgrid  Generate success! \n");
        return meshgrid;
    }

    // NMS
    static void do_nms(std::vector<DetectRect> &rects, std::vector<float> &out)
    {
        std::sort(rects.begin(),rects.end(),[](const DetectRect&a,const DetectRect&b){return a.score>b.score;});
        for (int i=0;i<(int)rects.size();i++)
        {
            if (rects[i].classId==-1) continue;
            out.push_back((float)rects[i].classId);
            out.push_back(rects[i].score);
            out.push_back(rects[i].xmin);
            out.push_back(rects[i].ymin);
            out.push_back(rects[i].xmax);
            out.push_back(rects[i].ymax);
            for (int j=i+1;j<(int)rects.size();j++)
            {
                if (rects[j].classId==-1) continue;
                if (IOU(rects[i].xmin,rects[i].ymin,rects[i].xmax,rects[i].ymax,
                        rects[j].xmin,rects[j].ymin,rects[j].xmax,rects[j].ymax)>nmsThreshold)
                    rects[j].classId=-1;
            }
        }
    }

    // =============================================================
    // 6输出模型（旧格式，yolov8.rknn）
    // 输出格式（经日志确认）：
    //   index0 cls1: [1,1,4,6400]  → 已解码的置信度，NCHW但实质是[1, 1, 4, H*W]
    //                  dim2=4表示每个anchor有4个值(可能是x,y,w,h或其他)
    //                  实际上：[1,1,4,6400] 意味着1类×4坐标×80×80
    //   index1 reg1: [1,80,80,80]  → 80类别分数，H=80,W=80
    //   index2 cls2: [1,1,4,1600]  → [1,1,4,40×40]
    //   index3 reg2: [1,80,40,40]  → 80类别分数，H=40,W=40
    //   index4 cls3: [1,1,4,400]   → [1,1,4,20×20]
    //   index5 reg3: [1,80,20,20]  → 80类别分数，H=20,W=20
    //
    // 注意：输出名字是cls/reg但含义与yolov8n相反！
    //   cls(index0,2,4) 实际是 box坐标输出 [1,1,4,HW] → 4个坐标(x1,y1,x2,y2归一化)
    //   reg(index1,3,5) 实际是 class分数输出 [1,80,H,W]
    // =============================================================
    int GetConvDetectionResultInt8(int8_t **pBlob,
                                   std::vector<int> &qnt_zp,
                                   std::vector<float> &qnt_scale,
                                   std::vector<float> &DetectiontRects)
    {
        std::vector<DetectRect> detectRects;

        for (int index=0; index<headNum; index++)
        {
            // 旧模型: pBlob[index*2+0]=cls(box坐标), pBlob[index*2+1]=reg(类别分数)
            int8_t *box_data = pBlob[index*2+0]; // [1,1,4,HW] → 4个坐标
            int8_t *cls_data = pBlob[index*2+1]; // [1,80,H,W] → 80类分数

            int zp_box  = qnt_zp[index*2+0];
            int zp_cls  = qnt_zp[index*2+1];
            float sc_box = qnt_scale[index*2+0];
            float sc_cls = qnt_scale[index*2+1];

            int H  = mapSize[index][0];
            int W  = mapSize[index][1];
            int HW = H * W;

            for (int pos=0; pos<HW; pos++)
            {
                // box坐标：[1,1,4,HW]中，4个坐标存储在dim2，pos为dim3索引
                // 布局: [x1区HW个, y1区HW个, x2区HW个, y2区HW个]
                float x1 = DeQnt2F32(box_data[0*HW + pos], zp_box, sc_box);
                float y1 = DeQnt2F32(box_data[1*HW + pos], zp_box, sc_box);
                float x2 = DeQnt2F32(box_data[2*HW + pos], zp_box, sc_box);
                float y2 = DeQnt2F32(box_data[3*HW + pos], zp_box, sc_box);

                // 类别分数：[1,80,H,W]
                float cls_max = 0.0f;
                int   cls_idx = 0;
                for (int c=0; c<class_num; c++)
                {
                    float v = sigmoid(DeQnt2F32(cls_data[c*HW+pos], zp_cls, sc_cls));
                    if (v > cls_max) { cls_max=v; cls_idx=c; }
                }
                if (cls_max < objectThreshold) continue;

                // 坐标已经是归一化的[0,1]
                float xmin = ZQ_MAX(0.0f, ZQ_MIN(1.0f, x1));
                float ymin = ZQ_MAX(0.0f, ZQ_MIN(1.0f, y1));
                float xmax = ZQ_MAX(0.0f, ZQ_MIN(1.0f, x2));
                float ymax = ZQ_MAX(0.0f, ZQ_MIN(1.0f, y2));
                if (xmax<=xmin || ymax<=ymin) continue;

                DetectRect dr;
                dr.xmin=xmin; dr.ymin=ymin; dr.xmax=xmax; dr.ymax=ymax;
                dr.classId=cls_idx; dr.score=cls_max;
                detectRects.push_back(dr);
            }
        }

        NN_LOG_DEBUG("6out NMS Before: %ld", detectRects.size());
        do_nms(detectRects, DetectiontRects);
        return 0;
    }

    // =============================================================
    // 9输出模型（yolov8n.rknn）
    // 临时方案：提高阈值到0.52，因为该模型的cls输出量化导致sigmoid后全在[0.5,0.54]
    // =============================================================
    int GetConvDetectionResultInt8_9Out(int8_t **pBlob,
                                        std::vector<int> &qnt_zp,
                                        std::vector<float> &qnt_scale,
                                        std::vector<float> &DetectiontRects)
    {
        auto &grid = GenerateMeshgrid();
        std::vector<DetectRect> detectRects;
        int gridIndex = -2;

        // 临时解决方案：提高阈值到0.52，因为该模型cls输出的量化参数导致
        // 所有值反量化后sigmoid都在[0.5,0.54]区间
        const float temp_threshold = 0.52f;

        for (int index=0; index<headNum; index++)
        {
            int8_t *reg = pBlob[index*2+0]; // box [1,64,H,W]
            int8_t *cls = pBlob[index*2+1]; // cls [1,80,H,W]

            int zp_reg  = qnt_zp[index*2+0];
            int zp_cls  = qnt_zp[index*2+1];
            float sc_reg = qnt_scale[index*2+0];
            float sc_cls = qnt_scale[index*2+1];

            int H  = mapSize[index][0];
            int W  = mapSize[index][1];
            int HW = H * W;

            for (int h=0; h<H; h++)
            {
                for (int w=0; w<W; w++)
                {
                    gridIndex += 2;
                    int offset = h*W+w;

                    // 找最大类别分数
                    float cls_max_raw = -1e9f;
                    int   cls_idx = 0;
                    for (int c=0; c<class_num; c++)
                    {
                        float raw = DeQnt2F32(cls[c*HW+offset], zp_cls, sc_cls);
                        if (raw > cls_max_raw) { cls_max_raw=raw; cls_idx=c; }
                    }
                    float cls_max = sigmoid(cls_max_raw);
                    
                    if (cls_max < temp_threshold) continue;

                    // DFL解码box
                    float bins[16];
                    auto dfl = [&](int ch_start) -> float {
                        for (int i=0;i<REG_CHANNEL;i++)
                            bins[i] = DeQnt2F32(reg[(ch_start+i)*HW+offset], zp_reg, sc_reg);
                        return dfl_decode(bins, REG_CHANNEL);
                    };

                    float l = dfl(0);
                    float t = dfl(REG_CHANNEL);
                    float r = dfl(REG_CHANNEL*2);
                    float b = dfl(REG_CHANNEL*3);

                    float cx = grid[gridIndex+0];
                    float cy = grid[gridIndex+1];

                    float xmin = ZQ_MAX((cx-l)*strides[index], 0.0f);
                    float ymin = ZQ_MAX((cy-t)*strides[index], 0.0f);
                    float xmax = ZQ_MIN((cx+r)*strides[index], (float)input_w);
                    float ymax = ZQ_MIN((cy+b)*strides[index], (float)input_h);
                    if (xmax<=xmin || ymax<=ymin) continue;

                    DetectRect dr;
                    dr.xmin    = xmin/input_w;
                    dr.ymin    = ymin/input_h;
                    dr.xmax    = xmax/input_w;
                    dr.ymax    = ymax/input_h;
                    dr.classId = cls_idx;
                    dr.score   = cls_max;
                    detectRects.push_back(dr);
                }
            }
        }

        do_nms(detectRects, DetectiontRects);
        return 0;
    }

    // 浮点版本
    int GetConvDetectionResult(float **pBlob, std::vector<float> &DetectiontRects)
    {
        auto &grid = GenerateMeshgrid();
        std::vector<DetectRect> detectRects;
        int gridIndex = -2;

        for (int index=0; index<headNum; index++)
        {
            float *reg = pBlob[index*2+0];
            float *cls = pBlob[index*2+1];
            int H=mapSize[index][0], W=mapSize[index][1], HW=H*W;

            for (int h=0;h<H;h++) for (int w=0;w<W;w++)
            {
                gridIndex+=2;
                int offset=h*W+w;
                float cls_max_raw=-1e9f; int cls_idx=0;
                for (int c=0;c<class_num;c++)
                {
                    float v=cls[c*HW+offset];
                    if(v>cls_max_raw){cls_max_raw=v;cls_idx=c;}
                }
                float cls_max=sigmoid(cls_max_raw);
                if(cls_max<objectThreshold) continue;

                float bins[16];
                auto dfl=[&](int cs)->float{
                    for(int i=0;i<REG_CHANNEL;i++) bins[i]=reg[(cs+i)*HW+offset];
                    return dfl_decode(bins,REG_CHANNEL);
                };
                float l=dfl(0),t=dfl(REG_CHANNEL),r=dfl(REG_CHANNEL*2),b=dfl(REG_CHANNEL*3);
                float cx=grid[gridIndex],cy=grid[gridIndex+1];
                float xmin=ZQ_MAX((cx-l)*strides[index],0.0f);
                float ymin=ZQ_MAX((cy-t)*strides[index],0.0f);
                float xmax=ZQ_MIN((cx+r)*strides[index],(float)input_w);
                float ymax=ZQ_MIN((cy+b)*strides[index],(float)input_h);
                if(xmax<=xmin||ymax<=ymin) continue;
                DetectRect dr;
                dr.xmin=xmin/input_w; dr.ymin=ymin/input_h;
                dr.xmax=xmax/input_w; dr.ymax=ymax/input_h;
                dr.classId=cls_idx; dr.score=cls_max;
                detectRects.push_back(dr);
            }
        }
        do_nms(detectRects, DetectiontRects);
        return 0;
    }

} // namespace yolo