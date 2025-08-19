#include "yolord.h"
#include <opencv2/opencv.hpp>
#include <fstream>

int reg_max=16;
YOLODetector::YOLODetector(){
    const char* node_names_[] = {"Conv_259", "Conv_274", "Conv_289", "Conv_252", "Conv_267", "Conv_282"};
    resized_img_width = 672;
    resized_img_height = 384;
    num_class = 8;
    confThreshold = 0.20;
    nmsThreshold = 0.60;
    pad_h = 0;
    pad_w = 0;
    num_stride = 3;
    
    // *
    for (int i = 0; i < num_stride * 2; i++){
        node_names[i] = node_names_[i];
        if (i < num_stride){
            stride[i] = (1 << i) * 8;
        }
    }
}


YOLODetector::~YOLODetector(){
}


int YOLODetector::Init(const char* model_path){
    ratio_h = 1.0 * resized_img_height / up.output_img_height;
    ratio_w = 1.0 * resized_img_width / up.output_img_width;

    NOVT_CNN_PARAM_S cnnPara;
    cnnPara.pModelPath = model_path;
    for (int i = 0; i < num_stride * 2; i++){
        cnnPara.output_layer_name.push_back(node_names[i]);
    }
    pCnnModel = (void*)(new novt_hwcnn::CNN(&cnnPara));
    if (NULL == pCnnModel) {
        std::cout << "Failed YOLOv6DetectionWork pCnnModel is NULL\n";
        return -100;
    }
    int ret = 0;
    
    ret = rmImageCreate(&rm_dst_img, EN_YUV420SP, resized_img_width, resized_img_height, RM_FALSE);
    if (ret != 0){
        std::cout << "rmImageCreate faild\n";
        return -1;
    }

    return 0;
}


bool cmp(Bbox box1, Bbox box2){
    return (box1.score > box2.score);
}


float iou(Bbox box1, Bbox box2){
    int max_x = std::max(box1.x0, box2.x0);
    int min_x = std::min(box1.x1, box2.x1);
    int max_y = std::max(box1.y0, box2.y0);
    int min_y = std::min(box1.y1, box2.y1);
    if (min_x <= max_x || min_y <= max_y)
        return 0;
    float over_area = (min_x - max_x) * (min_y - max_y);
    float area_a = (box1.x1 - box1.x0) * (box1.y1 - box1.y0);
    float area_b = (box2.x1 - box2.x0) * (box2.y1 - box2.y0);
    float iou = over_area / (area_a + area_b - over_area);
    return iou;
}


std::vector<Bbox> nms(std::vector<Bbox>& vec_boxs, float threshold){
    std::vector<Bbox> res;
    sort(vec_boxs.begin(), vec_boxs.end(), cmp);
    while (vec_boxs.size() > 0){
        res.push_back(vec_boxs[0]);
        for (unsigned int i = 0; i < vec_boxs.size() - 1; i++){
            float iou_value = iou(vec_boxs[0], vec_boxs[i + 1]);
            if (iou_value > threshold){
                vec_boxs.erase(vec_boxs.begin() + i + 1);
                i -= 1;
            }
        }
        vec_boxs.erase(vec_boxs.begin());
    }
    return res;
}


float softmax(float* dst, int length){
    float alpha = *std::max_element(dst, dst + length);
    float denominator = 0;
    float dis_sum = 0;
    for (int i = 0; i < length; i++){
        dst[i] = exp(dst[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; i++){
        dst[i] /= denominator;
        dis_sum += i * dst[i];
    }
    return dis_sum;
}


int YOLODetector::Run(AlgImage* img, std::vector<Bbox>& BBB){
    rm_src_img.s32Stride[0] = img->au32Stride[0];
    rm_src_img.s32Stride[1] = img->au32Stride[1];
    rm_src_img.s32Stride[2] = img->au32Stride[2];
    rm_src_img.u32PhyAddr[0] = (RM_U32)img->au32PhyAddr[0];
    rm_src_img.u32PhyAddr[1] = (RM_U32)img->au32PhyAddr[1];
    rm_src_img.u32PhyAddr[2] = (RM_U32)img->au32PhyAddr[2];
    rm_src_img.pu8VirAddr[0] = (RM_U8*)img->au32VirAddr[0];
    rm_src_img.pu8VirAddr[1] = (RM_U8*)img->au32VirAddr[1];
    rm_src_img.pu8VirAddr[2] = (RM_U8*)img->au32VirAddr[2];
    rm_src_img.s32Width = img->u32Width;
    rm_src_img.s32Height = img->u32Height;
    rm_src_img.enImageType = EN_YUV420SP;
    if (rm_src_img.s32Width == rm_dst_img.s32Width && rm_src_img.s32Height == rm_dst_img.s32Height){
        NOVT_image_roi.au32Stride[0] = rm_src_img.s32Stride[0];
        NOVT_image_roi.au32Stride[1] = rm_src_img.s32Stride[1];
        NOVT_image_roi.au32Stride[2] = rm_src_img.s32Stride[2];
        NOVT_image_roi.au32PhyAddr[0] = (unsigned int)rm_src_img.u32PhyAddr[0];
        NOVT_image_roi.au32PhyAddr[1] = (unsigned int)rm_src_img.u32PhyAddr[1];
        NOVT_image_roi.au32PhyAddr[2] = (unsigned int)rm_src_img.u32PhyAddr[2];
        NOVT_image_roi.au32VirAddr[0] = (unsigned int)rm_src_img.pu8VirAddr[0];
        NOVT_image_roi.au32VirAddr[1] = (unsigned int)rm_src_img.pu8VirAddr[1];
        NOVT_image_roi.au32VirAddr[2] = (unsigned int)rm_src_img.pu8VirAddr[2];
        NOVT_image_roi.u32Width = rm_src_img.s32Width;
        NOVT_image_roi.u32Height = rm_src_img.s32Height;
        NOVT_image_roi.enType = NOVT_IMAGE_TYPE_YUV420SP;
    }
    else{
        rmImageResize(&rm_src_img, &rm_dst_img, NULL, RM_TRUE);
        NOVT_image_roi.au32Stride[0] = rm_dst_img.s32Stride[0];
        NOVT_image_roi.au32Stride[1] = rm_dst_img.s32Stride[1];
        NOVT_image_roi.au32Stride[2] = rm_dst_img.s32Stride[2];
        NOVT_image_roi.au32PhyAddr[0] = (unsigned int)rm_dst_img.u32PhyAddr[0];
        NOVT_image_roi.au32PhyAddr[1] = (unsigned int)rm_dst_img.u32PhyAddr[1];
        NOVT_image_roi.au32PhyAddr[2] = (unsigned int)rm_dst_img.u32PhyAddr[2];
        NOVT_image_roi.au32VirAddr[0] = (unsigned int)rm_dst_img.pu8VirAddr[0];
        NOVT_image_roi.au32VirAddr[1] = (unsigned int)rm_dst_img.pu8VirAddr[1];
        NOVT_image_roi.au32VirAddr[2] = (unsigned int)rm_dst_img.pu8VirAddr[2];
        NOVT_image_roi.u32Width = rm_dst_img.s32Width;
        NOVT_image_roi.u32Height = rm_dst_img.s32Height;
        NOVT_image_roi.enType = NOVT_IMAGE_TYPE_YUV420SP;
    }
    
    std::vector<NOVT_BLOB_S> layerout;
    if (NULL != pCnnModel){
        ((novt_hwcnn::CNN*)(pCnnModel))->Run(&NOVT_image_roi, layerout);
    }
    else{
        std::cout << "Failed YOLODetectionWork pCnnModel is NULL\n";
        return -1;
    }
    if (!layerout.empty()){
        std::vector<Bbox> boxes;
        float thres_conv = log(confThreshold / (1 - confThreshold));
        for (int n = 0; n < num_stride; n++){
            unsigned int u32Height = layerout[n].u32Height;
            unsigned int u32Width = layerout[n].u32Width;
            unsigned int u32Chn = layerout[n].u32Chn;
            int cls_shift = layerout[n].out_shift;
            int reg_shift = layerout[n + num_stride].out_shift;
            short* cls = (short*)layerout[n].u32VirAddr;
            short* reg = (short*)layerout[n + num_stride].u32VirAddr;

            for (unsigned int h = 0; h < u32Height; h++){
                for (unsigned int w = 0; w < u32Width; w++){
                    float obj = INT_MIN;
                    int cls_index = -1;
                    short* conf = cls + h * u32Width + w;
                    for (unsigned int c = 0; c < u32Chn; c++){
                        float p = *(conf + c * u32Width * u32Height);
                        if (p > obj){
                            obj = p / float(1 << cls_shift);
                            cls_index = c;
                        }
                    }
                    if (obj >= thres_conv){
                        int x1, y1, x2, y2;
                        float pred_ltrb[4];
                        for (int k = 0; k < 4; k++){
                            float dis_after_sm[reg_max];
                            for (int i = 0; i < reg_max; i++){
                                dis_after_sm[i] = *(reg + h * u32Width + w + u32Width * u32Height * (i + k * reg_max)) / float(1 << reg_shift);
                            }
                            pred_ltrb[k] = softmax(dis_after_sm, reg_max);
                        }
                        x1 = int((w + 0.5 - pred_ltrb[0]) * stride[n]);
                        y1 = int((h + 0.5 - pred_ltrb[1]) * stride[n]);
                        x2 = int((w + 0.5 + pred_ltrb[2]) * stride[n]);
                        y2 = int((h + 0.5 + pred_ltrb[3]) * stride[n]);
                        Bbox b = {x1, y1, x2, y2, obj, cls_index};
                        boxes.push_back(b);
                    }
                }
            }
        }
        BBB =  nms(boxes, nmsThreshold);
        for (std::vector<Bbox>::iterator it = BBB.begin(); it != BBB.end(); it++){
            (*it).x0 = std::min(std::max(int(((*it).x0 + pad_w) / ratio_w), 0), up.output_img_width);
            (*it).x1 = std::min(std::max(int(((*it).x1 + pad_w) / ratio_w), 0), up.output_img_width);
            (*it).y0 = std::min(std::max(int(((*it).y0 + pad_h) / ratio_h), 0), up.output_img_height);
            (*it).y1 = std::min(std::max(int(((*it).y1 + pad_h) / ratio_h), 0), up.output_img_height);
            (*it).score = 1 / (1 + exp(-(*it).score));
            (*it).label = idMap[(*it).label];
        }
    }
    return 0;
}


int YOLODetector::Unit(){
    if(NULL != pCnnModel){
        delete (novt_hwcnn::CNN*)pCnnModel;
        pCnnModel = NULL;
    }
    rmImageFree(&rm_src_img);
    rmImageFree(&rm_dst_img);
    return 0;
}
