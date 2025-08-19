#ifndef __YOLO_H__
#define __YOLO_H__
#include <vector>
#include <iostream>
#include "px_types_common.h"
#include "image_process_interface.h"
#include "rmai_novt.hpp"
#include <unordered_map>


typedef struct Bbox{
    int x0;
    int y0;
    int x1;
    int y1;
    float score;
    int label;
} Bbox;


typedef struct UsrParams{
    int org_img_width;
    int org_img_height;
    int output_img_width;
    int output_img_height;
} UsrParams;


class YOLODetector{
    private:
        std::unordered_map<int, int> idMap;
        int num_class;
        float confThreshold;
        float nmsThreshold;
        int resized_img_width;
        int resized_img_height;
        float ratio_w;
        float ratio_h;
        int pad_w;
        int pad_h;
        int num_stride;
        int stride[3];
        const char* node_names[6];

        rmImage rm_src_img;
        rmImage rm_dst_img;
        void* pCnnModel = NULL;
        NOVT_IMAGE_S NOVT_image_roi;
	public:
        UsrParams up;
        YOLODetector();
        ~YOLODetector();
		int Init(const char* model_path);
		int Run(AlgImage* img, std::vector<Bbox>& BBB);
		int Unit();
};

#endif
