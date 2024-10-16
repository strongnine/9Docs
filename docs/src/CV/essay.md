## OpenCV 里图像数据的保存方式



```c++
void pixel_compare(const uint8_t *p_img, const cv::Mat img, int depth)
{
    int height = img.rows;
    int width = img.cols;
    int stride = img.cols;
    uchar pix;

    for (int Y = 0; Y < height; Y++) {
        const uint8_t *ptr_img = p_img + (Y * stride) * depth;
        for (int X = 0; X < width; X++, ptr_img += depth) {
            for (int D = 0; D < depth; D++) {
                if (depth == 3) {
                    pix = img.at<cv::Vec3b>(Y, X)[D];
                } else {
                    pix = img.at<uchar>(Y, X);
                }

                if (pix != ptr_img[D]) {
                    std::cout << "pixel dismatch at position: (height, width) = (" << Y << ", " << X
                              << ")" << std::endl;
                    return;
                }
            }
        }
    }
    std::cout << "all pixel is match. " << std::endl;
    return;
}
```

