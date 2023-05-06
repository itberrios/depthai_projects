#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <depthai/depthai.hpp>

static std::atomic<bool> syncNN{ true };
static const std::string BLOB_PATH = "C:/Users/itber/.cache/blobconverter/deeplab_v3_mnv2_256x256_openvino_2021.4_6shave.blob";
// static const std::string BLOB_PATH = "C:/Users/itber/.cache/blobconverter/deeplabv3p_person_openvino_2021.4_5shave.blob";

// NN Dimensions
const int HEIGHT = 256;
const int WIDTH = 256;

// Display Dimensions
const int TGT_HEIGHT = 400;
const int TGT_WIDTH = 400;

cv::Mat cropToSquare(cv::Mat frame) {
    int rows = frame.rows;
    int cols = frame.cols;
    int delta = int((cols - rows) / 2);
    // cv::Rect roi(0, rows, delta, cols - delta);
    // 
    // cv::Rect roi(delta, 0, cols - delta, rows);
    cv::Rect roi(delta, 0, cols - 2*delta, rows);

    return frame(roi);
}

cv::Mat getMask(std::vector<std::int32_t> detections, const int rows, const int cols) {
    // construct matrix with detections vector as initial data (no copy ;))
    cv::Mat mask(rows, cols, CV_32S, detections.data());

    // convert to uint8 and scale by 255
    mask.convertTo(mask, CV_8U, 255);
    return mask;
}


cv::Mat computeFlow(cv::Mat prev, cv::Mat next) {
    // compute Farenback Optical Flow

    cv::Mat prevGray, nextGray;
    cv::cvtColor(prev, prevGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(next, nextGray, cv::COLOR_BGR2GRAY);
    cv::Mat flow(prevGray.size(), CV_32FC2);

    // optical flow computation
    cv::calcOpticalFlowFarneback(prevGray, nextGray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
  
    // visualization
    cv::Mat flow_parts[2];
    cv::split(flow, flow_parts);
    cv::Mat magnitude, angle, magn_norm;
    cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));
    //build hsv image
    cv::Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    cv::merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);

    return bgr;
}

int main(int argc, char** argv) {
    std::string nnPath(BLOB_PATH);

    // If path to blob specified, use that
    if (argc > 1) {
        nnPath = std::string(argv[1]);
    }

    // Create pipeline
    dai::Pipeline pipeline;

    // Setup RGB Camera
    auto camRgb = pipeline.create<dai::node::ColorCamera>();
    camRgb->setBoardSocket(dai::CameraBoardSocket::RGB);
    camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    camRgb->setIspScale(2, 3);
    camRgb->setInterleaved(false);

    // DeepLabV3 specific settings
    camRgb->setPreviewSize(HEIGHT, WIDTH);
    camRgb->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);
    camRgb->setFps(40);

    // Link NN output to XLinkOut
    auto isp_xout = pipeline.create<dai::node::XLinkOut>();
    isp_xout->setStreamName("rgb");
    camRgb->isp.link(isp_xout->input);
    
    // Define NN for detections
    auto detectionNetwork = pipeline.create<dai::node::NeuralNetwork>();
    detectionNetwork->setBlobPath(nnPath);
    detectionNetwork->input.setBlocking(false);
    detectionNetwork->setNumInferenceThreads(2);

    // link camera to network
    camRgb->preview.link(detectionNetwork->input); 
    
    // link NN output to XLinkOut
    auto nnOut = pipeline.create<dai::node::XLinkOut>();
    nnOut->setStreamName("detections");
    detectionNetwork->out.link(nnOut->input);
    
    // Connect to device and start pipeline
    dai::Device device(pipeline);

    // Output queues will be used to get the rgb frames and nn data from the outputs defined above
    auto qRgb = device.getOutputQueue("rgb", 4, false);
    auto qDet = device.getOutputQueue("detections", 4, false);

    cv::Mat frame, mask, maskedFrame, bgrFlow, flowFilt;
    cv::Mat prevMaskedFrame (TGT_HEIGHT, TGT_WIDTH, 16, cv::Scalar(0)); // ensure same type as frame
    std::vector<std::int32_t> detections;
    auto startTime = std::chrono::steady_clock::now();
    int counter = 0;
    float fps = 0;
    auto textColor = cv::Scalar(255, 255, 255);

    while (true) {
        std::shared_ptr<dai::ImgFrame> inRgb;
        std::shared_ptr<dai::NNData> inDet;

        if (syncNN) {
            inRgb = qRgb->get<dai::ImgFrame>();
            inDet = qDet->get<dai::NNData>();
        }
        else {
            inRgb = qRgb->tryGet<dai::ImgFrame>();
            inDet = qDet->tryGet<dai::NNData>();
        }

        counter++;
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(currentTime - startTime);
        // update FPS every second
        if (elapsed > std::chrono::seconds(1)) {
            fps = counter / elapsed.count();
            counter = 0;
            startTime = currentTime;
        }

        if (inRgb) {
            frame = inRgb->getCvFrame();
        }

        if (inDet) {
            detections = inDet->getFirstLayerInt32();
            mask = getMask(detections, HEIGHT, WIDTH);
            cv::resize(mask, mask, cv::Size(TGT_HEIGHT, TGT_WIDTH));
            // cv::imshow("mask", mask);
        }

        if (!frame.empty()) {
            frame = cropToSquare(frame); // crops to preview size
            cv::resize(frame, frame, cv::Size(TGT_HEIGHT, TGT_WIDTH));

            // mask frame
            frame.copyTo(maskedFrame, mask);

            // write FPS on frame
            std::stringstream fpsStr;
            fpsStr << "NN fps: " << std::fixed << std::setprecision(2) << fps;
            cv::putText(frame, fpsStr.str(), cv::Point(2, TGT_HEIGHT - 4), cv::FONT_HERSHEY_TRIPLEX, 0.4, textColor);

            // perform optical flow on masked frame
            bgrFlow = computeFlow(prevMaskedFrame, maskedFrame);

            // get filter
            cv::addWeighted(frame, 0.5, bgrFlow, 0.5, 0.0, flowFilt);

            // cv::imshow("video", frame);
            // cv::imshow("masked frame", maskedFrame);
            cv::imshow("masked flow", bgrFlow);
            cv::imshow("flow filter", flowFilt);

            // update prev masked frame
            prevMaskedFrame = maskedFrame.clone();

            // reset maskedFrame for display
            maskedFrame *= 0;
        }

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q') {
            return 0;
        }
    }

    return 0;
}
