#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <type_traits>


namespace py = pybind11;
using namespace std;


string cv_version() {
    return CV_VERSION;
}


template<class T>
cv::Mat vector2Mat(vector<vector<T>> v) {
    /* convert std vector into cv::Mat */
    size_t rn = v.size();       // row number
    size_t cn = v[0].size();    // column number

    // image type selection based on vector type
    int img_type;
    if (is_same<T, uchar>::value) {
        img_type = CV_8UC1;
    }
    if (is_same<T, float>::value) {
        img_type = CV_32FC1;
    }

    cv::Mat m(rn,cn,img_type);   // output mat

    for (size_t r=0; r<rn; r++) {
        // copy one line
        memcpy(m.data + r*cn*sizeof(T), v[r].data(), cn*sizeof(T));
    }
    return m;
}


template<class T>
vector<vector<T>> Mat2Vector(cv::Mat m) {
    /* convert cv::Mat into std vector */
    size_t rn = m.rows;         // row number
    size_t cn = m.cols;         // column number
    vector<vector<T>> v(rn);    // output mat

    for (size_t r=0; r<rn; r++) {
        v[r] = vector<T>(cn);
        // copy une line
        memcpy(v[r].data() , m.data + r*cn*sizeof(T), cn*sizeof(T));
    } 
    return v;
}


cv::Mat blur(cv::Mat m, int kernel_size) {
    /* Example of C++ function using an openCV function */
    cv::Mat out;
    cv::blur(m,out,cv::Size(kernel_size,kernel_size));
    return out;
}


template<class T>
vector<vector<T>> blur( vector<vector<T>> v, int kernel_size) {
    /* interface to the blur() function */
    cv::Mat m = vector2Mat(v);
    return Mat2Vector<T>(blur(m, kernel_size));
}


/* ====== PYBIND CODE ====== */
PYBIND11_MODULE(cvMat, m) {
    m.doc() = "cvMat <-> Python"; // optional module docstring
    m.def("cv_version", &cv_version, "get the current OpenCV version");
    m.def("blur", &blur<uchar>, "blur image with a specified kernel size", py::arg("img"), py::arg("kernel_size"));
    m.def("blur", &blur<float>, "blur image with a specified kernel size", py::arg("img"), py::arg("kernel_size"));
}