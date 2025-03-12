#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "src/ifps.cuh"
#include "src/IFPS_SphereTree.cuh"

namespace py = pybind11;

PYBIND11_MODULE(PyIfps,m){

    //Ifps
    py::class_ <Ifps>(m,"Ifps")
    .def(py::init<>())
    // -fps
    .def("fps",&Ifps::fps,"A function of FPS")

    // -inverse fps
    .def("inverse_fps",&Ifps::inverse_fps,"A function of inverse_fps")
    .def("inverse_shell_cpu",&Ifps::inverse_shell_cpu,"A function of inverse_shell")
    .def("inverse_shell_gpu",py::overload_cast<const std::vector<Eigen::Vector3f>&, const std::vector<Eigen::Vector3f>&,const std::vector<float> &,const unsigned int>(&Ifps::inverse_shell_gpu),"A function of inverse_shell_gpu")
    .def("inverse_shell_gpu",py::overload_cast<const torch::Tensor &,const torch::Tensor &,const torch::Tensor &,const unsigned int>(&Ifps::inverse_shell_gpu),"A function of inverse_shell_gpu");

    m.def("check_treepts_cpu_time",&check_treepts_cpu_time,"A function of 8-way CPU");
    m.def("check_treepts_gpu_time",&check_treepts_gpu_time,"A function of 8-way GPU");
    m.def("check_treepts_gpu_id",&check_treepts_gpu_id,"A function of 8-way GPU");
}   