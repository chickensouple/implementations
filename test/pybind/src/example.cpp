#include <iostream>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;




double mean(const Eigen::VectorXd& vec) {
    return vec.mean();
}

double mean_helper(Eigen::Ref<const Eigen::VectorXd> vec) {
    return mean(vec);
}


Eigen::VectorXd mean_vec(const Eigen::MatrixXd& mat) {
    return mat.rowwise().mean();
}

Eigen::VectorXd mean_vec_helper(py::EigenDRef<const Eigen::MatrixXd> mat) {
    return mean_vec(mat);
}



int add(int i, int j) {
    return i + j*2;
}


double add2(double i, double j) {
    return i + j;
}

struct Pet {
    Pet(const std::string &name) : name(name) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }

    std::string name;
};



struct myobj {
    int a;

    myobj(int a_in) {a=a_in;};
};

myobj newObj(int a) {
    return myobj(a);
}

std::shared_ptr<myobj> newObjPtr(int a) {
    std::shared_ptr<myobj> ptr = std::make_shared<myobj>(a);
    return ptr;
}


template <class T>
class myvector {
public:
    myvector() { idx = 0; };

    void push(T& obj) {_data[idx++] = obj;};

    T get(int idx) {return _data[idx];};

private:
    int idx;
    T _data[10];
};


PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("mean", &mean_helper, "mean", py::arg().noconvert());

    m.def("mean_vec", &mean_vec_helper, "mean", py::arg().noconvert());

    m.def("add", &add, "A function which adds two numbers", "a"_a, "b"_a=2);

    m.def("add", &add2, "blah");

    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName);

    py::class_<myobj>(m, "myobj")
        .def(py::init<int>())
        .def_readwrite("a", &myobj::a);

    m.def("newObj", &newObj, "new obj", "a"_a=1);

    m.def("newObjPtr", &newObjPtr, "new obj ptr", "a"_a=1);

    py::class_<myvector<py::object>>(m, "myvector")
        .def(py::init<>())
        .def("push", &myvector<py::object>::push)
        .def("get", &myvector<py::object>::get);


}

