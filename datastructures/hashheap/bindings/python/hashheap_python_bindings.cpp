#include "hashheap/hashheap.hpp"
#include <pybind11/pybind11.h>
#include <stdexcept>

#include <Python.h>

namespace py = pybind11;


struct PyObjectHash {
    size_t operator()(const py::object& p) const {return py::hash(p);};
};

struct PyObjectEqual {
    bool operator()(const py::object& lhs, const py::object& rhs) const {


        int res = PyObject_RichCompareBool(lhs.ptr(), rhs.ptr(), Py_EQ);

        if (res == -1) {
            throw std::runtime_error("Comparison not valid.");
        }

        return (bool)res;
    };
};

typedef HashHeap<py::object, int, std::less<int>, PyObjectHash, PyObjectEqual> HashHeapPyObj;


PYBIND11_MODULE(HashHeap, module) {
    module.doc() = "Datastructure that is a combination of a Min Binary Heap and a Hash Map for fast update_priority() and remove() operations";

    py::class_<HashHeapPyObj>(module, "HashHeap")
        .def(py::init<>())
        .def("push",
            &HashHeapPyObj::push,
            "push",
            py::arg("obj"),
            py::arg("priority"))
        .def("pop",
            &HashHeapPyObj::pop,
            "pop")
        .def("size",
            &HashHeapPyObj::size)
        .def("empty",
            &HashHeapPyObj::empty)
        .def("update_priority",
            &HashHeapPyObj::update_priority,
            "update priority",
            py::arg("obj"),
            py::arg("new_priority"))
        .def("remove",
            &HashHeapPyObj::remove,
            "remove",
            py::arg("obj"));
}






