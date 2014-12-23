
#include <Python.h> // This must be included before anything else. See http://bugs.python.org/issue10910
#include <boost/python.hpp>
#include <string>
#include <vector>
#include <iostream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h> 

namespace bp = boost::python;
namespace bpn = boost::python::numeric;

namespace {
  template <typename T> int numpy_typenum() {}
  template <> int numpy_typenum<bool>() { return NPY_BOOL; }
  template <> int numpy_typenum<char>() { return NPY_INT8; }
  template <> int numpy_typenum<short>() { return NPY_INT16; }
  template <> int numpy_typenum<int>() { return NPY_INT32; }
  template <> int numpy_typenum<long long>() { return NPY_INT64; }
  template <> int numpy_typenum<unsigned char>() { return NPY_UINT8; }
  template <> int numpy_typenum<unsigned short>() { return NPY_UINT16; }
  template <> int numpy_typenum<unsigned int>() { return NPY_UINT32; }
  template <> int numpy_typenum<unsigned long long>() { return NPY_UINT64; }
  template <> int numpy_typenum<float>() { return NPY_FLOAT32; }
  template <> int numpy_typenum<double>() { return NPY_FLOAT64; }

  bpn::array boost_ndarray_from_pyarray(PyObject *pyarray) {
    bp::handle<> handle(pyarray);
    return bpn::array(handle);
  }

  bpn::array boost_ndarray_ravel(const bpn::array &array) {
    PyObject *flat = PyArray_Ravel((PyArrayObject *)array.ptr(), NPY_ANYORDER);
    return boost_ndarray_from_pyarray(flat);
  }

  template <typename T>
  bp::object boost_ndarray_from_data(int nd,
                                              npy_intp *shape,
                                              T *data) {
    PyObject *pyObj = PyArray_SimpleNewFromData(
        nd, shape, numpy_typenum<T>(), data);
    return boost_ndarray_from_pyarray(pyObj).copy(); // copy the object. numpy owns the copy now.
  }

  template <typename T>
  bp::object boost_ndarray_from_vector(std::vector<T> v) {
    npy_intp shape[] = { v.size() };
    T *data = v.size() ? &v[0] : NULL;
    return boost_ndarray_from_data(1, shape, data);
  }

  std::string greet() { return "hello, world"; }

  bp::object square(int number) {
    std::vector<double> s;
    s.push_back(number);
    s.push_back(number * number);
    return boost_ndarray_from_vector(s);
  }

  float sum(bpn::array array) {
    bpn::array flat = boost_ndarray_ravel(array);
    PyArrayObject *pflat = (PyArrayObject *)flat.ptr();

    double *data = (double *)PyArray_DATA(pflat);
    int n = PyArray_DIMS(pflat)[0];
    double s = 0;
    for (int i = 0; i < n; ++i) {
      s += data[i];
    }
    return s;
  }
}

BOOST_PYTHON_MODULE(csfm) {
  using namespace boost::python;
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  import_array();

  // Add regular functions to the module.
  def("greet", greet);
  def("square", square);
  def("sum", sum);
}
