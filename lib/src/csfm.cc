
#include <Python.h> // This must be included before anything else. See http://bugs.python.org/issue10910
#include <boost/python.hpp>
#include <string>
#include <vector>
#include <iostream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h> 

#include "train_vocabulary.cc"

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

  template <typename T>
  bp::object bpn_array_from_data(int nd, npy_intp *shape, const T *data) {
    PyObject *pyarray = PyArray_SimpleNewFromData(
        nd, shape, numpy_typenum<T>(), (void *)data);
    bp::handle<> handle(pyarray);
    return bpn::array(handle).copy(); // copy the object. numpy owns the copy now.
  }

  template <typename T>
  bp::object bpn_array_from_vector(const std::vector<T> &v) {
    npy_intp shape[] = { v.size() };
    const T *data = v.size() ? &v[0] : NULL;
    return bpn_array_from_data(1, shape, data);
  }

  template <typename T>
  bp::object bpn_array_from_cvmat(const cv::Mat &m) {
    npy_intp shape[] = { m.rows, m.cols };
    const T *data = m.rows ? m.ptr<T>(0) : NULL;
    return bpn_array_from_data(2, shape, data);
  }

  class PyArrayCvMatView {
   public:
    PyArrayCvMatView(bpn::array array) {
      init((PyArrayObject *)array.ptr());
    }

    PyArrayCvMatView(PyArrayObject *py_array) {
      init(py_array);
    }

    ~PyArrayCvMatView() {
      Py_DECREF(flat_);
    }

    const cv::Mat &get() const {
      return cvmat_;
    }

   private:
    void init(PyArrayObject *py_array) {
      npy_intp *shape = PyArray_DIMS(py_array);
      flat_ = (PyArrayObject *)PyArray_Ravel(py_array, NPY_ANYORDER);
      cvmat_ = cv::Mat(shape[0], shape[1], CV_32F, PyArray_DATA(flat_));
    };

    cv::Mat cvmat_;
    PyArrayObject *flat_;
  };



  bp::object approximate_k_means(bpn::array points,
                                 int k, 
                                 int max_iterations,
                                 float tolerance,
                                 int num_kdtrees,
                                 int kdtree_checks) {
    PyArrayCvMatView cv_points(points);
    cv::Mat centers;
    std::vector<int> labels;

    ApproximateKMeansParams params(max_iterations,
                                   tolerance,
                                   num_kdtrees,
                                   kdtree_checks);
    ApproximateKMeans(cv_points.get(), k, params, &centers, &labels);

    bp::list retn;
    retn.append(bpn_array_from_cvmat<float>(centers));
    retn.append(bpn_array_from_vector(labels));
    return retn;
  }
}

BOOST_PYTHON_MODULE(csfm) {
  using namespace boost::python;
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  import_array();

  // Add regular functions to the module.
  def("approximate_k_means", approximate_k_means,
      (arg("max_iterations") = 100,
       arg("tolerance") = 1e-6,
       arg("num_kdtrees") = 4,
       arg("kdtree_check") = 32
      )
  );
}
