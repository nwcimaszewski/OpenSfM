
#include <Python.h> // This must be included before anything else. See http://bugs.python.org/issue10910
#include <boost/python.hpp>
#include <string>
using namespace boost::python;

namespace { // Avoid cluttering the global namespace.
  // A couple of simple C++ functions that we want to expose to Python.
  std::string greet() { return "hello, world"; }
  int square(int number) { return number * number; }
}

BOOST_PYTHON_MODULE(csfm)
{
  // Add regular functions to the module.
  def("greet", greet);
  def("square", square);
}
