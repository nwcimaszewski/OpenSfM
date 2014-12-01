from distutils.core import setup, Extension
import numpy

# define the name of the extension to use
extension_name    = 'greet'
extension_version = '1.0'
 
# define the directories to search for include files
# to get this to work, you may need to include the path
# to your boost installation. Mine was in 
# '/usr/local/include', hence the corresponding entry.
include_dirs = [ '/usr/local/include', 'include', numpy.get_include() ]
 
# define the library directories to include any extra
# libraries that may be needed.  The boost::python
# library for me was located in '/usr/local/lib'
library_dirs = [ '/usr/local/lib' ]
 
# define the libraries to link with the boost python library
libraries = [ 'boost_python-mt' ]
 
# define the source files for the extension
source_files = [ 'lib/src/greet.cc' ]
 
# create the extension and add it to the python distribution
setup(name=extension_name,
	  version=extension_version,
	  ext_modules=[
	  	Extension(extension_name,
	  		source_files,
	  		include_dirs=include_dirs,
	  		library_dirs=library_dirs,
	  		libraries=libraries
	  		)
	  	]
	  )
