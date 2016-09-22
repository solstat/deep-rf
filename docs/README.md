## Quick-Start

* Install `sphinx` using `pip`
* Follow the Google style guide for docstrings
* Call `sphinx-apidoc` on the directory with our python modules
to generate '.rst' files
* Call `make html` within the 'docs' folder to build 'html' documentation 

See below for more details.

## Docstring Style Guide
There are many different docstring style guidelines. 

We choose to follow the Google style guide for readability:
* [Google style guide](https://google.github.io/styleguide/pyguide.html)
* [Sphinx example module](http://www.sphinx-doc.org/en/stable/ext/example_google.html)

## Auto-Doc Generation with `sphinx`
To create 'html' and 'pdf' documentation of our module, we use [`sphinx`](http://www.sphinx-doc.org/en/stable/tutorial.html).
Sphinx is a utility for generating python documentation that uses reStructured text (rst) files. 

### Installation
Install `sphinx` using `pip`
```
$ pip install Sphinx
```

### Setting Up `sphinx`
To setup the documentation sources, first switch to the `docs` directory of the project. Then call `sphinx-quickstart`
```
$ mkdir docs
$ cd docs
$ sphinx-quickstart
```
Be sure to say yes to the "autodoc" extension. It is also recommend to separate the `source` and `build` directories.

`sphinx-quickstart` should generate
* `Makefile` - a makefile for building the documention (i.e. call `$ make html$` to build html documentation)
* `conf.py` - a configuration file for `sphinx`. Edit this to change settings and add extensions
* `index.rst` - the home page for documentation

To get "autodoc" to recognize the Google style, enable `sphinx.ext.napoleon` by editing `conf.py`. 
See [sphinx support for Google style docstrings](http://www.sphinx-doc.org/en/stable/ext/napoleon.html) for more details.

### Adding Module Documentation
To add documentation, we can either modify `rst` files manually or use `sphinx-apidoc` ([example_link](https://codeandchaos.wordpress.com/2012/07/30/sphinx-autodoc-tutorial-for-dummies/), [apidoc documentation](http://www.sphinx-doc.org/en/stable/man/sphinx-apidoc.html)) .

I recommend using `sphinx-apidoc` to generate the `rst` files and then modify them manually.
To create documentation call
```
$ sphinx-apidoc -o <outputdir> <sourcedir>
``` 
where `<sourcedir>` is the path to the module folder (e.g. `src/`) and `<outputdir>` is `docs/source`. 
Note that `sphinx-apidoc` will import each module in `<sourcedir>`.

Useful `sphinx-apidoc` options (`--separate` for separate pages and `--private` to include all functions)


### Documentation Output Formatting
To change the html output style, change sthe `html_theme` value in `conf.py`.
(http://www.sphinx-doc.org/en/stable/theming.html)



