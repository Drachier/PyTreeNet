PyTreeNet
=========

.. doc-inclusion-marker1-start

`PyTreeNet <https://github.com/Drachier/PyTreeNet>`_ is a Python implemention of tree tensor networks with a focus on the simulation of quantum systems admitting a tree topology. It is based on the array implementation of NumPy.

Main Features
--------
- General tree tensor networks, tree tensor networks states and tree tensor network operators
- Conversion of symbolic Hamiltonians into tree tensor networks
- Time-evolution algorithms: TEBD, one-site TDVP and two-site TDVP, BUG

Installation
------------
To install PyTeNet from PyPI, call

.. code-block:: python

    python3 -m pip install pytreenet

Alternatively, you can clone the `repository <https://github.com/Drachier/PyTreeNet>`_ and install it in development mode via

.. code-block:: python

    python3 -m pip install -e <path/to/repo>

.. doc-inclusion-marker1-end

Documentation
-------------
You can find the documentation of PyTreeNet at `Read the Docs <https://pytreenet.readthedocs.io/>`_.

Directory structure
-------------------
- **pytreenet**: Source code of PyTreeNet
- **examples**: Examples of PyTreeNet usage
- **docs**: Documentation of PyTreeNet
- **tests**: Unit tests of PyTreeNet functionality

.. doc-inclusion-marker2-start

Contributing
------------
You are welcome to contribute to PyTreeNet. For code contributions create a `pull request on GitHub <https://github.com/Drachier/PyTreeNet/pulls>`_. For bug reports and feature requests, please open an `issue on GitHub <https://github.com/Drachier/PyTreeNet/issues>`_. Please document code contributions using `Google style docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_ and unit-test it with tests placed in the *tests* folder.

Citing
------
If you use PyTreeNet in your work, please cite the accompanying paper `PyTreeNet: A Python Library for easy Utilisation of Tree Tensor Networks (arxiv:2407.13249) <https://arxiv.org/abs/2407.13249>`

License
-------
PyTreeNet is licensed under the `EUPL v1.2 <https://eupl.eu/1.2/en/>`_.


References
----------
1. | Y.-Y. Shi, L.-M. Duan, G. Vidal,
   | *Classical simulation of quantum many-body systems with a tree tensor network*
   | `Phys. Rev. A 74, 022320 (2006) <https://doi.org/10.1103/PhysRevA.74.022320>`_ (`arXiv:quant-ph/0511070 <https://arxiv.org/abs/quant-ph/0511070>`_)
2. | D. Bauernfeind, M. Aichhorn,
   | *Time dependent variational principle for tree Tensor Networks*
   | `SciPost Physics 8, 024 (2020) <https://doi.org/10.21468/SciPostPhys.8.2.024>`_ (`arXiv:1908.03090 <https://arxiv.org/abs/1908.03090>`_)
3. | P. Silvi, F. Tschirsich, M. Gerster, J. Jünemann, D. Jaschke, M. Rizzi, S. Montangero
   | *The Tensor Networks Anthology: Simulation techniques for many-body quantum lattice systems*
   | `SciPost Phys. Lect. Notes, 8 (2019) <https://doi.org/10.21468/SciPostPhysLectNotes.8>`_ (`arxiv:1710.03733 <https://arxiv.org/abs/1710.03733>`_)
4. | G. Ceruti, C. Lubich, D. Sulz,
   | *Rank-adaptive time integration of tree tensor networks*
   | `SIAM JNA, Vol. 6, Iss. 1, 194 - 222 (2023) <https://doi.org/10.1137/22M1473790>`_ (`arXiv:2201.10291 <https://arxiv.org/abs/2201.10291>`_)

.. doc-inclusion-marker2-end