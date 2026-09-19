.. index:: Tables, Basic Elements; Tables

******
Tables
******

Sphinx provides a powerful way to create tables in your documentation. You can use the ``table`` directive to create tables with various formatting options. Here are some examples of how to create tables in Sphinx.

Simple Table
============

.. code-block:: rst

   .. table:: This is a simple table

      =====  =====  =======
      A      B      A and B
      =====  =====  =======
      False  False  False
      True   False  False
      False  True   False
      True   True   True
      =====  =====  =======

This will create a simple table with three columns and four rows. The first row is the header row, and the second row is the separator row. The remaining rows are the data rows.

Simple tables are limited: they must contain more than one row, and the first column cells cannot contain multiple lines.

.. table:: This is a simple table

   =====  =====  =======
   A      B      A and B
   =====  =====  =======
   False  False  False
   True   False  False
   False  True   False
   True   True   True
   =====  =====  =======

Grid Table
==========

.. code-block:: rst

   .. table:: This is a grid table

      +--------------+--------------+
      | Header 1     | Header 2     |
      +==============+==============+
      | Row 1, Col 1 | Row 1, Col 2 |
      +--------------+--------------+
      | Row 2, Col 1 | Row 2, Col 2 |
      +--------------+--------------+

This will create a grid table with two columns and three rows. The first row is the header row, and the remaining rows are the data rows. Grid tables allow for more complex formatting, such as multi-line cells and cells that span multiple columns or rows.

.. table:: This is a grid table

   +--------------+--------------+
   | Header 1     | Header 2     |
   +==============+==============+
   | Row 1, Col 1 | Row 1, Col 2 |
   +--------------+--------------+
   | Row 2, Col 1 | Row 2, Col 2 |
   +--------------+--------------+

CSV Table
=========

You can create tables from CSV data using the ``csv-table`` directive. This allows you to easily create tables from data that is already in CSV format. You can work with ``internal CSV data``, which is baked directly into the document or use ``external CSV files``, where the CSV data is stored in a separate file (which is usually the preferred approach).

Internal CSV Data
-----------------

Here's an example of how to create a table from internal CSV data:

.. code-block:: rst

   .. csv-table:: Frozen Delights!
      :header: "Treat", "Quantity", "Description"
      :widths: 15, 10, 30

      "Albatross", 2.99, "On a stick!"
      "Crunchy Frog", 1.49, "If we took the bones out,
      it wouldn't be crunchy, now would it?"
      "Gannet Ripple", 1.99, "On a stick!"

This will create a table from CSV data. The ``csv-table`` directive allows you to specify the header row and the widths of the columns. The data rows are specified in CSV format, with each row on a new line. This is a convenient way to create tables from data that is already in CSV format.

.. csv-table:: Frozen Delights!
   :header: "Treat", "Quantity", "Description"
   :widths: 15, 10, 30

   "Albatross", 2.99, "On a stick!"
   "Crunchy Frog", 1.49, "If we took the bones out,
   it wouldn't be crunchy, now would it?"
   "Gannet Ripple", 1.99, "On a stick!"

External CSV Files
------------------

You can also create tables from external CSV files. Here's an example of how to do that if the header row is included in the CSV file:

.. code-block:: rst

   .. csv-table:: Frozen Delights!
      :header-rows: 1
      :widths: 15, 10, 30
      :file: table-data/frozen_delights.csv

This will create the following table from the data in the CSV file:

.. csv-table:: Frozen Delights!
   :header-rows: 1
   :widths: 15, 10, 30
   :file: table-data/frozen_delights.csv

.. note:: If the header row is not included in the CSV file, you can specify the header row using the ``:header:`` option, as shown in the |xlink-ref-internal-csv-data| section, like this:

   .. code:: rst

      .. csv-table:: Frozen Delights!
         :header: "Treat", "Quantity", "Description"
         :widths: 15, 10, 30
         :file: table-data/frozen_delights_no_header.csv

List Table
==========

The "list-table" directive is used to create a table from data in a uniform two-level bullet list. "Uniform" means that each sublist (second-level list) must contain the same number of list items.

.. code-block:: rst

   .. list-table:: Frozen Delights!
      :widths: 15 10 30
      :header-rows: 1

      * - Treat
      - Quantity
      - Description
      * - Albatross
      - 2.99
      - On a stick!
      * - Crunchy Frog
      - 1.49
      - If we took the bones out, it wouldn't be
         crunchy, now would it?
      * - Gannet Ripple
      - 1.99
      - On a stick!

This will create the following table from the data in the list:

.. list-table:: Frozen Delights!
   :widths: 15 10 30
   :header-rows: 1

   * - Treat
     - Quantity
     - Description
   * - Albatross
     - 2.99
     - On a stick!
   * - Crunchy Frog
     - 1.49
     - If we took the bones out, it wouldn't be
       crunchy, now would it?
   * - Gannet Ripple
     - 1.99
     - On a stick!

Table Edge Cases
================

With these tests we want to make sure that the table directive works in all edge cases. The following tests are included:

* Table following a section title
* Long table that spans multiple pages
* Long and wide table that spans multiple pages

Table Following a Section Title
-------------------------------

.. csv-table:: Frozen Delights!
   :header-rows: 1
   :widths: 15, 10, 30
   :file: table-data/frozen_delights.csv

Long Table
----------

This is an example of a long table that spans multiple pages.

.. csv-table:: Personal Information
   :header-rows: 1
   :file: table-data/long-table-data.csv

Long and Wide Table
-------------------

This is an example of a long and wide table that spans multiple pages.

.. csv-table:: Personal Information with Details
   :header-rows: 1
   :file: table-data/long-wide-table-data.csv

