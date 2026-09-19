*****
xlink
*****

The sphinxcontrib-xlink extension allows you to create links to other documents in your documentation. It provides a simple syntax for creating links, and it also allows you to create links to specific sections of other documents.

It generates support files for the auto-complete feature of Visual Studio Code, which allows you to quickly find and link to other documents in your documentation.

Configuration
=============

To use the :code:`sphinxcontrib-xlink` extension, you need to add it to the list of extensions in your :file:`conf.py` file:

.. code-block:: python
   :caption: conf.py options for sphinxcontrib-xlink

    extensions = [
        ...
        "sphinxcontrib.xlink",
        ...
    ]

Creating xlinks
===============

A basic xlink entry is a simple entry in a plain text file written on a single line and looks like this:

.. code:: text

   unique-id :: Document Title :: link :: tag1, tag2, tag3:subtag1, tag3:subtag2

With these entries, you can create links to other documents in your documentation. The unique-id is a unique identifier for the link, the Document Title is the title of the document you want to link to, and the link is the URL or path to the document. The tags are optional and can be used to categorize the links.

The tags can also be used to filter the links when generating lists of links in your documentation. For example, you can generate a list of all links that have the tag "documentation" or "diagram". This allows you to organize your links and make it easier for readers to find related documents.

Using xlinks
============

You can use xlinks, with the ``xlink`` role, to create links in your documentation. The syntax for using xlinks follows the format of a standard link in reStructuredText, but with the exception, that the link target is the unique-id of the xlink entry you want to link to. For example, if you have an xlink entry with the unique-id "rstPrimer", you can create a link to it like this: 

.. code:: rst

   :xlink:`rstPrimer`

Which will render to :xlink:`rstPrimer`. 

Without any additional text, this will create a link with the text "reStructuredText Primer", which is given as the title in the xlink file. You can also specify the link text by using the standard reStructuredText link syntax, like this:

.. code:: rst

   :xlink:`Link Text Override <xlink:rstPrimer>`

Which will render to :xlink:`Link Text Override <rstPrimer>`.

Listing xlinks
==============

xlinks can also be listed in your documentation using the ``xlink-list`` directive. This directive allows you to generate a list of links based on the tags specified in the xlink entries. For example, if you want to generate a list of all links that are saved in the `fundamentals.xlink` file, you can use the following directive:

.. code:: rst

   .. xlink-list::
      :file: fundamentals.xlink

Which will render like this:

.. xlink-list::
   :files: fundamentals

Filtering xlinks
================

xinks can also be filtered by various criteria when generating the list of links. For example, you can filter the links based on their tags, url patter, title name or you can filter the links based on their unique-id field. This allows you to generate lists of links that are relevant to a specific topic or category.

Tag-based xlink filters
=======================

You can filter xlinks based on their tags when generating the list of links. For example, if you want to generate a list of all links that have the tag "rst", you can use the following directive:

.. code:: rst

   .. xlink-list::
      :tags: rst

Which will render like this:

.. xlink-list::
   :tags: rst

Combined Filtering
==================

You can also combine e.g. the file and tag filters to generate a list of links that are relevant to a specific topic or category. For example, if you want to generate a list of all links that are saved in the `fundamentals.xlink` file and have the tag "rst", you can use the following directive:

.. code:: rst

   .. xlink-list::
      :files: fundamentals.xlink
      :tags: rst

Which will render like this:

.. xlink-list::
   :files: fundamentals
   :tags: rst

Using the query engine
======================

xlinks can also be filtered using a query engine, which allows you to filter the links based on more complex criteria. For example, you can filter the links based on their title, url pattern, or unique-id field. This allows you to generate lists of links that are relevant to a specific topic or category, even if they do not have specific tags. For example, if you want to generate a list of all links that have the word "primer" in their title, you can use the following directive:

.. code:: rst

   .. xlink-list::
      :query: re.search("um", title, re.IGNORECASE)

Which will render like this:

.. xlink-list::
   :query: re.search("um", title, re.IGNORECASE)