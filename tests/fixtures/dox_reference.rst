Reference
*********

Directive options reference
===========================

The following table summarises every option accepted by the ``.. roadmap::`` directive.  Per-directive options always override the corresponding global ``doxtr_roadmap_*`` config value.

.. list-table::
   :header-rows: 1
   :widths: 30 28 42

   * - Option
     - Accepted values / type
     - Description
   * - ``:file:``
     - path(s) / glob(s)
     - One or more CSV paths and/or glob patterns, space- or comma-separated.  All matched files are combined into a single roadmap: sections merge by name and subtasks / ``row_group`` references work across files.  Glob metacharacters (``*``, ``?``, ``[…]``) are expanded; within each glob the matches are sorted lexicographically.  A single path with no separators behaves as before.  Paths are resolved relative to the document directory first, then ``srcdir``.  Adding a new file matching a glob may require a clean rebuild (see :ref:`roadmap-multiple-files-and-globs`).
   * - ``:title:``
     - text
     - Override the diagram title.  Falls back to ``doxtr_roadmap_default_title`` (``"Roadmap"`` by default).
   * - ``:scale:``
     - ``daily`` | ``weekly`` | ``monthly``
     - Timeline granularity.  Defaults to ``doxtr_roadmap_default_scale`` (``"monthly"``).
   * - ``:start:``
     - ISO date (``YYYY-MM-DD``)
     - Clip-window start date.  Tasks ending before this date are hidden.
   * - ``:end:``
     - ISO date (``YYYY-MM-DD``)
     - Clip-window end date.  Tasks starting after this date are hidden.
   * - ``:period:``
     - comma-separated period names
     - Named period(s) to zoom to.  When a single period is given, scale auto-switches to ``daily`` and weekends are closed.
   * - ``:close-weekends:``
     - flag (no value)
     - Force weekend columns closed regardless of the active scale.
   * - ``:clean-style:``
     - ``true`` | ``false``; bare flag = ``true``
     - Hide the Start, End, and Duration columns.  Overrides ``doxtr_roadmap_clean_style`` for this chart.
   * - ``:tags:``
     - nested filter expression
     - Tag-based row filter (``eng``, ``!security``, ``eng, ops``, etc.).
   * - ``:query:``
     - safe Python expression
     - Expression-based row filter evaluated per row.  Available names: ``name``, ``start``, ``end``, ``section``, ``tags``, ``row_group``, ``match(pattern, string)``.
   * - ``:collision-detection:``
     - ``true`` | ``false``; bare flag = ``true``
     - Enable or disable automatic collision detection for same-row tasks.  Overrides ``doxtr_roadmap_collision_detection``.
   * - ``:collision-char-width-factor:``
     - float (e.g. ``0.7``)
     - Label-width tuning: higher values split sooner, lower values pack more tightly.  Overrides ``doxtr_roadmap_collision_char_width_factor``.
   * - ``:column-zoom:``
     - float (e.g. ``2``)
     - Gantt time-column width multiplier.  ``1`` = unchanged (default).  Overrides ``doxtr_roadmap_column_zoom``.
   * - ``:width:``
     - length or ``%`` (e.g. ``100%``, ``600px``)
     - Force the rendered image to fill the specified width in HTML and PDF.  Passed directly to ``sphinxcontrib.plantuml``.
   * - ``:link-appendix:``
     - ``list`` | ``footnote`` | ``off``; bare flag = ``list``
     - Render task links as real docutils nodes below the chart.  ``list`` = bullet list; ``footnote`` = real RST auto-numbered footnotes (LaTeX ``\footnote`` in PDF, numbered footnotes in HTML); ``off`` = disable.  Overrides ``doxtr_roadmap_link_appendix``.
   * - ``:link-appendix-title:``
     - text
     - Heading text above the link appendix.  Empty string suppresses the heading.  Overrides ``doxtr_roadmap_link_appendix_title``.
   * - ``:caption:``
     - text
     - Caption text for the roadmap chart.  When present, wraps the chart in a ``nodes.figure`` so it is numbered and appears in the List of Figures.  Enable ``numfig = True`` in ``conf.py`` for "Figure N" numbering and ``:numref:`` cross-references.
   * - ``:align:``
     - ``left`` | ``center`` | ``right``
     - Horizontal alignment of the figure.  Also triggers figure wrapping even without ``:caption:``.
   * - ``:name:``
     - text
     - Cross-reference target for the figure (use with ``:numref:`` or ``:ref:``).  Attach only to a figure that has a caption or align.


Configuration reference
=======================

The following table lists all ``doxtr_roadmap_*`` config values with their defaults.  Set any of them in :file:`conf.py`.

.. list-table::
   :header-rows: 1
   :widths: 42 16 42

   * - Config key
     - Default
     - Description
   * - ``doxtr_roadmap_default_scale``
     - ``"monthly"``
     - Default timeline unit: ``daily``, ``weekly``, or ``monthly``.
   * - ``doxtr_roadmap_scale_factor``
     - ``1.25``
     - PlantUML top-level ``scale`` zoom factor.
   * - ``doxtr_roadmap_default_start``
     - ``None``
     - Default clip-window start date (ISO string).  ``None`` falls back to today.
   * - ``doxtr_roadmap_default_title``
     - ``"Roadmap"``
     - Diagram title when ``:title:`` is not set.
   * - ``doxtr_roadmap_clean_style``
     - ``True``
     - Hide Start/End/Duration columns.  Requires PlantUML v1.2026.7+.
   * - ``doxtr_roadmap_close_weekends_on_single_period``
     - ``True``
     - Auto-close weekends when exactly one period is rendered.
   * - ``doxtr_roadmap_bar``
     - see above
     - Global bar colours: ``done_color``, ``undone_color``, ``frame_color``.
   * - ``doxtr_roadmap_sections``
     - ``{}``
     - Per-section colour overrides.
   * - ``doxtr_roadmap_today``
     - ``{"color": "#E53935"}``
     - Today-marker colour.
   * - ``doxtr_roadmap_fonts``
     - see above
     - Font name/size/style/color for ``title``, ``task``, ``separator``, ``month``, and ``year`` elements.
   * - ``doxtr_roadmap_closed``
     - ``{"background_color": None}``
     - Background colour for closed (non-working) days.
   * - ``doxtr_roadmap_use_theme_core``
     - ``"auto"``
     - Theme-core integration: ``"auto"``, ``True``, or ``False``.
   * - ``doxtr_roadmap_allowed_tags``
     - ``{}``
     - Optional tag allow-list mapping tag names to display strings.
   * - ``doxtr_roadmap_allowed_tag_patterns``
     - ``{}``
     - Optional regex patterns for allowed tag names.
   * - ``doxtr_roadmap_require_plantuml_version``
     - ``"error"``
     - Version-check behaviour: ``"error"`` aborts the build if PlantUML is below v1.2026.7; ``"warn"`` emits a warning but continues; ``"off"`` or ``False`` skips the check entirely.
   * - ``doxtr_roadmap_collision_detection``
     - ``True``
     - Enable automatic collision detection for ``row_group`` same-row tasks.  When ``True`` (default), tasks whose bars or labels overlap are split onto additional rows.  Set to ``False`` to force all same-group tasks onto one row regardless of overlap.
   * - ``doxtr_roadmap_collision_char_width_factor``
     - ``1.0``
     - Label-width tuning knob for collision detection.  Higher values reserve more horizontal space per label character, causing splits sooner.  Lower values pack tasks more tightly.
   * - ``doxtr_roadmap_collision_gap_days``
     - ``2``
     - Minimum calendar-day gap between adjacent tasks on the same lane after accounting for the left task's label.
   * - ``doxtr_roadmap_column_zoom``
     - ``1``
     - Gantt time-column width multiplier.  ``1`` (default) leaves column widths unchanged.  Higher values (e.g. ``3``) widen each time column proportionally via ``projectscale monthly zoom 3``.  Requires PlantUML v1.2026.7+ (already the extension minimum).
   * - ``doxtr_roadmap_link_appendix``
     - ``"list"``
     - Render task links as real docutils nodes below the chart image.  ``"list"`` (default) produces a bullet list; ``"footnote"`` produces real RST auto-numbered footnotes (rendered as page-bottom ``\footnote{}`` in PDF/LaTeX, numbered footnotes in HTML); ``False`` / ``"off"`` disables the feature.  The ``link_appendix_builders`` setting restricts *which* builders render it (default PDF/latex only).
   * - ``doxtr_roadmap_link_appendix_builders``
     - ``["latex"]``
     - Builder names/formats that render the appendix.  Checked against both ``builder.name`` and ``builder.format``.  Use ``"all"`` (or ``["*"]``) for every builder.  Default ``["latex"]`` means PDF gets the appendix; HTML/epub don't unless opted in.
   * - ``doxtr_roadmap_link_appendix_title``
     - ``"Links"``
     - Heading text rendered above the link appendix list.  Set to an empty string or ``None`` to suppress the heading.
   * - ``doxtr_roadmap_figure``
     - ``False``
     - When ``True``, every ``.. roadmap::`` is wrapped as a numbered figure automatically, even without a per-directive ``:caption:``.  The caption defaults to the chart title (or ``doxtr_roadmap_figure_caption`` when set).  Enable ``numfig = True`` in ``conf.py`` for ``Figure N`` numbering and ``:numref:`` cross-references.
   * - ``doxtr_roadmap_figure_caption``
     - ``None``
     - Optional default caption text used when ``doxtr_roadmap_figure = True`` and no per-directive ``:caption:`` is given.  ``None`` (default) falls back to the chart title.  Caption precedence: explicit ``:caption:`` option → ``doxtr_roadmap_figure_caption`` (if non-empty) → chart title.
