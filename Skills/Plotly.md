Valid properties:
    bgcolor
        Sets the color of padded area.
    bordercolor
        Sets the axis line color.
    borderwidth
        Sets the width (in px) or the border enclosing this
        color bar.
    dtick
        Sets the step in-between ticks on this axis. Use with
        `tick0`. Must be a positive number, or special strings
        available to "log" and "date" axes. If the axis `type`
        is "log", then ticks are set every 10^(n*dtick) where n
        is the tick number. For example, to set a tick mark at
        1, 10, 100, 1000, ... set dtick to 1. To set tick marks
        at 1, 100, 10000, ... set dtick to 2. To set tick marks
        at 1, 5, 25, 125, 625, 3125, ... set dtick to
        log_10(5), or 0.69897000433. "log" has several special
        values; "L<f>", where `f` is a positive number, gives
        ticks linearly spaced in value (but not position). For
        example `tick0` = 0.1, `dtick` = "L0.5" will put ticks
        at 0.1, 0.6, 1.1, 1.6 etc. To show powers of 10 plus
        small digits between, use "D1" (all digits) or "D2"
        (only 2 and 5). `tick0` is ignored for "D1" and "D2".
        If the axis `type` is "date", then you must convert the
        time to milliseconds. For example, to set the interval
        between ticks to one day, set `dtick` to 86400000.0.
        "date" also has special values "M<n>" gives ticks
        spaced by a number of months. `n` must be a positive
        integer. To set ticks on the 15th of every third month,
        set `tick0` to "2000-01-15" and `dtick` to "M3". To set
        ticks every 4 years, set `dtick` to "M48"
    exponentformat
        Determines a formatting rule for the tick exponents.
        For example, consider the number 1,000,000,000. If
        "none", it appears as 1,000,000,000. If "e", 1e+9. If
        "E", 1E+9. If "power", 1x10^9 (with 9 in a super
        script). If "SI", 1G. If "B", 1B. "SI" uses prefixes
        from "femto" f (10^-15) to "tera" T (10^12). *SI
        extended* covers instead the full SI range from
        "quecto" q (10^-30) to "quetta" Q (10^30). If "SI" or
        *SI extended* is used and the exponent is beyond the
        above ranges, the formatting rule will automatically be
        switched to the power notation.
    labelalias
        Replacement text for specific tick or hover labels. For
        example using {US: 'USA', CA: 'Canada'} changes US to
        USA and CA to Canada. The labels we would have shown
        must match the keys exactly, after adding any
        tickprefix or ticksuffix. For negative numbers the
        minus sign symbol used (U+2212) is wider than the
        regular ascii dash. That means you need to use −1
        instead of -1. labelalias can be used with any axis
        type, and both keys (if needed) and values (if desired)
        can include html-like tags or MathJax.
    len
        Sets the length of the color bar This measure excludes
        the padding of both ends. That is, the color bar length
        is this length minus the padding on both ends.
    lenmode
        Determines whether this color bar's length (i.e. the
        measure in the color variation direction) is set in
        units of plot "fraction" or in *pixels. Use `len` to
        set the value.
    minexponent
        Hide SI prefix for 10^n if |n| is below this number.
        This only has an effect when `tickformat` is "SI" or
        "B".
    nticks
        Specifies the maximum number of ticks for the
        particular axis. The actual number of ticks will be
        chosen automatically to be less than or equal to
        `nticks`. Has an effect only if `tickmode` is set to
        "auto".
    orientation
        Sets the orientation of the colorbar.
    outlinecolor
        Sets the axis line color.
    outlinewidth
        Sets the width (in px) of the axis line.
    separatethousands
        If "true", even 4-digit integers are separated
    showexponent
        If "all", all exponents are shown besides their
        significands. If "first", only the exponent of the
        first tick is shown. If "last", only the exponent of
        the last tick is shown. If "none", no exponents appear.
    showticklabels
        Determines whether or not the tick labels are drawn.
    showtickprefix
        If "all", all tick labels are displayed with a prefix.
        If "first", only the first tick is displayed with a
        prefix. If "last", only the last tick is displayed with
        a suffix. If "none", tick prefixes are hidden.
    showticksuffix
        Same as `showtickprefix` but for tick suffixes.
    thickness
        Sets the thickness of the color bar This measure
        excludes the size of the padding, ticks and labels.
    thicknessmode
        Determines whether this color bar's thickness (i.e. the
        measure in the constant color direction) is set in
        units of plot "fraction" or in "pixels". Use
        `thickness` to set the value.
    tick0
        Sets the placement of the first tick on this axis. Use
        with `dtick`. If the axis `type` is "log", then you
        must take the log of your starting tick (e.g. to set
        the starting tick to 100, set the `tick0` to 2) except
        when `dtick`=*L<f>* (see `dtick` for more info). If the
        axis `type` is "date", it should be a date string, like
        date data. If the axis `type` is "category", it should
        be a number, using the scale where each category is
        assigned a serial number from zero in the order it
        appears.
    tickangle
        Sets the angle of the tick labels with respect to the
        horizontal. For example, a `tickangle` of -90 draws the
        tick labels vertically.
    tickcolor
        Sets the tick color.
    tickfont
        Sets the color bar's tick label font
    tickformat
        Sets the tick label formatting rule using d3 formatting
        mini-languages which are very similar to those in
        Python. For numbers, see:
        https://github.com/d3/d3-format/tree/v1.4.5#d3-format.
        And for dates see: https://github.com/d3/d3-time-
        format/tree/v2.2.3#locale_format. We add two items to
        d3's date formatter: "%h" for half of the year as a
        decimal number as well as "%{n}f" for fractional
        seconds with n digits. For example, *2016-10-13
        09:15:23.456* with tickformat "%H~%M~%S.%2f" would
        display "09~15~23.46"
    tickformatstops
        A tuple of :class:`plotly.graph_objects.heatmap.colorba
        r.Tickformatstop` instances or dicts with compatible
        properties
    tickformatstopdefaults
        When used in a template (as layout.template.data.heatma
        p.colorbar.tickformatstopdefaults), sets the default
        property values to use for elements of
        heatmap.colorbar.tickformatstops
    ticklabeloverflow
        Determines how we handle tick labels that would
        overflow either the graph div or the domain of the
        axis. The default value for inside tick labels is *hide
        past domain*. In other cases the default is *hide past
        div*.
    ticklabelposition
        Determines where tick labels are drawn relative to the
        ticks. Left and right options are used when
        `orientation` is "h", top and bottom when `orientation`
        is "v".
    ticklabelstep
        Sets the spacing between tick labels as compared to the
        spacing between ticks. A value of 1 (default) means
        each tick gets a label. A value of 2 means shows every
        2nd label. A larger value n means only every nth tick
        is labeled. `tick0` determines which labels are shown.
        Not implemented for axes with `type` "log" or
        "multicategory", or when `tickmode` is "array".
    ticklen
        Sets the tick length (in px).
    tickmode
        Sets the tick mode for this axis. If "auto", the number
        of ticks is set via `nticks`. If "linear", the
        placement of the ticks is determined by a starting
        position `tick0` and a tick step `dtick` ("linear" is
        the default value if `tick0` and `dtick` are provided).
        If "array", the placement of the ticks is set via
        `tickvals` and the tick text is `ticktext`. ("array" is
        the default value if `tickvals` is provided).
    tickprefix
        Sets a tick label prefix.
    ticks
        Determines whether ticks are drawn or not. If "", this
        axis' ticks are not drawn. If "outside" ("inside"),
        this axis' are drawn outside (inside) the axis lines.
    ticksuffix
        Sets a tick label suffix.
    ticktext
        Sets the text displayed at the ticks position via
        `tickvals`. Only has an effect if `tickmode` is set to
        "array". Used with `tickvals`.
    ticktextsrc
        Sets the source reference on Chart Studio Cloud for
        `ticktext`.
    tickvals
        Sets the values at which ticks on this axis appear.
        Only has an effect if `tickmode` is set to "array".
        Used with `ticktext`.
    tickvalssrc
        Sets the source reference on Chart Studio Cloud for
        `tickvals`.
    tickwidth
        Sets the tick width (in px).
    title
        :class:`plotly.graph_objects.heatmap.colorbar.Title`
        instance or dict with compatible properties
    x
        Sets the x position with respect to `xref` of the color
        bar (in plot fraction). When `xref` is "paper",
        defaults to 1.02 when `orientation` is "v" and 0.5 when
        `orientation` is "h". When `xref` is "container",
        defaults to 1 when `orientation` is "v" and 0.5 when
        `orientation` is "h". Must be between 0 and 1 if `xref`
        is "container" and between "-2" and 3 if `xref` is
        "paper".
    xanchor
        Sets this color bar's horizontal position anchor. This
        anchor binds the `x` position to the "left", "center"
        or "right" of the color bar. Defaults to "left" when
        `orientation` is "v" and "center" when `orientation` is
        "h".
    xpad
        Sets the amount of padding (in px) along the x
        direction.
    xref
        Sets the container `x` refers to. "container" spans the
        entire `width` of the plot. "paper" refers to the width
        of the plotting area only.
    y
        Sets the y position with respect to `yref` of the color
        bar (in plot fraction). When `yref` is "paper",
        defaults to 0.5 when `orientation` is "v" and 1.02 when
        `orientation` is "h". When `yref` is "container",
        defaults to 0.5 when `orientation` is "v" and 1 when
        `orientation` is "h". Must be between 0 and 1 if `yref`
        is "container" and between "-2" and 3 if `yref` is
        "paper".
    yanchor
        Sets this color bar's vertical position anchor This
        anchor binds the `y` position to the "top", "middle" or
        "bottom" of the color bar. Defaults to "middle" when
        `orientation` is "v" and "bottom" when `orientation` is
        "h".
    ypad
        Sets the amount of padding (in px) along the y
        direction.
    yref
        Sets the container `y` refers to. "container" spans the
        entire `height` of the plot. "paper" refers to the
        height of the plotting area only.



        st.plotly_chart
Streamlit Version
Version 1.52.0
Display an interactive Plotly chart.

Plotly is a charting library for Python. The arguments to this function closely follow the ones for Plotly's plot() function.

To show Plotly charts in Streamlit, call st.plotly_chart wherever you would call Plotly's py.plot or py.iplot.

Important

You must install plotly>=4.0.0 to use this command. Your app's performance may be enhanced by installing orjson as well. You can install all charting dependencies (except Bokeh) as an extra with Streamlit:

pip install streamlit[charts]
Copy
Function signature[source]
st.plotly_chart(figure_or_data, use_container_width=None, *, width="stretch", height="content", theme="streamlit", key=None, on_select="ignore", selection_mode=('points', 'box', 'lasso'), config=None, **kwargs)

Parameters
figure_or_data (plotly.graph_objs.Figure, plotly.graph_objs.Data, or dict/list of plotly.graph_objs.Figure/Data)

The Plotly Figure or Data object to render. See https://plot.ly/python/ for examples of graph descriptions.

Note

If your chart contains more than 1000 data points, Plotly will use a WebGL renderer to display the chart. Different browsers have different limits on the number of WebGL contexts per page. If you have multiple WebGL contexts on a page, you may need to switch to SVG rendering mode. You can do this by setting render_mode="svg" within the figure. For example, the following code defines a Plotly Express line chart that will render in SVG mode when passed to st.plotly_chart: px.line(df, x="x", y="y", render_mode="svg").

width ("stretch", "content", or int)

The width of the chart element. This can be one of the following:

"stretch" (default): The width of the element matches the width of the parent container.
"content": The width of the element matches the width of its content, but doesn't exceed the width of the parent container.
An integer specifying the width in pixels: The element has a fixed width. If the specified width is greater than the width of the parent container, the width of the element matches the width of the parent container.
height ("content", "stretch", or int)

The height of the chart element. This can be one of the following:

"content" (default): The height of the element matches the height of its content.
"stretch": The height of the element matches the height of its content or the height of the parent container, whichever is larger. If the element is not in a parent container, the height of the element matches the height of its content.
An integer specifying the height in pixels: The element has a fixed height. If the content is larger than the specified height, scrolling is enabled.
use_container_width (bool or None)

delete
use_container_width is deprecated and will be removed in a future release. For use_container_width=True, use width="stretch".

Whether to override the figure's native width with the width of the parent container. This can be one of the following:

None (default): Streamlit will use the value of width.
True: Streamlit sets the width of the figure to match the width of the parent container.
False: Streamlit sets the width of the figure to fit its contents according to the plotting library, up to the width of the parent container.
theme ("streamlit" or None)

The theme of the chart. If theme is "streamlit" (default), Streamlit uses its own design default. If theme is None, Streamlit falls back to the default behavior of the library.

The "streamlit" theme can be partially customized through the configuration options theme.chartCategoricalColors and theme.chartSequentialColors. Font configuration options are also applied.

key (str)

An optional string to use for giving this element a stable identity. If key is None (default), this element's identity will be determined based on the values of the other parameters.

Additionally, if selections are activated and key is provided, Streamlit will register the key in Session State to store the selection state. The selection state is read-only.

on_select ("ignore" or "rerun" or callable)

How the figure should respond to user selection events. This controls whether or not the figure behaves like an input widget. on_select can be one of the following:

"ignore" (default): Streamlit will not react to any selection events in the chart. The figure will not behave like an input widget.
"rerun": Streamlit will rerun the app when the user selects data in the chart. In this case, st.plotly_chart will return the selection data as a dictionary.
A callable: Streamlit will rerun the app and execute the callable as a callback function before the rest of the app. In this case, st.plotly_chart will return the selection data as a dictionary.
selection_mode ("points", "box", "lasso" or an Iterable of these)

The selection mode of the chart. This can be one of the following:

"points": The chart will allow selections based on individual data points.
"box": The chart will allow selections based on rectangular areas.
"lasso": The chart will allow selections based on freeform areas.
An Iterable of the above options: The chart will allow selections based on the modes specified.
All selections modes are activated by default.

config (dict or None)

A dictionary of Plotly configuration options. This is passed to Plotly's show() function. For more information about Plotly configuration options, see Plotly's documentation on Configuration in Python.

**kwargs (null)

delete
**kwargs are deprecated and will be removed in a future release. Use config instead.

Additional arguments accepted by Plotly's plot() function.

This supports config, a dictionary of Plotly configuration options. For more information about Plotly configuration options, see Plotly's documentation on Configuration in Python.

Returns
(element or dict)

If on_select is "ignore" (default), this command returns an internal placeholder for the chart element. Otherwise, this command returns a dictionary-like object that supports both key and attribute notation. The attributes are described by the PlotlyState dictionary schema.

Examples
Example 1: Basic Plotly chart

The example below comes from the examples at https://plot.ly/python. Note that plotly.figure_factory requires scipy to run.

import plotly.figure_factory as ff
import streamlit as st
from numpy.random import default_rng as rng

hist_data = [
    rng(0).standard_normal(200) - 2,
    rng(1).standard_normal(200),
    rng(2).standard_normal(200) + 2,
]
group_labels = ["Group 1", "Group 2", "Group 3"]

fig = ff.create_distplot(
    hist_data, group_labels, bin_size=[0.1, 0.25, 0.5]
)

st.plotly_chart(fig)
Copy

Built with Streamlit 🎈
Fullscreen
open_in_new
Example 2: Plotly Chart with configuration

By default, Plotly charts have scroll zoom enabled. If you have a longer page and want to avoid conflicts between page scrolling and zooming, you can use Plotly's configuration options to disable scroll zoom. In the following example, scroll zoom is disabled, but the zoom buttons are still enabled in the modebar.

import plotly.graph_objects as go
import streamlit as st

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=[1, 2, 3, 4, 5],
        y=[1, 3, 2, 5, 4]
    )
)

st.plotly_chart(fig, config = {'scrollZoom': False})
Copy

Built with Streamlit 🎈
Fullscreen
open_in_new
Chart selections
PlotlyState
Streamlit Version
Version 1.52.0
The schema for the Plotly chart event state.

The event state is stored in a dictionary-like object that supports both key and attribute notation. Event states cannot be programmatically changed or set through Session State.

Only selection events are supported at this time.

Attributes
selection (dict)

The state of the on_select event. This attribute returns a dictionary-like object that supports both key and attribute notation. The attributes are described by the PlotlySelectionState dictionary schema.

Example
Try selecting points by any of the three available methods (direct click, box, or lasso). The current selection state is available through Session State or as the output of the chart function.

import plotly.express as px
import streamlit as st

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length")

event = st.plotly_chart(fig, key="iris", on_select="rerun")

event
Copy

Built with Streamlit 🎈
Fullscreen
open_in_new
PlotlySelectionState
Streamlit Version
Version 1.52.0
The schema for the Plotly chart selection state.

The selection state is stored in a dictionary-like object that supports both key and attribute notation. Selection states cannot be programmatically changed or set through Session State.

Attributes
points (list[dict[str, Any]])

The selected data points in the chart, including the data points selected by the box and lasso mode. The data includes the values associated to each point and a point index used to populate point_indices. If additional information has been assigned to your points, such as size or legend group, this is also included.

point_indices (list[int])

The numerical indices of all selected data points in the chart. The details of each identified point are included in points.

box (list[dict[str, Any]])

The metadata related to the box selection. This includes the coordinates of the selected area.

lasso (list[dict[str, Any]])

The metadata related to the lasso selection. This includes the coordinates of the selected area.

Example
When working with more complicated graphs, the points attribute displays additional information. Try selecting points in the following example:

import plotly.express as px
import streamlit as st

df = px.data.iris()
fig = px.scatter(
    df,
    x="sepal_width",
    y="sepal_length",
    color="species",
    size="petal_length",
    hover_data=["petal_width"],
)

event = st.plotly_chart(fig, key="iris", on_select="rerun")

event.selection
Copy

Built with Streamlit 🎈
Fullscreen
open_in_new
This is an example of the selection state when selecting a single point:

{
  "points": [
    {
      "curve_number": 2,
      "point_number": 9,
      "point_index": 9,
      "x": 3.6,
      "y": 7.2,
      "customdata": [
        2.5
      ],
      "marker_size": 6.1,
      "legendgroup": "virginica"
    }
  ],
  "point_indices": [
    9
  ],
  "box": [],
  "lasso": []
}
Copy
Theming
Plotly charts are displayed using the Streamlit theme by default. This theme is sleek, user-friendly, and incorporates Streamlit's color palette. The added benefit is that your charts better integrate with the rest of your app's design.

The Streamlit theme is available from Streamlit 1.16.0 through the theme="streamlit" keyword argument. To disable it, and use Plotly's native theme, use theme=None instead.

Let's look at an example of charts with the Streamlit theme and the native Plotly theme:

import plotly.express as px
import streamlit as st

df = px.data.gapminder()

fig = px.scatter(
    df.query("year==2007"),
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    hover_name="country",
    log_x=True,
    size_max=60,
)

tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
with tab1:
    # Use the Streamlit theme.
    # This is the default. So you can also omit the theme argument.
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab2:
    # Use the native Plotly theme.
    st.plotly_chart(fig, theme=None, use_container_width=True)



If you're wondering if your own customizations will still be taken into account, don't worry! You can still make changes to your chart configurations. In other words, although we now enable the Streamlit theme by default, you can overwrite it with custom colors or fonts. For example, if you want a chart line to be green instead of the default red, you can do it!

Here's an example of an Plotly chart where a custom color scale is defined and reflected:

import plotly.express as px
import streamlit as st

st.subheader("Define a custom colorscale")
df = px.data.iris()
fig = px.scatter(
    df,
    x="sepal_width",
    y="sepal_length",
    color="sepal_length",
    color_continuous_scale="reds",
)

tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
with tab1:
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab2:
    st.plotly_chart(fig, theme=None, use_container_width=True)
Notice how the custom color scale is still reflected in the chart, even when the Streamlit theme is enabled 

