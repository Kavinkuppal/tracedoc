# Third-party notices

## Apache ECharts

`src/tracedoc/report/vendor/echarts.min.js` is Apache ECharts (version in
`src/tracedoc/report/vendor/ECHARTS_VERSION`), inlined into every HTML report.

- License: Apache License 2.0, full text in `src/tracedoc/report/vendor/ECHARTS_LICENSE.txt`
- Notice: `src/tracedoc/report/vendor/ECHARTS_NOTICE.txt`
  (Apache ECharts, Copyright 2017-2026 The Apache Software Foundation. This product includes
  software developed at The Apache Software Foundation, https://www.apache.org/.)
- ECharts embeds parts of d3.js under the BSD 3-Clause license:
  `src/tracedoc/report/vendor/ECHARTS_LICENSE-d3.txt`

Generated reports credit Apache ECharts and its license in their footer.

## Inter

`src/tracedoc/report/fonts/Inter-*.ttf` are the static TrueType fonts of Inter (version in
`src/tracedoc/report/fonts/INTER_VERSION`), from https://github.com/rsms/inter, Copyright (c)
2016 The Inter Project Authors. They are embedded (subset) in every PDF report.

- License: SIL Open Font License 1.1, full text in `src/tracedoc/report/fonts/Inter-OFL.txt`

## ReportLab

The PDF reports are drawn with ReportLab (https://www.reportlab.com/opensource/), installed as
a dependency, not bundled. License: BSD.
