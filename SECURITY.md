# Security

## Reporting a security problem

Please don't report security problems in a public issue. Report them privately through
GitHub: on the repository's **Security** tab, choose **Report a vulnerability**, or go
straight to https://github.com/Kavinkuppal/tracedoc/security/advisories/new. Only the
maintainers can see the report.

Please include:

- what the problem is and what an attacker could do with it;
- the steps to reproduce it;
- your tracedoc version (`tracedoc --version`), Python version and operating system.

Don't send real API keys or private trace data. If a key was exposed, revoke it with its
provider first.

We will reply to your report and keep you informed while we fix it. Please give us a chance
to release a fix before you disclose the problem publicly.

## What counts

tracedoc runs on your machine and reads traces that can hold your users' data. Examples of
security problems:

- data leaving your machine other than in calls to the model provider you configured;
- an API key or other secret written to a report, cache or log file;
- redaction missing data it is meant to remove;
- text from a trace that runs as code when you open a report (for example, a script in a
  trace that the HTML report doesn't escape).

A finding that is wrong, or a suggested fix you disagree with, is not a security problem:
please open an issue with the "False finding" form instead.

## Supported versions

Security fixes are made on the `main` branch, for the latest version.
