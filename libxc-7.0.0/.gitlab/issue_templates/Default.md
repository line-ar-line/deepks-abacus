IMPORTANT: this bug tracker is for the Libxc source distribution.
Libxc is a volunteer-run project. To eliminate noise in the bug tracker,
we have collected a list of common issues and solutions below.

1. 99.999% of users don't use Libxc directly, but instead use Libxc through an electronic structure package. If you are a end user and your electronic structure calculation fails with an error in Libxc, this is often due to misuse of Libxc in the package. File a bug with the package. If the maintainers confirm the issue really is in Libxc, they should open up an issue here.
2. If you encounter a problem with a binary package of Libxc, file a bug with the binary package. We do not have any control over pip, conda, or *nix packages of Libxc. We only maintain the source distribution.
3. If you have an issues compiling or running Libxc from the source code on your local system, this is usually caused by having more than one installation of Libxc and an inconsistent environment. It is perfectly normal to get compile errors or missing symbols, if you compile a code for one version of a library and try to run it against another version. Double check your environment and see 

If you have successfully passed all the checks above, feel free to report the bug.
Please include all relevant information.

- What is the bug? Summarize the bug encountered concisely.
- How did you compile Libxc?
- What version(s) of Libxc is affected?
- Detail exactly how one can reproduce the issue; this is very important. If you can, link to the line of code that might be responsible for the problem.

