# EE102 Signals and Systems LaTeX Project

This repository contains all course materials for EE102: Signals and Systems, including problem sets, solutions, and lecture notes.

## Structure
- `template.tex`: Main LaTeX template for assignments and notes.
- `ee102.cls`, `ee102.sty`: Custom class and style files for EE102.
- `macros.tex`: Macro file for course meta information (semester, year, instructor, etc.).
- `solutions/`: Submodule for solutions (see below).

## Customization
- Update `macros.tex` with your semester, year, and instructor name.

## Solutions Submodule
The `solutions/` folder is a git submodule pointing to a private repository. To initialize and sync it, run:

```sh
git submodule update --init --recursive
```

## Getting Started
1. Clone this repository.
2. Initialize submodules: `git submodule update --init --recursive`
3. Compile `template.tex` using your preferred LaTeX editor.

## License
This repository is for educational use in EE102 at UC Merced.
