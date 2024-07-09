# Jevo

[![Build Status](https://github.com/jarbus/Jevo.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/jarbus/Jevo.jl/actions/workflows/CI.yml?query=branch%3Amaster) 
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://jarbus.github.io/Jevo.jl/dev/)

Jevo is a high-performance, distributed, and modular (co-)evolutionary algorithm framework written in Julia. It is designed to be flexible and easy to use, with a focus on deep neuroevolutionary applications using [Flux.jl](https://fluxml.ai/Flux.jl/stable/). Jevo is designed to be easy to use, with a simple API that allows users to quickly define custom evolutionary algorithms and run them on distributed systems.


| **Warning:** Jevo is currently alpha software and is under active development. |
| ----------------------------------------- |

# Install

This package requires modified versions of Transformers.jl and NeuralAttentionlib.jl, which are unregistered. In addition, it depends on a custom plotting library (XPlot.jl) & a library for phylogenies (PhylogeneticTrees.jl), both unregistered as well. To install all dependencies, run the following command in the environment of your choice:

```julia
julia ./install.jl
```
