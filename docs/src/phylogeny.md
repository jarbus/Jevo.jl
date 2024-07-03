# Phylogeny

## Design of phylo

- we store phylo tree and delta cache with population, because those are NOT LRU
- we store genotype cache and weight cache outside of population because those ARE LRU

## Design of gene pool
