### MEEP Functions

To avoid a forced dependency on MEEP, the code for `connectivity_constraint()`
and various filters have been included here.

This code has also been modified to avoid `autograd` as FDTDx is
`jax` exclusively. Ever reference to `npa` (or similar) has been
replaced with the adjoint helper dynamic stand in `ah.anp`
