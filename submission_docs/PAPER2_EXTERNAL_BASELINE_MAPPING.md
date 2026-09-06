# BigMEC Greedy Adapter Mapping

External source:

- Brandherm et al., "BigMEC: Scalable Service Migration for Mobile Edge Computing," IEEE/ACM SEC 2022. DOI: 10.1109/SEC54971.2022.00018.
- Public software: `https://github.com/flbrandh/MEC-Simulator-2-BigMEC`
- Audited revision: `a586ae3b98119f83bb7883bd94ea95b1fbed7a8c`
- Archived software DOI: 10.5281/zenodo.10810301.

The adapter implements the public non-learning, non-displacing, highest-utility
greedy contract rather than claiming bit-for-bit execution of the original
Python simulator. The mapping is explicit:

| BigMEC concept | Paper 2 adapter |
|---|---|
| service | persistent DNN task |
| cloud neighborhood | all five edge nodes (the complete small neighborhood) |
| equal service priority | unit priority for every task |
| latency utility | negative `Pre.Comm + Pre.Comp` |
| available cloud memory | projected DNN memory no greater than node capacity |
| mobility-triggered event | every slot, because every task receives a mobility update |
| no service displacement | no task is evicted while proposing a move |
| simulator feasibility enforcement | the same final Safe Harbor repair used for every method |

This is reported as a code-derived BigMEC greedy adapter, not as a reproduction
of the complete BigMEC learning system or the original San Francisco topology.
