# A6: MIMII-DG all 5 machines (breadth)
Status: completed. results/exp010_mimii_results.json (all 5). Conditioning helps 4/5 machines
both domains: bearing(speed) .609->.649/.623->.645; gearbox(voltage) .748->.809/.645->.710;
slider(velocity) .726->.789/.622->.662; valve(pattern) .690->.734/.376->.632 (target below-chance
recovered). Exception: fan(noise-mix, categorical, NOT an operating covariate) .955->.943/.716->.670
(hurts). Honest scope: conditioning helps where the attribute is a genuine operating covariate.
Folded into paper Table 5 (all 5 machines) + scope paragraph.
