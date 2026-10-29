Development-phase scripts (seeds 0-9). Run from the repository root, e.g.
  PYTHONPATH=. python dev/explore.py logreg 1,3 10 '{}' '{"fixed":{"dp":"fixed"},"bona":{"dp":"bona"}}'
  PYTHONPATH=. python dev/grid.py 1 8
Their logs are in results/dev/ and underlie Table 7 (negative results) and Fig. 7.
They were NOT used for any number in Tables 2-6, which use fresh seeds 100-119.
