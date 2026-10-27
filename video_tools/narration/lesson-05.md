# Lesson 6: The training run

## intro

Lesson six. CMA-ES climbs whatever you tell it to climb. My first three attempts climbed the wrong hill. The walker learned to dive, to lunge, and to fall on purpose. Here's why, and what finally worked.

## shaping

First, a distinction. The score we report is fixed: the raw reward over full episodes. But the fitness that training climbs is ours to design.

## shaping.2

Two choices matter. Episodes are cut short when the walker stops moving, which saves a lot of time. And the price of a fall, or of a stall, can be changed for training only.

## traps

Attempt one used the real fall penalty of a hundred, plus a stall penalty of a hundred. Standing still scored worse than falling. So the walker learned to dive forward and fall, and every dive scored about the same. There was no slope left to climb.

## traps.2

Attempt two made falls free, but kept the stall penalty. Now the natural next step, lunge, catch yourself, and stand, was cut off by the stall detector and punished. So the walker lunged and fell.

## traps.3

Attempt three made both free, and it still fell on purpose. The reason was motor effort. A random network pushes its motors hard, and every step alive costs points. Falling early simply ended the bill.

## traps.4

Attempt four set the fall cost to twenty. That's more than the effort a fall saves, but far less than the hundred that made my old NEAT walker afraid to step. Standing became the safe choice, and from standing, every step forward paid.

## breakthrough

For about fifty four generations, nothing seemed to happen. Candidates learned to stand and shuffle.

## breakthrough.2

Then, within a few generations, the best candidate jumped from about zero to over a hundred. The first real steps. By generation fifty nine, the champion scored over a hundred on the held out tracks.

## stages

Its first positive score and over one hundred, both at generation fifty nine.

## stages.2

Over two hundred, at generation one hundred and nine.

## curve

Here's the whole run: five hundred and thirty nine generations and about one hundred and eighteen million simulation steps, in three hours on this laptop. The orange line is the best training fitness. The blue line is the champion on held out tracks.

## curve.2

Notice the step size, sigma. It stayed large through the breakthrough and only shrank later, as the walker settled into its gait. Progress came in bursts, not in a smooth line.

## outro

The lesson: a good fitness makes every small improvement on the way to walking score a little better than the last. Next, we open up the champion and look inside.
