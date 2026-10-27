# Bipedal CMA-ES: the explainer

## title

This network learned to walk without a single gradient. It's nine hundred and thirty two numbers, tuned by an evolution strategy called CMA-ES, on a four core laptop. Here's how.

## task

The task is Bipedal Walker. A body on two legs, four motors, and a bumpy track that's different every time. Every step, the walker senses twenty four numbers and sets four motor commands.

## task.2

Moving forward earns points, effort costs points, and a fall costs a hundred. Reaching the finish scores about three hundred. Averaged over a hundred tracks, three hundred counts as solved.

## network

The controller is a small neural network: twenty four inputs, thirty two hidden neurons, and four outputs. All its weights fit in one list of nine hundred and thirty two numbers. That list is the walker.

## loop

To improve it, CMA-ES samples a cloud of candidate walkers around its current best guess, lets each one walk, ranks them, and moves toward the winners.

## loop.2

It also learns the shape of the cloud, which weights should change together, and how big a step to take. No gradients, only rankings of whole episodes.

## noise

Every score is noisy, because every track is different. So all candidates in a generation walk the same five tracks, and the champion is chosen on twenty tracks that training never sees.

## traps

Attempt one used the real fall penalty of a hundred, plus a stall penalty of a hundred. Standing still scored worse than falling. So the walker learned to dive forward and fall, and every dive scored about the same. There was no slope left to climb.

## traps.4

Attempt four set the fall cost to twenty. That's more than the effort a fall saves, but far less than the hundred that made my old NEAT walker afraid to step. Standing became the safe choice, and from standing, every step forward paid.

## breakthrough.2

Then, within a few generations, the best candidate jumped from about zero to over a hundred. The first real steps. By generation fifty nine, the champion scored over a hundred on the held out tracks.

## stages

Its first positive score and over one hundred, both at generation fifty nine.

## stages.2

Over two hundred, at generation one hundred and nine.

## heat.2

Look for repeating stripes. Many neurons switch in step with the feet. And about forty six percent of the time, a neuron sits at its limit, fully on or fully off. Many of them act like switches.

## ablation.2

Most neurons barely matter. But freeze neuron twenty three, and the score drops by two hundred and eighty five points. The walker falls on ten of ten tracks.

## inputs.2

And the lidar, the rays that look ahead, barely matters. This walker walks by feel, reacting to its own joints. That's enough on these gentle tracks, but it's a likely weakness on unusual bumps.

## final

The champion averages two hundred and five points, give or take seventeen. It falls on one of a hundred tracks, and reaches the finish on sixty nine.

## final.2

So it's not solved. Three hundred is the bar, and none of a hundred runs reached it. But look at how consistent it is: almost every track lands close to the average.

## budget.2

So the limit isn't falling, and it isn't speed. It's effort. The network pushes its motors almost flat out. To score higher, it would have to walk as far while working less hard.

## neat

Now the same hundred tracks for my NEAT walker. NEAT averages one hundred and eighty. CMA-ES averages two hundred and three. The difference is twenty three points, and CMA-ES scores higher on sixty three of the hundred tracks.

## outro

The full lessons, with interactive demos for every idea in this video, are on the project page. Thanks for watching.
