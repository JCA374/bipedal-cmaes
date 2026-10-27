# Lesson 3: Evolution strategies and CMA-ES

## intro

Lesson three. How do you climb a landscape you can't see, when all you can do is stand somewhere and measure the height? You throw a cloud of guesses, keep the best ones, and learn the shape of the cloud as you go.

## loop

Here's a landscape in two dimensions. Green is better, and the cross marks the top. We start with one guess, the mean, in orange.

## loop.2

First, sample a cloud of candidates around the mean. The size of the cloud is the step size, called sigma.

## loop.3

Then evaluate every candidate, rank them, and keep the best half, shown in green.

## loop.4

Move the mean to a weighted average of the winners, with the best ones counting most. That's one generation. Repeat.

## loop.5

Only the ranking matters, not the exact scores. One terrible score can't drag the search off course. That makes the method robust.

## compare

Now two versions side by side, on a long, tilted valley. On the right, a plain strategy, with a fixed, round cloud. On the left, CMA-ES.

## compare.2

Watch the left cloud stretch and turn until it lines up with the valley. The plain one keeps sampling a circle, wastes most of its guesses on the valley walls, and zig zags.

## sigma

CMA-ES also adapts its step size. It keeps a running sum of its recent moves. If they point the same way, it's on a long slope, and sigma grows. If they cancel out, it's circling around a peak, and sigma shrinks.

## sigma.2

On this plot, the error of CMA-ES drops by more than thirty orders of magnitude, while the plain version gets stuck near the bottom of the valley.

## cov

The C in CMA stands for covariance. It's the shape of the cloud. In two dimensions, it's an ellipse. For the walker, it's a table with almost nine hundred thousand numbers, one for every pair of weights.

## cov.2

It learns which weights should move together. If raising a hip weight only helps when a knee weight rises too, CMA-ES starts proposing both at once.

## pop

The number of candidates matters too. On a bumpy landscape full of small traps, a small population can get caught in one. A bigger population sees more of the landscape at once and finds the real peak.

## code

For the walker, the whole loop is a few lines. Each generation asks for forty eight candidate walkers. Every candidate walks the same five random tracks. Then we hand back the scores, and CMA-ES updates its mean, its step size and its covariance.

## why

Why not gradients? Methods like PPO and TD3 estimate a gradient from individual steps. They use fewer samples, but they need many more moving parts. An evolution strategy only looks at whole episodes, which is simple and runs in parallel. The price is samples. Next, in lesson four: what happens when every measurement is noisy.
