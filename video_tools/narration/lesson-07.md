# Lesson 7: Results and comparison

## intro

Lesson seven, the honest test. A hundred tracks the walker has never seen, full episodes, and the unmodified reward.

## final

The champion averages two hundred and five points, give or take seventeen. It falls on one of a hundred tracks, and reaches the finish on sixty nine.

## final.2

So it's not solved. Three hundred is the bar, and none of a hundred runs reached it. But look at how consistent it is: almost every track lands close to the average.

## budget

Where do the points go? Walking to the finish earns about three hundred and sixty. On average, this walker earns three hundred and sixty one from moving, and spends one hundred and fifty six on motor effort.

## budget.2

So the limit isn't falling, and it isn't speed. It's effort. The network pushes its motors almost flat out. To score higher, it would have to walk as far while working less hard.

## neat

Now the same hundred tracks for my NEAT walker. NEAT averages one hundred and eighty. CMA-ES averages two hundred and three. The difference is twenty three points, and CMA-ES scores higher on sixty three of the hundred tracks.

## neat.2

It also falls far less: one fall against ten. A fixed network with the right size, plus covariance learning, beat evolving the network's shape.

## table

How does it compare with gradient methods? PPO reached two hundred and seventy two. TD3 reached about three hundred and ten, in only two hours.

## table.2

CMA-ES used far more simulation steps: about sixteen million. But it needed no gradients, no critic and no replay buffer, and its whole policy is about a hundred and forty times smaller than TD3's.

## outro

That's the whole story: a task, a network of nine hundred and thirty two numbers, an algorithm that only ranks whole episodes, and a fitness that leads it, one small step at a time, to walking.
