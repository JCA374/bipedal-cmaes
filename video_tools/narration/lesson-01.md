# Lesson 1: The task

## intro

This is lesson one. Before a network can learn to walk, we need to be exact about the task. What can the walker sense, what can it do, and what counts as doing well?

## walker

Meet the walker. It's a box-shaped body on two legs, in a small two dimensional physics world. Each leg has a hip and a knee, and every joint has a motor.

## walker.2

The ground is a bumpy grass track. It's generated from a random seed, so every episode gets a different track. The goal is to reach the finish, about eighty four units to the right.

## walker.3

The world runs at fifty steps per second. An episode ends when the body touches the ground, when the walker reaches the finish, or after sixteen hundred steps. That's thirty two seconds.

## senses

At every step, the walker gets twenty four numbers. It never sees a picture. Four numbers describe the body: its tilt, how fast it's turning, and how fast it moves forward and up.

## senses.2

Ten numbers describe the legs. For each leg, there's the angle and speed of the hip and the knee, and whether the foot is touching the ground.

## senses.3

The last ten come from a lidar. It's a fan of rays that measures how far away the ground is, from straight down to almost straight ahead. That's how the walker can notice a bump before it gets there.

## actions

What it can do is just as simple: four numbers between minus one and one, one per joint. The sign says which way to turn the joint. The size says how hard to push. That's the whole controller.

## reward

Now the score. Moving forward earns reward. Crossing the whole track is worth about three hundred and sixty points.

## reward.2

Every bit of motor effort costs a little, and so does tilting the body. A good walker spends thirty to sixty points on effort over a run.

## reward.3

And falling is expensive. If the body hits the ground, the walker loses a hundred points, and the episode is over.

## reward.4

So a walker that reaches the finish without falling scores about three hundred. Averaged over a hundred random tracks, three hundred counts as solved.

## time

There's a hidden catch: time. Eighty four units in sixteen hundred steps means an average speed of about two point six units per second.

## time.2

My previous walker, evolved with a method called NEAT, rarely fell. But it was too slow. It ran out of time at about seventy four units and scored about one seventy five. To reach three hundred, a walker has to be safe and fast at the same time.

## outro

So the job is a function from twenty four numbers to four numbers that earns as much reward as possible, on tracks it has never seen. In lesson two, we look at what that function is: a small neural network.
