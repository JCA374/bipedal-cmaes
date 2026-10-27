# Lesson 5: The reward function

## intro

Lesson five. The walker never sees our goal. It sees one number per step, and it learns whatever makes the total of those numbers large. So let's read that number, line by line.

## code

This is the reward, straight from the environment's source code. It runs fifty times per second.

## code.2

It has four parts. Progress: every unit the body moves forward earns about four points. Effort: every unit of motor command costs a little. Tilt: changing the body's angle costs a little. And a fall replaces everything with minus one hundred and ends the episode. There's no bonus for finishing.

## budget

Here's where a walker's score comes from. My old NEAT walker earns about two hundred and eighty from moving, and spends only about ninety on effort. But it runs out of time before the finish.

## budget.2

The CMA-ES walker at the two hundred milestone earns the full distance, about three hundred and sixty, but spends about a hundred and sixty on effort. Two walkers, two different weaknesses, and the reward turns each into one number. Later in training, CMA-ES cut that effort by half.

## notpaid

Just as important is what the reward does not pay for. Not footsteps: a shuffle and a stride over the same distance earn the same. Not style: feet passing each other aren't worth anything. Not staying alive, and not finishing.

## notpaid.2

But every step costs effort, and the clock runs out after sixteen hundred steps. So the reward quietly favours speed, even though time never appears in it.

## proxy

That makes the reward a proxy. We want a walker that walks well. The reward asks it to move right, spend little, and not touch the ground. An optimizer can't tell the difference, and it finds the gaps.

## proxy.2

In this project it found several: diving forward, falling early to stop paying for effort, standing still, and skipping with one leg as a prop. Each was a good answer to the question the reward actually asked.

## outro

So the reward decides what gets learned. Next, in lesson six, we see how the training run used it, including three false starts.
