# Lesson 2: A policy is just numbers

## intro

Lesson two. The walker's brain is a small neural network, and once you see it up close, it's just nine hundred and thirty two numbers and a little arithmetic.

## neuron

Start with one neuron. It takes the inputs, multiplies each one by its own weight, and adds them up. Then it adds one more number, called the bias.

## neuron.2

The sum goes through a curve called tanh. Small values pass almost unchanged, and big values get squashed to plus or minus one. Motor commands have to stay between minus one and one, and tanh guarantees that.

## neuron.3

A positive weight means this input pushes the neuron up. A negative weight means it pushes it down. So a neuron can learn to fire when the body tilts forward and one foot is in the air.

## network

The walker's network has three layers. On the left, the twenty four inputs. In the middle, thirty two hidden neurons, and each one looks at every input. On the right, four outputs, one per motor.

## network.2

Here it is running on real observations from a walking episode. Every step, the inputs change, and the signal flows through to four motor commands. The network has no memory, so it decides each step from this snapshot alone.

## hidden

Why the hidden layer? Without it, every motor command would just be a weighted sum of the inputs. That can say, push harder the more you tilt. It can't say, if this foot is down and I'm tilting forward, swing the other leg.

## hidden.2

Hidden neurons learn combinations like that. In this project, at least one hidden layer is a hard rule, and the code checks it.

## count

Let's count the numbers. Twenty four inputs times thirty two hidden neurons gives seven hundred and sixty eight weights. Plus thirty two biases. Then thirty two times four output weights, which is one hundred and twenty eight. Plus four more biases. That's nine hundred and thirty two in total.

## theta

Write all of them in one long list, always in the same order. That list is called theta, and it is the walker. Save it to a file, and you've saved the whole walker.

## theta.2

Add a little random noise to theta, and you get a slightly different walker. The outputs shift a bit. That's exactly how the training algorithm makes new candidates.

## space

Now imagine a space with nine hundred and thirty two axes. Every possible walker is one point in it. Give each point a height: its average score on a few tracks. Training means finding a high point in that landscape.

## space.2

The catch is that we can measure the height anywhere, but nobody tells us which way is uphill. The physics can't be differentiated, falls are sudden, and every track is different. In lesson three, we climb a landscape we can only sample.
