# Lesson 7: Inside the champion

## intro

Lesson seven. The champion is nine hundred and thirty two numbers. What do they do? Let's open the network while it walks.

## heat

Each row is one of the thirty two hidden neurons, and time runs to the right. Blue is positive, orange is negative. At the bottom, you can see when each foot is on the ground.

## heat.2

Look for repeating stripes. Many neurons switch in step with the feet. And about fifty seven percent of the time, a neuron sits at its limit, fully on or fully off. Many of them act like switches.

## gait

Now cut the walk into strides, from one touchdown of the first foot to the next, and average. A stride takes about fifteen steps, which is roughly three hundred milliseconds. It's a quick, short gait.

## gait.2

The network has no memory, but it still produces a rhythm. The rhythm lives in the loop between body and network. The legs move, their angles feed back as inputs, and the next command follows.

## ablation

Which neurons matter? To find out, freeze one at its average value and let the walker try twenty tracks. This is called ablation.

## ablation.2

Most neurons barely matter. But freeze neuron twenty five, and the score drops by three hundred and fifty points. The walker falls on eighteen of twenty tracks.

## ablation.3

In this network, nine of the thirty two neurons each cost more than fifty points when frozen. The rest is spare capacity, or fine tuning.

## inputs

The same test works for the senses. The most important ones are hip 1 angle, knee 1 angle, hip 1 speed.

## inputs.2

And the lidar, the rays that look ahead, barely matters. This walker walks by feel, reacting to its own joints. That's enough on these gentle tracks, but it's a likely weakness on unusual bumps.

## outro

Next, the final test: a hundred tracks it has never seen, and the comparison with my other walkers.
