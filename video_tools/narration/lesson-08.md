# Lesson 9: Two ways to walk

## intro

Lesson nine. The champion scores well, but it never puts one foot in front of the other. Let's look at why, and test a network built to take turns.

## skip

Watch the feet. The line below shows which foot is in front. Above the line, foot one leads. Below it, foot two leads. A normal walk crosses the line on every step.

## skip.2

This walker never crosses it. On a hundred tracks, foot two leads one hundred percent of the time. Its touchdowns still alternate, so it isn't hopping. It's skipping, with one leg always reaching forward.

## why

Why did evolution keep this gait? The reward pays for distance, not for style, and the skip earns well. To switch, the walker would have to pass through clumsy in between gaits that score worse. CMA-ES searches nearby, so it stays where it is.

## fix

So instead of asking nicely, change the network. One controller now drives both legs. Each leg sees the body, its own leg, the other leg and the lidar. Leg two uses the same weights, with the legs swapped. The hope: with only one set of weights, neither leg can become a permanent prop.

## fix.2

The controller also gets a clock, a rhythm whose speed it learns. Leg two's clock runs half a cycle behind. With no shift, both legs would move together, which is a hop. With half a cycle, they take turns.

## fix.3

On top, a small training bonus for every step where the front foot changes and the body moves forward. Reported scores stay the raw reward.

## results

The result. The new walker never changes its leading foot either. Foot two leads one hundred percent of the time. It still skips.

## results.2

On the same hundred tracks, it scores two hundred and thirty six, against two hundred and seventy for the skipping walker. That's significantly lower, by thirty four points. It falls on eleven tracks, against seven.

## results.3

Why? Same weights doesn't mean same movement. Each leg's controller also sees where the other leg is, so one set of weights can say: if I'm in front, reach. If I'm behind, push. The legs settle into fixed roles. The clock still matters: freeze it, and this walker can't move at all. It just doesn't force the legs to swap.

## outro

The lesson: evolution finds what you reward, the nearest way, even through a structure built to prevent it. If you care how it's done, you have to close every loophole.
