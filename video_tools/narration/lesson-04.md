# Lesson 4: Noisy fitness, honest selection

## intro

Lesson four. A walker's score depends on the track it gets. So every time we measure fitness, we're also rolling dice. This lesson is about not fooling yourself.

## spread

Here's one walker, my previous champion, on a hundred different tracks. Same network, same weights. The scores spread out by about sixty points either way.

## spread.2

So a measurement is true skill plus luck. Averaging over more tracks shrinks the luck. Five tracks cut it roughly in half, and a hundred tracks cut it by a factor of ten.

## curse

Now pick a winner. Here are forty eight candidates. Their true skill is in grey, and what we measured on five tracks is in blue.

## curse.2

We pick the highest blue dot. It's probably a good candidate, but it's also the one that got the luckiest tracks. Its measured score is too high.

## curse.3

Repeat this many times, and the pattern is clear. With five tracks, the winner looks about thirty points better than it really is. With one track, about eighty. This is called the winner's curse.

## crn

Part of the luck is shared. A hard track is hard for everyone. So every candidate in a generation walks the same five tracks. Then a hard track lowers all the scores together, and it cancels out of the ranking.

## crn.2

This trick is called common random numbers. It costs nothing, and the ranking becomes fairer. And remember, CMA-ES only uses the ranking.

## champ

To pick the champion, every ten generations, we take the mean of the search cloud and the best three candidates. Each is tested on the same twenty held-out tracks, which training never sees. The best one becomes the champion.

## champ.2

The mean is often the most robust choice. It's the average of many good candidates, so tricks that only worked on lucky tracks tend to cancel out.

## claim

The final score comes from a hundred more tracks the walker has never seen, with full episodes and the unmodified reward. Every purpose has its own range of seeds, and the ranges never overlap.

## claim.2

This matters. My previous champion scored two hundred during selection. On a hundred new tracks, it scored about one seventy five. When we compare two walkers, both walk the same hundred tracks, and a difference smaller than about twenty points is usually noise.

## outro

That's the background. Next, the training run itself: how the walker went from falling over to walking, including three false starts.
