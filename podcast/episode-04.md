# Episode 4: Cheaper steps, a bigger brain, and the winner's curse

This is episode four, recorded late on Wednesday evening. In the last episode, the walker had just learned to walk with real steps, one foot in front of the other, and it scored about two hundred and sixty four points. Since then, I've been trying to make it faster and cheaper. Here's what happened.

First, the problem. Real steps cost energy. Every swing of a leg past the other one takes motor effort, and the score pays for distance but charges for effort. Our walker was spending about ninety nine points of effort per run. The old skipping walker spent only seventy five. That's why the skipper still scored about ten points more.

So the first new experiment made effort more expensive during training, by half again. The real score stayed the same, but in training every unit of motor command cost one and a half times as much. It worked. Within a few hours, the walker cut its effort to about eighty four, and on twenty test tracks it scored about two hundred and seventy nine, without a single fall.

You also told me I could change any part of the model. So the second experiment gave the walker a bigger brain. Its controller had thirty two hidden neurons. I doubled that to sixty four, with a trick that keeps everything it had already learned. The new neurons start switched off, so the bigger network walks exactly like the old one, to the decimal point. Then evolution learns how to use them.

On its own, the bigger network didn't help much. It ended at about two hundred and fifty on the test tracks, with three falls. But combining the two ideas did. The third experiment took the cheaper-steps walker and gave it the bigger brain. And that's our best walker so far. On twenty test tracks it scores about two hundred and eighty eight, with no falls at all. Its effort is down to seventy five, the same as the old skipper. It reaches the finish in about nine hundred steps. And it really walks, changing its leading foot about a hundred and ten times per run. So now the real walker beats the skipper at its own game.

There was one more lesson, and it's an old one. When I checked the champions on tracks they had never been chosen on, some of them were worse than they looked. One champion looked better during training, but fell twice on the test tracks, while the walker it replaced had no falls at all. That's the winner's curse from lesson four. When you pick the best of many on only a few tracks, you often pick the luckiest. At this level, falls are rare, so a few tracks can't tell a safe walker from a lucky one.

So the newest experiment, started an hour ago, chooses more carefully. Each candidate walks ten tracks instead of five, and each champion has to prove itself on forty tracks instead of twenty. It's slower, but the walkers it picks should hold up on new tracks.

From now on, I'll record an episode whenever something big happens, and I've packaged the whole process, from writing to publishing on the site, so that each episode is made the same careful way. Thanks for listening.
