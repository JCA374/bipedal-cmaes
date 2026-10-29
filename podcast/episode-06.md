# Episode 6: Safety pays

This is episode six, recorded at about three in the morning on Thursday. Last time I had to correct myself: the faster walkers were falling on more than one track in ten. This time the news is good, and I've checked it properly.

The fix was simple. During training, a fall now costs twice as much as in the real game. The walker still learns from the real reward for distance and effort, but it now treats falling as a much bigger deal. I started this run from the fast walker that kept falling.

Here's how I checked it. Both walkers, the new one and the one it started from, walked the same hundred tracks. The old one scored about two hundred and sixty three, and fell on fourteen of them. The new one scored about two hundred and eighty five, and fell on only two. That's a gain of about twenty two points, and the statistics say it's real: even the cautious end of the estimate is a gain of about eight. It was better on sixty five of the hundred tracks.

And it kept its speed. It still finishes in about nine hundred steps. It still walks with real steps, changing its leading foot more than a hundred times per run. And its effort stayed at about seventy five, the same as the old skipping walker. On those same hundred tracks, the skipper scores about two hundred and seventy. So our walker now beats the skipper by about fifteen points, with a proper gait and fewer falls.

Two other experiments failed, and both taught me something.

The careful-selection run, the one that tested each champion on forty tracks, settled down without becoming safer. It still fell nine times in fifty. Choosing more carefully doesn't help if what you're choosing between are all risky walkers. You have to make safety part of the goal.

And I tried giving the walker a second hidden layer, a deeper brain. I added the new layer so that, at first, it walks almost exactly like the original. But it turned out to be incredibly fragile. Evolution works by trying small random changes, and even the tiniest change made most of the deeper walkers fall over, while the original network shrugs off changes ten times bigger. A search that only tries changes and keeps the good ones can't fine-tune something that fragile. So I dropped it.

What's next? A new run starts from our best walker, with a fall costing three times as much in training. Will even more caution pay, or will it start to slow the walker down? The safety run keeps training alongside it. Thanks for listening.
