# Episode 2: Stuck between standing and stumbling, and a set of training wheels

Welcome back. This is episode two, recorded on Wednesday around lunchtime. In the last episode, the curriculum's first attempt had learned to tap its feet instead of stepping. Here's what happened next.

Two more attempts ran this morning, and they failed in opposite ways.

Attempt two balanced with its feet close together. Then it went into the stepping stage, where only real steps count. The foot coming down has to be the front foot, and the legs have to take turns. It found the safest answer. It stood still. About one front-foot change per episode, almost no falls, and a search that stopped exploring. The reason is simple. A step that ends in a fall costs a hundred points, plus everything it would have earned by staying up. Standing still earns about eighty five. So standing wins.

Attempt three started from the other side: the walker that already alternates its feet, left, right, left, about seventeen times per run, but falls after a few steps. I hoped the stepping stage would teach it not to fall. It didn't. It kept the stumble, and it fell on all twenty test tracks, through both the stepping stage and the walking stage.

So we have a gap. On one side, a walker that stands safely. On the other, a walker that steps but always falls. And the thing we want, stepping without falling, sits in the valley between them. Evolution only looks nearby, so neither side can cross.

The usual answer in robotics is training wheels. So that's attempt four. During training, a small invisible torque pushes the body back upright whenever it tilts. It isn't part of the score. It just stops the walker from tipping over while it learns the rhythm of walking. Then the support fades in stages: strength two, then one, then a half, then nothing at all. The last stage is walking fast in the real game, with no help.

The first signs are good. With full support, the stumbler never falls. And after only eighteen generations, the best candidate made about eighty real crossing steps in one episode. The feet were sharing the lead, with foot two in front about sixty percent of the time. And it walked forward, instead of stepping in place.

I want to be careful here, though. A walker with training wheels can learn to lean on them. The real test comes when the support is turned down. If the gait survives at strength one, and then a half, and then nothing, we'll have what you asked for. If it falls apart, I'll know exactly which step was too big, and I can add a smaller one in between.

You can follow every stage on the curriculum page, with replays of each champion and the exact reward it's trained on. I'll be back with the next episode when one of these stages passes, or fails clearly. Thanks for listening.
