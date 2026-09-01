**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* T Fong
  * [LinkedIn](https://www.linkedin.com/in/tzfong/), [personal website](https://www.tzfong.com/)
* Tested on: Windows 11, Intel Ultra9 275HX (2.70 GHz), RTX 5070 Mobile, 32 GB RAM (2800 MHz)

### What's a boid?
[Boids](https://en.wikipedia.org/wiki/Boids) are bird-like particles developed by Craig Reynolds in 1986. They're bird-like in that they exhibit flocking behavior by following 3 rules:  
* Cohesion: boids move towards a local center of mass.
* Separation: boids try to keep a distance from neighboring boids.
* Alignment: boids try to match the velocities of neighboring boids.

These are labeled rules 1, 2, and 3 respectively in my implementation, and the concept of "local" and "neighboring" is controlled by the radii `rule1Distance`, `rule2Distance`, and `rule3Distance`.

### Naïve simulation (brute force)
In implementation, at each time step, each boid will examine the positions and velocities of other boids at the previous time step, and update its own velocity according to the rules outlined above. We then update the positions of all the boids by moving it according to its current velocity. This is pretty clearly quite parallelizable, and therefore something that can be accelerated on the GPU.

We'll start with 3 buffers on the GPU, `pos`, `vel1`, and `vel2`. `vel1` represents the velocities of the boids at the previous time step, which we read from, and `vel2` represents the velocities of the boids at the current time step. We then swap `vel2` and `vel1`, called "ping-ponging", because the current `vel2` becomes the next iteration's `vel1`.

It's easy to get a naïve simulation up and running; each boid iterates over all other boids (in my testing, I mostly stuck with 100,000 boids). This is wasteful, because most of the boids will not be within the radius of the rules, but not incorrect.

**TODO: Finish writing**

### Using a spatial grid
We can get a much better result by iterating over a narrower number of boids that are more likely to be neighbors. By dividing the simulation space into a spatial grid, we can iterate over the grid cells that have some part contained within the radius of our boid.

**TODO: Finish**

### Semi-coherent memory access in the spatial grid
We can further optimize performance by making our memory reads semi-coherent. Instead of chasing pointers to unsorted position and velocity buffers, we sort them. There are two different ways I went about this, and I was very surprised by the results, which I'll cover in a second.

#### Zip-iterators
At least to me, it was conceptually easiest to improve upon the previous approach by using `pos`, `vel1` as the values, rather than `particleArrayIndices`. Thrust provides a way to do this: after wrapping `pos`, `vel1` into device pointers, we can wrap them into a `thrust::zip_iterator`, and use that as the value for the unstable sort.

#### Gather
Alternatively, we can sort `particleArrayIndices`, then add the extra step of shuffling `pos`, `vel1` to match that. Thrust also provides an implementation of this idea, which as you may have guessed, is called `thrust::gather`. I only learned about that after I had already made my own kernel implementing it, which from a very limited amount of testing seemed to perform a little better than the 2 calls to `thrust::gather` I would need to shuffled `pos`, `vel1`, so I used my kernel when benchmarking. Neither does the shuffle in-place, though, so I allocated `pos2` to shuffle `pos` into.

Let's take a look at the numbers now. As a fun exercise, take a guess of how you think the approaches stack up.

### A note about benchmarking
Fps benchmarks were done by taking the average fps over a window of `seconds = 20` on my laptop's maximum performance setting. I noticed that there tended to be a spike at the beginning of execution, so I included a `warmup = 2`.

### Benchmarks
Here's an fps performance comparison between the four methods I outlined earlier.
![buffers for generating a uniform grid using index sort](images/Benchmarks1.png)

**TODO: Investigate source of difference between zip iterator and coherent**

### Testing out different grid cell sizes

### Caching with shared memory

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)
