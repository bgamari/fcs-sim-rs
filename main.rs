extern crate rand;
extern crate nalgebra as na;
mod v3;

use rand::distributions::{Distribution};
use std::f64;
use v3::V3;

struct RandomWalk<Rng> {
    rng: Rng,
    diffusivity: f64,
    pos: V3<f64>
}

impl<Rng: rand::Rng> std::iter::Iterator for RandomWalk<Rng> {
    type Item = V3<f64>;
    fn next(&mut self) -> Option<V3<f64>> {
        use rand::distributions::Normal;
        let r = Normal::new(0.0, self.diffusivity).sample(&mut self.rng);
        self.pos = self.pos + OnSphere.sample(&mut self.rng) * r;
        Some(self.pos)
    }
}

struct OnSphere;

impl Distribution<V3<f64>> for OnSphere {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> V3<f64> {
        use std::f64::consts::PI;
        use rand::distributions::Range;
        let phi = rng.sample(Range::new(0.0, PI));
        let theta = rng.sample(Range::new(-PI, PI));
        V3 {
            x: theta.sin() * phi.cos(),
            y: theta.sin() * phi.sin(),
            z: theta.cos()
        }
    }
}

fn beam_intensity(beam_size: V3<f64>, p: V3<f64>) -> f64 {
    let alpha: f64 =
          beam_size.x.powi(2) / p.x.powi(2)
        + beam_size.y.powi(2) / p.y.powi(2)
        + beam_size.z.powi(2) / p.z.powi(2);
    (-alpha / 2.0).exp()
}

fn main() {
    use std::vec::Vec;
    use rand::NewRng;
    let mut rng = rand::SmallRng::new();
    let walk = RandomWalk {
        rng: rng, diffusivity: 1.0, pos: V3::origin()
    };
    let steps: Vec<V3<f64>> = walk.take(10000000).collect();
    let s: V3<f64> = steps.into_iter().sum();
    println!("hello {:?}", s);
    //debug!("Hello {}", steps);
}
