# pbrt-rs

[![Build Status](https://travis-ci.com/lazytiger/pbrt-rs.svg?branch=master)](https://travis-ci.com/lazytiger/pbrt-rs)
[![GitHub issues](https://img.shields.io/github/issues/lazytiger/pbrt-rs)](https://github.com/lazytiger/pbrt-rs/issues)
[![GitHub license](https://img.shields.io/github/license/lazytiger/pbrt-rs)](https://github.com/lazytiger/pbrt-rs/blob/master/LICENSE)
[![Releases](https://img.shields.io/github/v/release/lazytiger/pbrt-rs.svg?include_prereleases)](https://github.com/lazytiger/pbrt-rs/releases)

This is rust language port based on pbrt project.

### Operators in C++ and How I implement in Rust

C++|Rust
---|----
operator() | From
operator+ | Add
operator+= | AddAssign
operator- | Sub
operator-= | SubAssign
operator* | Mul
operator*= | MulAssign
operator/ | Div
operator/= | DivAssign
operator < | PartialOrd
operator == | PartialEq

### Inheritance in Rust and C++
Essentially, Rust is not an OOP language, so inheritance can't be fully implemented in rust, it can only be simulated.
Inheritance has two key features, composition and polymorphism. 
* Composition can be achieved by Deref trait, and the following code shows that.
```Rust
[test]
fn deref_test() {
struct Base {}

    impl Base {
        fn say_hello(&self) {
            println!("Hello");
        }
    }

    struct Upper {
        base: Base,
    }

    impl Upper {
        fn say_world(&self) {
            println!("World");
        }

        fn say_hello(&self) {
            println!("Hello, World");
        }
    }

    impl Deref for Upper {
        type Target = Base;

        fn deref(&self) -> &Self::Target {
            &self.base
        }
    }

    let u = Upper { base: Base {} };
    u.say_hello();
    u.say_world();
}
```
You can find more detail examples in core::interaction module.

* Polymorphism can be achieved by trait object. 
So if we want some polymorphism we should always extract a trait,
and implement it for all opaque types. You can find some example in shapes module.
  
* Generics is another approach when inheritance occurred in parameters. We can add a trait bound for functions like this
```Rust
pub trait Primitive {
    fn world_bound(&self) -> Bounds3f;
    fn intersect<T: Shape, P: Primitive>(&self, r: &Ray, si: &mut SurfaceInteraction<T, P>)
        -> bool;
    fn intersect_p(&self, r: &Ray) -> bool;
}
```