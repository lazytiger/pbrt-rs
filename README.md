# pbrt-rs
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