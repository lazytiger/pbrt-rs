use crate::Float;
use std::cmp::Eq;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

#[macro_export]
macro_rules! strip_plus {
    (+ $($rest:expr)+) => {
        $($rest)+
    };
}

#[macro_export]
macro_rules! match_index {
    ($index:ident, $n:expr,[$($indexes:expr),+], [$($arms:expr),+],) => {
        match $index {
            $(i if i == $indexes => $arms,)+
            _ => panic!("Out of index")
        }
    };
    ($index:ident, $n:expr, [$($indexes:expr),*], [$($arms:expr)*], $arm:expr $(;$rest:expr)*) => {
        $crate::match_index!($index, $n+1, [$($indexes,)* $n], [$($arms,)* $arm], $($rest);*)
    };
    ($index:ident, $($arms:expr);+) => {
        $crate::match_index!($index, 0, [], [], $($arms);+)
    }
}

#[macro_export]
macro_rules! make_vector {
    (struct $name:ident, $($field:ident),+) => {

        pub struct $name<T> {
            $($field:T,)+
        }

        impl<T> $name<T> {
            fn new($($field:T),+) ->Self  {
               $name { $($field:$field,)+}
            }
        }

        impl<T: Add<Output = T>> Add for $name<T> {
            type Output = $name<T>;

            fn add(self, rhs: Self) -> Self::Output {
                $name {
                    $($field:self.$field + rhs.$field,)+
                }
            }
        }

        impl<T: AddAssign> AddAssign for $name<T> {
            fn add_assign(&mut self, rhs: Self) {
                $(self.$field += rhs.$field;)+
            }
        }

        impl<T: Sub<Output = T>> Sub for $name<T> {
            type Output = $name<T>;

            fn sub(self, rhs: Self) -> Self::Output {
                $name {
                    $($field:self.$field - rhs.$field,)+
                }
            }
        }

        impl<T: SubAssign> SubAssign for $name<T> {
            fn sub_assign(&mut self, rhs: Self) {
                $(self.$field -= rhs.$field;)+
            }
        }

        impl<T: PartialEq> PartialEq for $name<T> {
            fn eq(&self, other: &Self) -> bool {
                $(self.$field == other.$field)&& +
            }
        }

        impl<T: Mul<Output = T>> Mul for $name<T> {
            type Output = $name<T>;

            fn mul(self, rhs: Self) -> Self::Output {
                $name {
                    $($field:self.$field * rhs.$field,)+
                }
            }
        }

        impl<T: MulAssign> MulAssign for $name<T> {
            fn mul_assign(&mut self, rhs: Self) {
                $(self.$field *= rhs.$field;)+
            }
        }

        impl<T: Div<Output = T>> Div for $name<T> {
            type Output = $name<T>;

            fn div(self, rhs: Self) -> Self::Output {
                $name {
                    $($field:self.$field / rhs.$field,)+
                }
            }
        }

        impl<T: DivAssign> DivAssign for $name<T> {
            fn div_assign(&mut self, rhs: Self) {
                $(self.$field /= rhs.$field;)+
            }
        }

        impl<T: Neg<Output = T>> Neg for $name<T> {
            type Output = $name<T>;

            fn neg(self) -> Self::Output {
                $name {
                    $($field: -self.$field,)+
                }
            }
        }

        impl<T> Index<usize> for $name<T> {
            type Output = T;

            fn index(&self, index: usize) -> &Self::Output {
                $crate::match_index!(index, $(&self.$field);+)
            }
        }

        impl<T> IndexMut<usize> for $name<T> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                $crate::match_index!(index, $(&mut self.$field);+)
            }
        }

        impl<T: Mul<Output = T> + Add<Output = T> + Copy> $name<T> {
            fn length_squared(&self) -> T {
                $crate::strip_plus!($(+ self.$field * self.$field)+)
            }
        }

        impl $name<Float> {
            fn length(&self) -> Float {
                self.length_squared().sqrt()
            }
        }
    };
}
