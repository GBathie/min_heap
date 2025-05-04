use std::{
    cmp::Reverse,
    ops::{Deref, DerefMut},
};
/// Structure wrapping a mutable reference to the smallest item in a
/// [`MinHeap`](super::MinHeap).
///
/// This `struct` is created by the [`MinHeap::peek_mut`](super::MinHeap::peek_mut). See
/// its documentation for more.
pub struct PeekMut<'a, T: Ord> {
    underlying: std::collections::binary_heap::PeekMut<'a, Reverse<T>>,
}

impl<'a, T: Ord> PeekMut<'a, T> {
    pub(super) fn new(underlying: std::collections::binary_heap::PeekMut<'a, Reverse<T>>) -> Self {
        Self { underlying }
    }
}

impl<T: Ord> Deref for PeekMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.underlying.0
    }
}

impl<T: Ord> DerefMut for PeekMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.underlying.0
    }
}
