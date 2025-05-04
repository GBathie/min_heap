use std::{
    cmp::Reverse,
    collections::{BinaryHeap, TryReserveError, binary_heap},
    hash::Hash,
    iter::FusedIterator,
};

use peek::PeekMut;

/// A min priority queue implemented as a thin wrapper around [`BinaryHeap<Reverse<T>>`].
///
///
/// # Examples
///
/// ```
/// use min_heap::MinHeap;
///
/// // Type inference lets us omit an explicit type signature (which
/// // would be `MinHeap<i32>` in this example).
/// let mut heap = MinHeap::new();
///
/// // We can use peek to look at the smallest item in the heap. In this case,
/// // there's no items in there yet so we get None.
/// assert_eq!(heap.peek(), None);
///
/// // Let's add some scores...
/// heap.push(1);
/// heap.push(5);
/// heap.push(2);
///
/// // Now peek shows the smallest item in the heap.
/// assert_eq!(heap.peek(), Some(&1));
///
/// // We can check the length of a heap.
/// assert_eq!(heap.len(), 3);
///
/// // We can iterate over the items in the heap, although they are returned in
/// // an unspecified order.
/// for x in &heap {
///     println!("{x}");
/// }
///
/// // If we instead pop these scores, they should come back in order.
/// assert_eq!(heap.pop(), Some(1));
/// assert_eq!(heap.pop(), Some(2));
/// assert_eq!(heap.pop(), Some(5));
/// assert_eq!(heap.pop(), None);
///
/// // We can clear the heap of any remaining items.
/// heap.clear();
///
/// // The heap should now be empty.
/// assert!(heap.is_empty())
/// ```
///
/// A `MinHeap` with a known list of items can be initialized from an array:
///
/// ```
/// use min_heap::MinHeap;
///
/// let heap = MinHeap::from([1, 5, 2]);
/// ```
///
/// # Time complexity
///
/// | [push]  | [pop]         | [peek]/[peek\_mut] |
/// |---------|---------------|--------------------|
/// | *O*(1)~ | *O*(log(*n*)) | *O*(1)             |
///
/// The value for `push` is an expected cost; the method documentation gives a
/// more detailed analysis.
///
/// [`Cell`]: core::cell::Cell
/// [`RefCell`]: core::cell::RefCell
/// [push]: MinHeap::push
/// [pop]: MinHeap::pop
/// [peek]: MinHeap::peek
/// [peek\_mut]: MinHeap::peek_mut
#[derive(Debug, Clone)]
pub struct MinHeap<T> {
    inner: BinaryHeap<Reverse<T>>,
}

pub(crate) mod peek;

impl<T> Default for MinHeap<T>
where
    T: Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> PartialEq for MinHeap<T>
where
    BinaryHeap<Reverse<T>>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<T> Eq for MinHeap<T> where BinaryHeap<Reverse<T>>: Eq {}

impl<T> PartialOrd for MinHeap<T>
where
    BinaryHeap<Reverse<T>>: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}

impl<T> Ord for MinHeap<T>
where
    BinaryHeap<Reverse<T>>: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.inner.cmp(&other.inner)
    }
}

impl<T> Hash for MinHeap<T>
where
    BinaryHeap<Reverse<T>>: Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl<T> MinHeap<T>
where
    T: Ord,
{
    /// Creates an empty min-heap.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let mut heap = MinHeap::new();
    /// heap.push(4);
    /// ```
    #[inline]
    pub const fn new() -> Self {
        Self {
            inner: BinaryHeap::new(),
        }
    }

    /// Creates an empty `MinHeap` with at least the specified capacity.
    ///
    /// The heap will be able to hold at least `capacity` elements without
    /// reallocating. This method is allowed to allocate for more elements than
    /// `capacity`. If `capacity` is zero, the heap will not allocate.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let mut heap = MinHeap::with_capacity(10);
    /// heap.push(4);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        MinHeap {
            inner: BinaryHeap::with_capacity(capacity),
        }
    }

    /// Returns a mutable reference to the smallest item in the heap, or
    /// `None` if it is empty.
    ///
    /// Note: If the `PeekMut` value is leaked, some heap elements might get
    /// leaked along with it, but the remaining elements will remain a valid
    /// heap.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let mut heap = MinHeap::new();
    /// assert!(heap.peek_mut().is_none());
    ///
    /// heap.push(1);
    /// heap.push(5);
    /// heap.push(2);
    /// if let Some(mut val) = heap.peek_mut() {
    ///     *val = 8;
    /// }
    /// assert_eq!(heap.peek(), Some(&2));
    /// ```
    ///
    /// # Time complexity
    ///
    /// If the item is modified then the worst case time complexity is *O*(log(*n*)),
    /// otherwise it's *O*(1).
    #[inline]
    pub fn peek_mut(&mut self) -> Option<PeekMut<'_, T>> {
        self.inner.peek_mut().map(PeekMut::new)
    }

    /// Removes the smallest item from the heap and returns it, or `None` if it
    /// is empty.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let mut heap = MinHeap::from([1, 3]);
    ///
    /// assert_eq!(heap.pop(), Some(1));
    /// assert_eq!(heap.pop(), Some(3));
    /// assert_eq!(heap.pop(), None);
    /// ```
    ///
    /// # Time complexity
    ///
    /// The worst case cost of `pop` on a heap containing *n* elements is *O*(log(*n*)).
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.inner.pop().map(|v| v.0)
    }

    /// Pushes an item onto the heap.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let mut heap = MinHeap::new();
    /// heap.push(5);
    /// heap.push(1);
    /// heap.push(3);
    ///
    /// assert_eq!(heap.len(), 3);
    /// assert_eq!(heap.peek(), Some(&1));
    /// ```
    ///
    /// # Time complexity
    ///
    /// The expected cost of `push`, averaged over every possible ordering of
    /// the elements being pushed, and over a sufficiently large number of
    /// pushes, is *O*(1). This is the most meaningful cost metric when pushing
    /// elements that are *not* already in any sorted pattern.
    ///
    /// The time complexity degrades if elements are pushed in predominantly
    /// decreasing order. In the worst case, elements are pushed in decreasing
    /// sorted order and the amortized cost per push is *O*(log(*n*)) against a heap
    /// containing *n* elements.
    ///
    /// The worst case cost of a *single* call to `push` is *O*(*n*). The worst case
    /// occurs when capacity is exhausted and needs a resize. The resize cost
    /// has been amortized in the previous figures.
    #[inline]
    pub fn push(&mut self, item: T) {
        self.inner.push(Reverse(item));
    }

    /// Consumes the `MinHeap` and returns a vector in sorted
    /// (ascending) order.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    ///
    /// let mut heap = MinHeap::from([5, 2, 4, 1, 7]);
    /// heap.push(6);
    /// heap.push(3);
    ///
    /// let vec = heap.into_sorted_vec();
    /// assert_eq!(vec, [1, 2, 3, 4, 5, 6, 7]);
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    pub fn into_sorted_vec(self) -> Vec<T> {
        let vec = self.inner.into_sorted_vec();
        let mut vec: Vec<T> = vec.into_iter().map(|v| v.0).collect();
        vec.reverse();
        vec
    }

    /// Moves all the elements of `other` into `self`, leaving `other` empty.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    ///
    /// let mut a = MinHeap::from([-10, 1, 2, 3, 3]);
    /// let mut b = MinHeap::from([-20, 5, 43]);
    ///
    /// a.append(&mut b);
    ///
    /// assert_eq!(a.into_sorted_vec(), [-20, -10, 1, 2, 3, 3, 5, 43]);
    /// assert!(b.is_empty());
    /// ```
    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        self.inner.append(&mut other.inner);
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` for which `f(&e)` returns
    /// `false`. The elements are visited in unsorted (and unspecified) order.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    ///
    /// let mut heap = MinHeap::from([-10, -5, 1, 2, 4, 13]);
    ///
    /// heap.retain(|x| x % 2 == 0); // only keep even numbers
    ///
    /// assert_eq!(heap.into_sorted_vec(), [-10, 2, 4])
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        let g = |v: &Reverse<T>| f(&v.0);
        self.inner.retain(g);
    }
}

impl<T> MinHeap<T> {
    /// Returns an iterator visiting all values in the underlying vector, in
    /// arbitrary order.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let heap = MinHeap::from([1, 2, 3, 4]);
    ///
    /// // Print 1, 2, 3, 4 in arbitrary order
    /// for x in heap.iter() {
    ///     println!("{x}");
    /// }
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        IterRefWrapper {
            inner: self.inner.iter(),
        }
    }

    /// Returns the smallest item in the heap, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let mut heap = MinHeap::new();
    /// assert_eq!(heap.peek(), None);
    ///
    /// heap.push(1);
    /// heap.push(5);
    /// heap.push(2);
    /// assert_eq!(heap.peek(), Some(&1));
    ///
    /// ```
    ///
    /// # Time complexity
    ///
    /// Cost is *O*(1) in the worst case.
    #[inline]
    #[must_use]
    pub fn peek(&self) -> Option<&T> {
        self.inner.peek().map(|v| &v.0)
    }

    /// Returns the number of elements the heap can hold without reallocating.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let mut heap = MinHeap::with_capacity(100);
    /// assert!(heap.capacity() >= 100);
    /// heap.push(4);
    /// ```
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Reserves the minimum capacity for at least `additional` elements more than
    /// the current length. Unlike [`reserve`], this will not
    /// deliberately over-allocate to speculatively avoid frequent allocations.
    /// After calling `reserve_exact`, capacity will be greater than or equal to
    /// `self.len() + additional`. Does nothing if the capacity is already
    /// sufficient.
    ///
    /// [`reserve`]: MinHeap::reserve
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows [`usize`].
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let mut heap = MinHeap::new();
    /// heap.reserve_exact(100);
    /// assert!(heap.capacity() >= 100);
    /// heap.push(4);
    /// ```
    ///
    /// [`reserve`]: MinHeap::reserve
    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.inner.reserve_exact(additional);
    }

    /// Reserves capacity for at least `additional` elements more than the
    /// current length. The allocator may reserve more space to speculatively
    /// avoid frequent allocations. After calling `reserve`,
    /// capacity will be greater than or equal to `self.len() + additional`.
    /// Does nothing if capacity is already sufficient.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows [`usize`].
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let mut heap = MinHeap::new();
    /// heap.reserve(100);
    /// assert!(heap.capacity() >= 100);
    /// heap.push(4);
    /// ```
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    /// Tries to reserve the minimum capacity for at least `additional` elements
    /// more than the current length. Unlike [`try_reserve`], this will not
    /// deliberately over-allocate to speculatively avoid frequent allocations.
    /// After calling `try_reserve_exact`, capacity will be greater than or
    /// equal to `self.len() + additional` if it returns `Ok(())`.
    /// Does nothing if the capacity is already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it
    /// requests. Therefore, capacity can not be relied upon to be precisely
    /// minimal. Prefer [`try_reserve`] if future insertions are expected.
    ///
    /// [`try_reserve`]: MinHeap::try_reserve
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// use std::collections::TryReserveError;
    ///
    /// fn find_min_slow(data: &[u32]) -> Result<Option<u32>, TryReserveError> {
    ///     let mut heap = MinHeap::new();
    ///
    ///     // Pre-reserve the memory, exiting if we can't
    ///     heap.try_reserve_exact(data.len())?;
    ///
    ///     // Now we know this can't OOM in the middle of our complex work
    ///     heap.extend(data.iter());
    ///
    ///     Ok(heap.pop())
    /// }
    /// # find_min_slow(&[1, 2, 3]).expect("why is the test harness OOMing on 12 bytes?");
    /// ```
    #[inline]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve_exact(additional)
    }

    /// Tries to reserve capacity for at least `additional` elements more than the
    /// current length. The allocator may reserve more space to speculatively
    /// avoid frequent allocations. After calling `try_reserve`, capacity will be
    /// greater than or equal to `self.len() + additional` if it returns
    /// `Ok(())`. Does nothing if capacity is already sufficient. This method
    /// preserves the contents even if an error occurs.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// use std::collections::TryReserveError;
    ///
    /// fn find_min_slow(data: &[u32]) -> Result<Option<u32>, TryReserveError> {
    ///     let mut heap = MinHeap::new();
    ///
    ///     // Pre-reserve the memory, exiting if we can't
    ///     heap.try_reserve(data.len())?;
    ///
    ///     // Now we know this can't OOM in the middle of our complex work
    ///     heap.extend(data.iter());
    ///
    ///     Ok(heap.pop())
    /// }
    /// # find_min_slow(&[1, 2, 3]).expect("why is the test harness OOMing on 12 bytes?");
    /// ```
    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve(additional)
    }

    /// Discards as much additional capacity as possible.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let mut heap: MinHeap<i32> = MinHeap::with_capacity(100);
    ///
    /// assert!(heap.capacity() >= 100);
    /// heap.shrink_to_fit();
    /// assert!(heap.capacity() == 0);
    /// ```
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit();
    }

    /// Discards capacity with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length
    /// and the supplied value.
    ///
    /// If the current capacity is less than the lower limit, this is a no-op.
    ///
    /// # Examples
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let mut heap: MinHeap<i32> = MinHeap::with_capacity(100);
    ///
    /// assert!(heap.capacity() >= 100);
    /// heap.shrink_to(10);
    /// assert!(heap.capacity() >= 10);
    /// ```
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.inner.shrink_to(min_capacity)
    }

    /// Returns a slice of all values in the underlying vector, in arbitrary
    /// order.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// use std::io::{self, Write};
    ///
    /// let heap = MinHeap::from([1, 2, 3, 4, 5, 6, 7]);
    ///
    /// io::sink().write(heap.as_slice()).unwrap();
    /// ```
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        let inner_slice = self.inner.as_slice();
        unsafe {
            // SAFETY: Reverse<T> is #[repr(transparent)], so transmuting it to T is safe.
            std::mem::transmute(inner_slice)
        }
    }

    /// Consumes the `MinHeap` and returns the underlying vector
    /// in arbitrary order.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let heap = MinHeap::from([1, 2, 3, 4, 5, 6, 7]);
    /// let vec = heap.into_vec();
    ///
    /// // Will print in some order
    /// for x in vec {
    ///     println!("{x}");
    /// }
    /// ```
    #[inline]
    #[must_use = "`self` will be dropped if the result is not used"]
    pub fn into_vec(self) -> Vec<T> {
        self.inner.into_vec().into_iter().map(|v| v.0).collect()
    }

    /// Returns the length of the heap.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let heap = MinHeap::from([1, 3]);
    ///
    /// assert_eq!(heap.len(), 2);
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Checks if the heap is empty.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let mut heap = MinHeap::new();
    ///
    /// assert!(heap.is_empty());
    ///
    /// heap.push(3);
    /// heap.push(5);
    /// heap.push(1);
    ///
    /// assert!(!heap.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clears the heap, returning an iterator over the removed elements
    /// in arbitrary order. If the iterator is dropped before being fully
    /// consumed, it drops the remaining elements in arbitrary order.
    ///
    /// The returned iterator keeps a mutable borrow on the heap to optimize
    /// its implementation.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let mut heap = MinHeap::from([1, 3]);
    ///
    /// assert!(!heap.is_empty());
    ///
    /// for x in heap.drain() {
    ///     println!("{x}");
    /// }
    ///
    /// assert!(heap.is_empty());
    /// ```
    #[inline]
    pub fn drain(&mut self) -> Drain<'_, T> {
        IterWrapper {
            inner: self.inner.drain(),
        }
    }

    /// Drops all items from the heap.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let mut heap = MinHeap::from([1, 3]);
    ///
    /// assert!(!heap.is_empty());
    ///
    /// heap.clear();
    ///
    /// assert!(heap.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.drain();
    }
}

/// Helper struct that wraps an iterator over items of type [`Reverse<T>`]
/// and converts it to items of type `T`.
///
/// Used for [`Drain`] and [`IntoIter`].
pub struct IterWrapper<It, T>
where
    It: Iterator<Item = Reverse<T>>,
{
    inner: It,
}

impl<It, T> Iterator for IterWrapper<It, T>
where
    It: Iterator<Item = Reverse<T>>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|v| v.0)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<It, T> DoubleEndedIterator for IterWrapper<It, T>
where
    It: Iterator<Item = Reverse<T>>,
    It: DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.inner.next_back().map(|v| v.0)
    }
}

impl<It, T> ExactSizeIterator for IterWrapper<It, T>
where
    It: Iterator<Item = Reverse<T>>,
    It: ExactSizeIterator,
{
}

impl<It, T> FusedIterator for IterWrapper<It, T>
where
    It: Iterator<Item = Reverse<T>>,
    It: FusedIterator,
{
}

impl<It, T> Default for IterWrapper<It, T>
where
    It: Iterator<Item = Reverse<T>>,
    It: Default,
{
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

/// An owning iterator over the elements of a [`MinHeap`].
///
/// This `struct` is created by [`MinHeap::into_iter()`]
/// (provided by the [`IntoIterator`] trait). See its documentation for more.
pub type IntoIter<T> = IterWrapper<binary_heap::IntoIter<Reverse<T>>, T>;
/// A draining iterator over the elements of a [`MinHeap`].
///
/// This `struct` is created by [`MinHeap::drain()`]. See its
/// documentation for more.
pub type Drain<'a, T> = IterWrapper<binary_heap::Drain<'a, Reverse<T>>, T>;

/// Helper struct that wraps an iterator over items of type [`&Reverse<T>`](std::cmp::Reverse)
/// and converts it to items of type `&T`.
///
/// Used for [`Iter`].
pub struct IterRefWrapper<'a, It, T>
where
    It: Iterator<Item = &'a Reverse<T>>,
    T: 'a,
{
    inner: It,
}

impl<'a, It, T> Iterator for IterRefWrapper<'a, It, T>
where
    It: Iterator<Item = &'a Reverse<T>>,
    T: 'a,
{
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|v| &v.0)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, It, T> DoubleEndedIterator for IterRefWrapper<'a, It, T>
where
    It: Iterator<Item = &'a Reverse<T>>,
    T: 'a,
    It: DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        self.inner.next_back().map(|v| &v.0)
    }
}

impl<'a, It, T> ExactSizeIterator for IterRefWrapper<'a, It, T>
where
    It: Iterator<Item = &'a Reverse<T>>,
    T: 'a,
    It: ExactSizeIterator,
{
}

impl<'a, It, T> FusedIterator for IterRefWrapper<'a, It, T>
where
    It: Iterator<Item = &'a Reverse<T>>,
    T: 'a,
    It: FusedIterator,
{
}

impl<'a, It, T> Default for IterRefWrapper<'a, It, T>
where
    It: Iterator<Item = &'a Reverse<T>>,
    T: 'a,
    It: Default,
{
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

/// An iterator over the elements of a [`MinHeap`].
///
/// This `struct` is created by [`MinHeap::iter()`]. See its
/// documentation for more.
pub type Iter<'a, T> = IterRefWrapper<'a, binary_heap::Iter<'a, Reverse<T>>, T>;

impl<T: Ord> From<Vec<T>> for MinHeap<T> {
    /// Converts a `Vec<T>` into a `MinHeap<T>`.
    ///
    /// This conversion happens in-place, and has *O*(*n*) time complexity.
    ///
    /// # Example
    ///
    /// ```
    /// use min_heap::MinHeap;
    ///
    /// let v = vec![3, 12, 5, 6, 9];
    /// let mut h: MinHeap<_> = v.into();
    /// assert_eq!(h.pop(), Some(3));
    /// assert_eq!(h.pop(), Some(5));
    /// assert_eq!(h.pop(), Some(6));
    /// assert_eq!(h.pop(), Some(9));
    /// assert_eq!(h.pop(), Some(12));
    /// assert_eq!(h.pop(), None);
    /// ```
    fn from(vec: Vec<T>) -> MinHeap<T> {
        MinHeap {
            inner: vec.into_iter().map(|x| Reverse(x)).collect(),
        }
    }
}

impl<T: Ord, const N: usize> From<[T; N]> for MinHeap<T> {
    /// ```
    /// use min_heap::MinHeap;
    ///
    /// let mut h1 = MinHeap::from([1, 4, 2, 3]);
    /// let mut h2: MinHeap<_> = [1, 4, 2, 3].into();
    /// while let Some((a, b)) = h1.pop().zip(h2.pop()) {
    ///     assert_eq!(a, b);
    /// }
    /// ```
    fn from(arr: [T; N]) -> Self {
        Self::from_iter(arr)
    }
}

impl<T> From<MinHeap<T>> for Vec<T> {
    /// Converts a `MinHeap<T>` into a `Vec<T>`.
    /// The order of the elements in the resulting `Vec<T>` is unspecified (most likely unsorted).
    ///
    /// This conversion requires no data movement or allocation, and has
    /// constant time complexity.
    fn from(heap: MinHeap<T>) -> Vec<T> {
        let vec: Vec<_> = heap.inner.into();
        vec.into_iter().map(|v| v.0).collect()
    }
}

impl<T: Ord> FromIterator<T> for MinHeap<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> MinHeap<T> {
        MinHeap {
            inner: iter.into_iter().map(|x| Reverse(x)).collect(),
        }
    }
}

impl<T> IntoIterator for MinHeap<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    /// Creates a consuming iterator, that is, one that moves each value out of
    /// the heap in arbitrary order. The heap cannot be used
    /// after calling this.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use min_heap::MinHeap;
    /// let heap = MinHeap::from([1, 2, 3, 4]);
    ///
    /// // Print 1, 2, 3, 4 in arbitrary order
    /// for x in heap.into_iter() {
    ///     // x has type i32, not &i32
    ///     println!("{x}");
    /// }
    /// ```
    fn into_iter(self) -> IntoIter<T> {
        IntoIter {
            inner: self.inner.into_iter(),
        }
    }
}

impl<'a, T> IntoIterator for &'a MinHeap<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<T: Ord> Extend<T> for MinHeap<T> {
    fn extend<It: IntoIterator<Item = T>>(&mut self, iter: It) {
        self.inner.extend(iter.into_iter().map(|x| Reverse(x)));
    }
}

impl<'a, T: 'a + Ord + Copy> Extend<&'a T> for MinHeap<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned());
    }
}

#[cfg(test)]
mod test {
    use std::{cmp::Reverse, collections::BinaryHeap};

    use super::MinHeap;

    #[test]
    fn min_heap_pops_min() {
        let mut mh: MinHeap<i32> = rand::random_iter().take(100).collect();

        while !mh.is_empty() {
            let min = mh.iter().min().copied();
            let pop = mh.pop();
            assert_eq!(min, pop, "MinHeap popped an item not equal to the min")
        }
    }

    #[test]
    fn min_heap_is_reverse_binary_heap() {
        let v: Vec<i32> = rand::random_iter().take(100).collect();
        let mut h: BinaryHeap<Reverse<i32>> = v
            .iter()
            .copied()
            .map(|x| Reverse(x))
            .collect::<Vec<_>>()
            .into();
        let mut mh: MinHeap<i32> = v.into_iter().collect();

        loop {
            match (h.pop(), mh.pop()) {
                (Some(Reverse(a)), Some(b)) => assert_eq!(a, b),
                (None, Some(_)) => panic!("More elements in MinHeap than in BinaryHeap!"),
                (Some(_), None) => panic!("More elements in BinaryHeap than in MinHeap!"),
                (None, None) => break,
            }
        }
    }
}
