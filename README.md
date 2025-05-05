# `MinHeap` &emsp; [![Build Status]][actions] [![Latest Version]][crates.io]

[Build Status]: https://img.shields.io/github/actions/workflow/status/GBathie/min_heap/rust.yml?branch=main
[actions]: https://github.com/GBathie/min_heap/actions?query=branch%3Amain
[Latest Version]: https://img.shields.io/crates/v/min-heap.svg
[crates.io]: https://crates.io/crates/min-heap

**MinHeap: A Min Priority Queue implemented as a Thin Wrapper around `BinaryHeap<Reverse<T>>` from the standard library.**

---

## Quickstart

Add min_heap as a dependency: run `cargo add min_heap`, or add the following to your `Cargo.toml` file.
```toml
[dependencies]
min_heap = "1.0"
```

`min_heap::MinHeap` mirrors the API of `std::collections::BinaryHeap`; here is an overview of what you can do with it.
```rs
use min_heap::MinHeap;
// Type inference lets us omit an explicit type signature (which
// would be `MinHeap<i32>` in this example).
let mut heap = MinHeap::new();

// We can use peek to look at the smallest item in the heap. In this case,
// there's no items in there yet so we get None.
assert_eq!(heap.peek(), None);

// Let's add some numbers...
heap.push(1);
heap.push(5);
heap.push(2);

// Now peek shows the smallest item in the heap.
assert_eq!(heap.peek(), Some(&1));

// We can check the length of a heap.
assert_eq!(heap.len(), 3);

// We can iterate over the items in the heap, although they are returned in
// an unspecified order.
for x in &heap {
    println!("{x}");
}

// If we instead pop these scores, they should come back in order.
assert_eq!(heap.pop(), Some(1));
assert_eq!(heap.pop(), Some(2));
assert_eq!(heap.pop(), Some(5));
assert_eq!(heap.pop(), None);

// We can clear the heap of any remaining items.
heap.clear();

// The heap should now be empty.
assert!(heap.is_empty())
```


## When to use `MinHeap`?

By default, the [`BinaryHeap`](https://doc.rust-lang.org/std/collections/binary_heap/struct.BinaryHeap.html) struct from `std::collections` is a *max-heap*, i.e. it provides efficient access and update to the largest element in a collection.
To use it as a min-heap, [one can either use `core::cmp::Reverse` or a custom `Ord` implementation](https://doc.rust-lang.org/std/collections/binary_heap/struct.BinaryHeap.html#min-heap), but this usually adds a lot of boilerplate to your code. This crate implements this boilerplate so that you never have to write it yourself again! It also allows you to use the `derive`d `Ord` implementation instead of manually reversing it.

Here is a comparison of code without and with `min_heap::MinHeap`.

#### `Reverse`

```rs
// Without `MinHeap`
let mut heap = BinaryHeap::new();

heap.push(Reverse(1));
heap.push(Reverse(3));
heap.push(Reverse(5));

if let Some(Reverse(x)) = heap.pop() {
    println!("Min is {x}");
}

let heap_items: Vec<_> = heap.into_iter().map(|Reverse(x)| x).collect();
```


```rs
// With `MinHeap`
let mut heap = MinHeap::new();

heap.push(1);
heap.push(3);
heap.push(5);

if let Some(x) = heap.pop() {
    println!("Min is {x}");
}

let heap_items: Vec<_> = heap.into_iter().collect();
```

#### Custom `Ord` implementation

```rs
// Without `MinHeap`
#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    cost: usize,
    position: usize,
}

// The priority queue depends on `Ord`.
// Explicitly implement the trait so the queue becomes a min-heap
// instead of a max-heap.
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Notice that we flip the ordering on costs.
        // In case of a tie we compare positions - this step is necessary
        // to make implementations of `PartialEq` and `Ord` consistent.
        other.cost.cmp(&self.cost)
            .then_with(|| self.position.cmp(&other.position))
    }
}

// `PartialOrd` needs to be implemented as well.
impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn main() {
    let mut h = BinaryHeap::new();

    h.push(State { cost: 5, position: 1});
    h.push(State { cost: 3, position: 2});

    assert_eq!(h.pop(), Some(State {cost: 3, position: 2}))
}
```


```rs
// With `MinHeap`: we can use the derived implementation!
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct State {
    cost: usize,
    position: usize,
}

fn main() {
    let mut h = MinHeap::new();

    h.push(State { cost: 5, position: 1});
    h.push(State { cost: 3, position: 2});

    assert_eq!(h.pop(), Some(State {cost: 3, position: 2}))
}
```

## Comparison with other min-heap crates

All other popular crates that provide min-heaps implementations are forks or reimplementations of `BinaryHeap` from the standard library, and some of them are no longer maintained. This crate provides a wrapper around the battle-tested `BinaryHeap` from `std::collections`, therefore it benefits from its updates and bug fixes. 

## Missing features

The unstable (nightly) features of `BinaryHeap`, such as the allocator API, are not available (yet!) in `min_heap`. 

#### License

<sup>
Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in min_heap by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
</sub>