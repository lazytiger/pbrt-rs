use std::{
    ops::{Index, IndexMut},
    sync::{Arc, RwLock},
};

pub struct Arena<T> {
    data: Vec<T>,
}

pub trait Indexed {
    fn index(&self) -> usize;
    fn set_index(&mut self, index: usize);
}

impl<T: Indexed> Arena<T> {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            data: Vec::with_capacity(n),
        }
    }

    pub fn alloc(&mut self, mut t: T) -> (usize, &mut T) {
        let offset = self.data.len();
        t.set_index(offset);
        self.data.push(t);
        (offset, &mut self.data[offset])
    }

    pub fn get(&self, offset: usize) -> Option<&T> {
        self.data.get(offset)
    }

    pub fn get_mut(&mut self, offset: usize) -> Option<&mut T> {
        self.data.get_mut(offset)
    }

    pub fn alloc_extend<I>(&mut self, i: I) -> usize
    where
        I: IntoIterator<Item = T>,
    {
        let offset = self.data.len();
        for t in i {
            self.alloc(t);
        }
        offset
    }
}

pub type ArenaRw<T> = Arc<RwLock<Arena<T>>>;

#[derive(Default, Clone)]
pub struct BlockedArray<T, const LOG_BLOCK_SIZE: usize> {
    data: Vec<T>,
    u_res: usize,
    v_res: usize,
    u_blocks: usize,
}

impl<T: Clone + Default, const LOG_BLOCK_SIZE: usize> BlockedArray<T, LOG_BLOCK_SIZE> {
    pub fn new(u_res: usize, v_res: usize, d: Option<&Vec<T>>) -> Self {
        let n_alloc = Self::round_up(u_res) * Self::round_up(v_res);
        let u_blocks = Self::round_up(u_res) >> LOG_BLOCK_SIZE;
        let mut data = vec![T::default(); n_alloc];
        let mut ba = Self {
            u_res,
            v_res,
            u_blocks,
            data,
        };
        if let Some(d) = d {
            for v in 0..v_res {
                for u in 0..u_res {
                    ba[(u, v)] = d[v * u_res + u].clone();
                }
            }
        }
        ba
    }

    pub fn block_size() -> usize {
        1 << LOG_BLOCK_SIZE
    }

    pub fn round_up(x: usize) -> usize {
        1 << LOG_BLOCK_SIZE
    }

    pub fn u_size(&self) -> usize {
        self.u_res
    }

    pub fn v_size(&self) -> usize {
        self.v_res
    }

    pub fn block(a: usize) -> usize {
        a >> LOG_BLOCK_SIZE
    }

    pub fn offset(a: usize) -> usize {
        a & (Self::block_size() - 1)
    }

    fn index(&self, u: usize, v: usize) -> usize {
        let bu = Self::block(u);
        let bv = Self::block(v);
        let ou = Self::offset(u);
        let ov = Self::offset(v);
        let mut offset = Self::block_size() * Self::block_size() * (self.u_blocks * bv + bu);
        offset += Self::block_size() * ov + ou;
        offset
    }
}

impl<T: Clone + Default, const LOG_BLOCK_SIZE: usize> Index<(usize, usize)>
    for BlockedArray<T, LOG_BLOCK_SIZE>
{
    type Output = T;

    fn index(&self, (u, v): (usize, usize)) -> &Self::Output {
        let index = self.index(u, v);
        &self.data[index]
    }
}

impl<T: Clone + Default, const LOG_BLOCK_SIZE: usize> IndexMut<(usize, usize)>
    for BlockedArray<T, LOG_BLOCK_SIZE>
{
    fn index_mut(&mut self, (u, v): (usize, usize)) -> &mut Self::Output {
        let index = self.index(u, v);
        &mut self.data[index]
    }
}
