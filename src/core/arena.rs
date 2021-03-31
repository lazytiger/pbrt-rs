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

    pub fn get(&self, offset: usize) -> &T {
        &self.data[offset]
    }

    pub fn get_mut(&mut self, offset: usize) -> &mut T {
        &mut self.data[offset]
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
