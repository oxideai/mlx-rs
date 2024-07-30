use mlx_rs::{error::Exception, ops::ClipBound, Array, StreamOrDevice};

pub struct Abs {
    pub stream: Option<StreamOrDevice>,
}

impl Default for Abs {
    fn default() -> Self {
        Self::new()
    }
}

impl Abs {
    pub fn new() -> Self {
        Self { stream: None }
    }

    pub fn stream(mut self, stream: StreamOrDevice) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn build(self, a: impl AsRef<Array>) -> Array {
        match self.stream {
            Some(stream) => mlx_rs::ops::abs_device(a.as_ref(), stream),
            None => mlx_rs::ops::abs(a.as_ref()),
        }
    }
}

pub struct Clip {
    pub stream: Option<StreamOrDevice>,
}

impl Default for Clip {
    fn default() -> Self {
        Self::new()
    }
}

impl Clip {
    pub fn new() -> Self {
        Self { stream: None }
    }

    pub fn stream(mut self, stream: StreamOrDevice) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn build<'min, 'max>(
        self,
        a: impl AsRef<Array>,
        bound: impl ClipBound<'min, 'max>,
    ) -> Result<Array, Exception> {
        match self.stream {
            Some(stream) => mlx_rs::ops::clip_device(a.as_ref(), bound, stream),
            None => mlx_rs::ops::clip(a.as_ref(), bound),
        }
    }
}

pub struct Softmax<'a> {
    pub stream: Option<StreamOrDevice>,
    pub axes: Option<&'a [i32]>,
    pub precise: Option<bool>,
}

impl<'a> Default for Softmax<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Softmax<'a> {
    pub fn new() -> Self {
        Self {
            stream: None,
            axes: None,
            precise: None,
        }
    }

    pub fn stream(mut self, stream: StreamOrDevice) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn axes(mut self, axes: &'a [i32]) -> Self {
        self.axes = Some(axes);
        self
    }

    pub fn precise(mut self, precise: bool) -> Self {
        self.precise = Some(precise);
        self
    }

    pub fn build(self, a: impl AsRef<Array>) -> Array {
        match self.stream {
            Some(stream) => {
                mlx_rs::ops::softmax_device(a.as_ref(), self.axes, self.precise, stream)
            }
            None => mlx_rs::ops::softmax(a.as_ref(), self.axes, self.precise),
        }
    }
}

#[cfg(test)]
mod tests {
    use mlx_rs::{array, StreamOrDevice};

    use super::*;

    #[test]
    fn test_abs() {
        let stream = StreamOrDevice::default();
        let a = array!([-1, -2, -3]);

        let _output = Abs::new().build(&a);
        let _output = Abs::new().stream(stream).build(&a);
    }

    #[test]
    fn test_clip() {
        let stream = StreamOrDevice::default();
        let a = array!([1, 2, 3]);

        let _output = Clip::new().build(&a, (0, 2));
        let _output = Clip::new().stream(stream).build(&a, (0, 2));
    }

    #[test]
    fn test_softmax() {
        let stream = StreamOrDevice::default();
        let a = array!([[1, 2], [3, 4]]);

        let _output = Softmax::new().build(&a);
        let _output = Softmax::new().stream(stream).build(&a);
        let _output = Softmax::new().axes(&[0]).build(&a);
        let _output = Softmax::new().precise(true).build(&a);
        let _output = Softmax::new().axes(&[0]).precise(true).build(&a);
    }
}
