use std::rc::Rc;

struct UnityBinding {
    update: fn(delta: f32),
}

static mut BINDING: Option<Rc<UnityBinding>> = None;

fn get_binding() -> &'static UnityBinding {
    unsafe { BINDING.as_ref().unwrap() }
}

fn call_update() {
    (get_binding().update)(0.0);
}

#[test]
fn test() {
    std::thread::spawn(|| {
        unsafe {
            (BINDING.as_ref().unwrap().update)(0.0);
        }
        //call_update();
    });
}
